import numpy as np
import torch
import glob
import cv2
import tifffile as tff  # For reading TIFF files if any raw data is in TIFF
import nibabel as nib  # For reading NIfTI files if any raw data is in NIfTI
import os
import zarr  # Import zarr library
from typing import List, Tuple
from tqdm import tqdm


class EMDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading 3D EM volumes and their labels from Zarr files on-demand.
    Supports multiple Zarr files across directories, random patch extraction,
    and on-the-fly preprocessing including optional downsampling.
    """

    def __init__(self,
                 zarr_data_dirs: List[str],
                 label_type: str,  # 'neuron' or 'mito'
                 patch_size: Tuple[int, int, int] = (32, 32, 32),  # Isotropic patch size (Z, Y, X)
                 stride: Tuple[int, int, int] = (16, 16, 16),  # Isotropic stride (Z, Y, X)
                 subsample_frac: float = 1.0,
                 subsample_number: int = 0,
                 seed: int = 27,
                 clahe: bool = True,
                 original_resolution_nm: Tuple[float, float, float] = (8.0, 8.0, 8.0),
                 # Native resolution of raw data (Z, Y, X)
                 target_resolution_nm: Tuple[float, float, float] = (8.0, 8.0, 8.0),
                 # Resolution to process/train at (Z, Y, X)
                 balance_patches: bool = True,  # New parameter to enable balancing
                 min_positive_pixels: int = 100,
                 # Minimum number of positive pixels for a patch to be considered 'positive'
                 expect_labels: bool = True  # Set to False for inference without labels
                 ):

        self.zarr_data_dirs = zarr_data_dirs
        self.label_type = label_type
        self.patch_size = patch_size
        self.stride = stride
        self.clahe_enabled = clahe
        self.original_resolution_nm = original_resolution_nm
        self.target_resolution_nm = target_resolution_nm
        self.expect_labels = expect_labels  # Store the new parameter

        if self.clahe_enabled:
            self.clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))

        self._volume_metadata = []  # Stores info about each Zarr volume
        self._patch_global_indices = []  # (volume_idx, z_start_eff, y_start_eff, x_start_eff)

        # Discover all Zarr files and collect their metadata
        print("Discovering Zarr volumes...")
        for dir_path in self.zarr_data_dirs:
            # Assuming each subdirectory or .zarr file is a separate volume
            # glob.glob can be adjusted based on your exact directory structure
            zarr_paths = glob.glob(os.path.join(dir_path, '*.zarr'))  # Assumes .zarr extension
            zarr_paths.extend([d for d in glob.glob(os.path.join(dir_path, '*')) if
                               os.path.isdir(d) and '.zarr' not in d])  # Handles dir as zarr

            for zarr_path in zarr_paths:
                try:
                    temp_zarr_group = zarr.open(zarr_path, mode='r')

                    # Open Zarr to get shape, but don't hold the handle for multiprocessing safety
                    # Always check for 'volumes/raw'
                    if 'volumes/raw' not in temp_zarr_group:
                        raise KeyError("'volumes/raw' not found in Zarr group.")
                    raw_array_shape = temp_zarr_group['volumes/raw'].shape

                    label_key_found = None
                    if self.expect_labels:  # Only check for labels if expected
                        expected_label_key = f'volumes/labels/{self.label_type}_ids'
                        if expected_label_key not in temp_zarr_group:
                            raise KeyError(
                                f"'{expected_label_key}' not found in Zarr group, but expect_labels is True.")
                        label_key_found = expected_label_key

                    self._volume_metadata.append({
                        'zarr_path': zarr_path,
                        'raw_key': 'volumes/raw',
                        'label_key': label_key_found,  # Store None if labels not expected
                        'shape': raw_array_shape,
                        'zarr_handle': None
                    })
                    print(f"Found Zarr volume: {zarr_path} with shape {raw_array_shape}")
                except Exception as e:
                    print(f"Warning: Could not process {zarr_path}. Skipping. Error: {e}")

        if not self._volume_metadata:
            raise ValueError("No valid Zarr volumes found or accessible. Check paths and Zarr structure.")

        # Generate all possible patch indices across all volumes
        print("Calculating all possible patch indices...")
        for volume_idx, metadata in enumerate(self._volume_metadata):
            volume_patches = self._calculate_patches_for_volume(
                volume_shape=metadata['shape'],
                patch_size=self.patch_size,
                stride=self.stride,
                original_res=self.original_resolution_nm,
                target_res=self.target_resolution_nm
            )
            # Add volume_idx to each patch tuple
            self._patch_global_indices.extend([(volume_idx,) + p for p in volume_patches])

        # Apply subsampling if specified
        n_total_patches = len(self._patch_global_indices)
        if subsample_frac != 1.0 or subsample_number != 0:
            np.random.seed(seed)
            if subsample_frac != 1.0:
                num_to_select = int(np.floor(subsample_frac * n_total_patches))
                indices = np.random.choice(np.arange(n_total_patches), size=num_to_select, replace=False)
                self._patch_global_indices = [self._patch_global_indices[i] for i in indices]

            if subsample_number != 0:
                n_current_patches = len(self._patch_global_indices)
                num_to_select = min(subsample_number, n_current_patches)
                indices = np.random.choice(np.arange(n_current_patches), size=num_to_select, replace=False)
                self._patch_global_indices = [self._patch_global_indices[i] for i in indices]
            print(f"Subsampled to {len(self._patch_global_indices)} patches.")

        print(f"Dataset initialized with {len(self._patch_global_indices)} total patches.")

        # After self._patch_global_indices is populated and subsampled:
        self.balance_patches = balance_patches
        self.min_positive_pixels = min_positive_pixels
        self.patch_weights = None  # Will store weights for WeightedRandomSampler

        # Only calculate patch positivity if labels are expected AND balancing is enabled
        if self.expect_labels and self.balance_patches:
            print("Calculating patch positivity for balanced sampling...")
            self._calculate_patch_positivity()  # This method relies on labels being present
            print("Patch positivity calculation complete.")
        elif self.balance_patches:
            print("Warning: Patch balancing is enabled but expect_labels is False. Balancing will be skipped.")

    def _calculate_patch_positivity(self):
        """
        Iterates through the dataset (without loading full patches into memory)
        to determine which patches contain positive labels.
        This will be used to generate weights for WeightedRandomSampler.
        It's only called if self.expect_labels is True and self.balance_patches is True.
        """
        positive_patch_indices = []
        negative_patch_indices = []

        # Temporarily store Zarr handles to avoid re-opening constantly during this pass
        # This is safe because this method is called once in __init__ before multiprocessing starts
        zarr_handles = [zarr.open(meta['zarr_path'], mode='r') for meta in self._volume_metadata]

        for i, (volume_idx, z_start_eff, y_start_eff, x_start_eff) in tqdm(enumerate(self._patch_global_indices),
                                                                           total=len(self._patch_global_indices)):
            metadata = self._volume_metadata[volume_idx]
            label_key = metadata['label_key']
            raw_volume_shape = metadata['shape']

            label_array = zarr_handles[volume_idx][label_key]

            # Calculate raw data coordinates for slicing
            scale_z_inv = self.original_resolution_nm[0] / self.target_resolution_nm[0]
            scale_y_inv = self.original_resolution_nm[1] / self.target_resolution_nm[1]
            scale_x_inv = self.original_resolution_nm[2] / self.target_resolution_nm[2]

            z_start_raw = int(z_start_eff * scale_z_inv)
            y_start_raw = int(y_start_eff * scale_y_inv)
            x_start_raw = int(x_start_eff * scale_x_inv)

            z_end_raw = int((z_start_eff + self.patch_size[0]) * scale_z_inv)
            y_end_raw = int((y_start_eff + self.patch_size[1]) * scale_y_inv)
            x_end_raw = int((x_start_eff + self.patch_size[2]) * scale_x_inv)

            # Ensure indices are within raw volume bounds
            z_end_raw = min(z_end_raw, raw_volume_shape[0])
            y_end_raw = min(y_end_raw, raw_volume_shape[1])
            x_end_raw = min(x_end_raw, raw_volume_shape[2])

            # Load only the label patch to check for positive pixels
            # Using .sum() on the boolean array is efficient for checking counts
            # We explicitly load as numpy to avoid Dask overhead if not needed, or
            # use .compute() if `label_array` is a Dask array.
            label_patch = label_array[z_start_raw:z_end_raw, y_start_raw:y_end_raw, x_start_raw:x_end_raw]

            # --- CRITICAL CHANGE FOR uint64 LABELS ---
            # Count positive pixels: simply count all non-zero pixels.
            # No need for float conversion or 0.5 threshold here.
            num_positive_in_patch = np.sum(label_patch != 0)  # Count all non-zero pixels

            if num_positive_in_patch >= self.min_positive_pixels:
                positive_patch_indices.append(i)
            else:
                negative_patch_indices.append(i)

        if not positive_patch_indices:
            print("Warning: No positive patches found based on min_positive_pixels threshold.")
            self.patch_weights = torch.ones(len(self._patch_global_indices), dtype=torch.double)
            return

        print(
            f"Found {len(positive_patch_indices)} positive patches and {len(negative_patch_indices)} negative patches.")

        # --- CORRECTED WEIGHT ASSIGNMENT FOR PRIORITIZING LABELED PATCHES ---
        # Goal: Make sure batches contain more patches with actual labels.
        # This means patches with labels (positive_patch_indices) should have a HIGHER weight.
        # Patches with only background (negative_patch_indices) should have a LOWER weight.

        # Strategy: Set the weight of the *unlabeled* patches (minority of patches, by your count) to 1.0.
        # Then, scale the weight of the *labeled* patches (majority of patches) inversely proportional to their abundance,
        # ensuring they still get priority in sampling.

        if len(negative_patch_indices) > 0:  # If there are any negative patches (unlabeled)
            # Base weight for patches with no labels (minority of patches identified)
            weight_for_unlabeled_patch = 1.0

            # Weight for patches with labels (majority of patches identified)
            # This ratio means labeled patches will be sampled (num_unlabeled / num_labeled) times less often than unlabeled patches
            # if we base the unlabeled weight at 1.0. This is still effectively downweighting.

            # Instead, let's assign higher weight to the "good" samples (those with labels)
            # And lower weight to the "bad" samples (those without labels).
            # The standard inverse frequency weighting for the sampler:
            # weight_i = 1.0 / frequency_of_class_of_sample_i

            # Let's say:
            # Class A = patches with labels (your `positive_patch_indices`, count = 1489)
            # Class B = patches without labels (your `negative_patch_indices`, count = 226)

            # We want to sample Class A (labeled) more.
            # So, Weight(Class A) > Weight(Class B).

            # A common strategy is to make Weight(Class A) = 1.0, and Weight(Class B) = (Count_A / Count_B) if Count_A < Count_B
            # Or make Weight(Class A) = (Count_B / Count_A) if Count_A > Count_B and Weight(Class B) = 1.0

            # Given your counts (Labeled: 1489, Unlabeled: 226), Labeled patches are the majority.
            # If we simply set Weight_Labeled = 1.0, and Weight_Unlabeled = 1.0, the sampler will pick 1489 Labeled for every 226 Unlabeled. This is good!

            # The issue with your previous output `positive patch weight: 0.15, negative patch weight: 1.00` was:
            # It implies `weight_positive = (Count_Negative / Count_Positive) * weight_negative_base`.
            # If `weight_negative_base = 1.0`, then `weight_positive = (226 / 1489) = 0.15`.
            # This makes the labeled patches LESS likely to be sampled than unlabeled ones.

            # To prioritize labeled patches for segmentation:
            weight_for_labeled_patch = 1.0  # Give base weight to the patches with labels
            weight_for_unlabeled_patch = float(len(negative_patch_indices)) / len(
                positive_patch_indices)  # This ratio is < 1.0

            # This effectively makes the minority class (unlabeled patches) sampled less frequently,
            # ensuring the majority class (labeled patches) dominates the sampling.
            # E.g., for every 1.0 labeled patch effectively sampled, ~0.15 unlabeled patches are sampled.
            # This will result in your batches being *overwhelmingly full of labeled patches*.

            # If the above (weight_for_labeled_patch = 1.0, weight_for_unlabeled_patch = 0.15) is still too aggressive (too few background samples in batch)
            # You might want to assign equal weight to both, or slightly boost minority background if background context is crucial.
            # However, for learning to segment a small object, focusing on positive samples is key.

            self.patch_weights = torch.zeros(len(self._patch_global_indices), dtype=torch.double)
            for idx in positive_patch_indices:  # Patches with labels
                self.patch_weights[idx] = weight_for_labeled_patch
            for idx in negative_patch_indices:  # Patches with NO labels
                self.patch_weights[idx] = weight_for_unlabeled_patch

            print(f"Assigned weights (prioritizing labeled patches for segmentation):")
            print(f"  Labeled patch weight: {weight_for_labeled_patch:.2f}")
            print(f"  Unlabeled patch weight: {weight_for_unlabeled_patch:.2f}")

        else:  # All patches have labels (len(negative_patch_indices) == 0)
            print("All identified patches contain labels. No specific balancing needed for patch selection.")
            self.patch_weights = torch.ones(len(self._patch_global_indices), dtype=torch.double)

    def _calculate_patches_for_volume(self, volume_shape, patch_size, stride, original_res, target_res):
        """
        Calculates the starting coordinates of patches in effective (target) resolution.
        """
        patches = []

        # Calculate scaling factors
        scale_z = original_res[0] / target_res[0]
        scale_y = original_res[1] / target_res[1]
        scale_x = original_res[2] / target_res[2]

        # Calculate effective volume dimensions after potential downsampling
        effective_z = int(volume_shape[0] / scale_z)
        effective_y = int(volume_shape[1] / scale_y)
        effective_x = int(volume_shape[2] / scale_x)

        # Ensure effective dimensions are at least patch_size
        if effective_z < patch_size[0] or effective_y < patch_size[1] or effective_x < patch_size[2]:
            print(
                f"Warning: Volume {volume_shape} is too small for patch size {patch_size} after scaling to {target_res}. Skipping volume.")
            return []

        for z in range(0, effective_z - patch_size[0] + 1, stride[0]):
            for y in range(0, effective_y - patch_size[1] + 1, stride[1]):
                for x in range(0, effective_x - patch_size[2] + 1, stride[2]):
                    patches.append((z, y, x))
        return patches

    def _preprocess_image(self, image_data: np.ndarray, current_resolution_nm: Tuple[float, float, float]):
        """
        Applies CLAHE, Gaussian blur, and normalizes the image data.
        Performs downsampling if current_resolution_nm is different from target_resolution_nm.
        """
        image_data = image_data.astype(np.float32)

        # Downsample if necessary
        if current_resolution_nm != self.target_resolution_nm:
            scale_factors = (
                self.target_resolution_nm[0] / current_resolution_nm[0],  # Z
                self.target_resolution_nm[1] / current_resolution_nm[1],  # Y
                self.target_resolution_nm[2] / current_resolution_nm[2]  # X
            )
            new_shape = (
                int(image_data.shape[0] * scale_factors[0]),
                int(image_data.shape[1] * scale_factors[1]),
                int(image_data.shape[2] * scale_factors[2])
            )
            # Use scipy.ndimage.zoom for proper 3D interpolation if available,
            # otherwise, iterate slices for cv2.resize (less ideal for 3D).
            # For this example, let's use a simple approach for 3D or assume 2D slices.
            # Given the context of EM, 3D interpolation is preferred.
            # If scipy is not available, you would need to add it to the environment.
            try:
                from scipy.ndimage import zoom
                image_data = zoom(image_data, scale_factors, order=1)  # order=1 for linear interpolation
            except ImportError:
                print(
                    "Warning: scipy not found. Using slice-by-slice 2D resizing for downsampling. Install scipy for better 3D interpolation.")
                resized_image = np.zeros(new_shape, dtype=image_data.dtype)
                for z_idx in range(new_shape[0]):
                    original_z_idx = int(z_idx / scale_factors[0])
                    if original_z_idx < image_data.shape[0]:
                        resized_slice = cv2.resize(image_data[original_z_idx, :, :],
                                                   (new_shape[2], new_shape[1]),
                                                   interpolation=cv2.INTER_AREA)
                        resized_image[z_idx, :, :] = resized_slice
                image_data = resized_image

        # Apply CLAHE
        if self.clahe_enabled:
            # CLAHE needs 8-bit single channel. Apply slice by slice for 3D.
            if len(image_data.shape) == 3:
                clahe_processed_volume = np.zeros_like(image_data, dtype=np.float32)
                for z in range(image_data.shape[0]):
                    temp_slice_uint8 = cv2.normalize(image_data[z, :, :], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    clahe_slice = self.clahe.apply(temp_slice_uint8)
                    clahe_processed_volume[z, :, :] = clahe_slice.astype(np.float32)
                image_data = clahe_processed_volume
            else:  # Assume 2D
                temp_image_uint8 = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                image_data = self.clahe.apply(temp_image_uint8).astype(np.float32)

        # Apply Gaussian blur (3D or slice-by-slice 2D)
        if len(image_data.shape) == 3:
            processed_image = np.zeros_like(image_data)
            for z in range(image_data.shape[0]):
                processed_image[z, :, :] = cv2.GaussianBlur(src=image_data[z, :, :], ksize=(3, 3), sigmaX=0.5)
            image_data = processed_image
        elif len(image_data.shape) == 2:
            image_data = cv2.GaussianBlur(src=image_data, ksize=(3, 3), sigmaX=0.5)

        # Normalize data to [-1, 1]
        image_data = (image_data - 127.5) / 127.5
        return image_data

    def _preprocess_label(self, label_data: np.ndarray, current_resolution_nm: Tuple[float, float, float]):
        """
        Resizes and binarizes label data.
        Performs downsampling if current_resolution_nm is different from target_resolution_nm.
        Handles uint64 instance/semantic labels by binarizing non-zero values.
        """
        # Convert to a common float type for resizing, but handle binarization before scaling
        # if the scaling interpolation can affect binary state.
        # For zoom with order=0 (nearest-neighbor), it preserves binary state well.

        # First, ensure it's binarized (0 or 1) based on non-zero IDs
        # Convert to bool, then to int8 for 0/1, avoiding float conversion for binarization logic
        label_data_binary = (label_data != 0).astype(np.int8)  # Convert any non-zero ID to 1, zero to 0

        # Downsample if necessary (apply to the binary data directly)
        if current_resolution_nm != self.target_resolution_nm:
            scale_factors = (
                self.target_resolution_nm[0] / current_resolution_nm[0],  # Z
                self.target_resolution_nm[1] / current_resolution_nm[1],  # Y
                self.target_resolution_nm[2] / current_resolution_nm[2]  # X
            )
            new_shape = (
                int(label_data_binary.shape[0] * scale_factors[0]),
                int(label_data_binary.shape[1] * scale_factors[1]),
                int(label_data_binary.shape[2] * scale_factors[2])
            )
            try:
                from scipy.ndimage import zoom
                # Use order=0 (nearest-neighbor) for labels to preserve distinct 0/1 values
                label_data_binary = zoom(label_data_binary, scale_factors, order=0)
            except ImportError:
                print(
                    "Warning: scipy not found. Using slice-by-slice 2D resizing for downsampling. Install scipy for better 3D interpolation.")
                resized_label = np.zeros(new_shape, dtype=label_data_binary.dtype)
                for z_idx in range(new_shape[0]):
                    original_z_idx = int(z_idx / scale_factors[0])
                    if original_z_idx < label_data_binary.shape[0]:
                        resized_slice = cv2.resize(label_data_binary[original_z_idx, :, :],
                                                   (new_shape[2], new_shape[1]),
                                                   interpolation=cv2.INTER_NEAREST)  # Nearest neighbor for binary labels
                        resized_label[z_idx, :, :] = resized_slice
                label_data_binary = resized_label

        # The data is already binarized (0 or 1) by this point
        return label_data_binary.astype(np.int8)  # Ensure final output is int8 (0 or 1)

    def __len__(self):
        return len(self._patch_global_indices)

    def __getitem__(self, idx):
        # Get the global index for the patch
        volume_idx, z_start_eff, y_start_eff, x_start_eff = self._patch_global_indices[idx]

        # Get metadata for the specific volume
        metadata = self._volume_metadata[volume_idx]
        zarr_path = metadata['zarr_path']
        raw_key = metadata['raw_key']
        # label_key will be None if self.expect_labels is False
        label_key = metadata['label_key']
        raw_volume_shape = metadata['shape']

        # Open the Zarr file within __getitem__ for multiprocessing safety
        zarr_group = zarr.open(zarr_path, mode='r')
        raw_array = zarr_group[raw_key]

        # Calculate raw data coordinates for slicing
        scale_z_inv = self.original_resolution_nm[0] / self.target_resolution_nm[0]
        scale_y_inv = self.original_resolution_nm[1] / self.target_resolution_nm[1]
        scale_x_inv = self.original_resolution_nm[2] / self.target_resolution_nm[2]

        z_start_raw = int(z_start_eff * scale_z_inv)
        y_start_raw = int(y_start_eff * scale_y_inv)
        x_start_raw = int(x_start_eff * scale_x_inv)

        z_end_raw = int((z_start_eff + self.patch_size[0]) * scale_z_inv)
        y_end_raw = int((y_start_eff + self.patch_size[1]) * scale_y_inv)
        x_end_raw = int((x_start_eff + self.patch_size[2]) * scale_x_inv)

        # Ensure indices are within raw volume bounds
        z_end_raw = min(z_end_raw, raw_volume_shape[0])
        y_end_raw = min(y_end_raw, raw_volume_shape[1])
        x_end_raw = min(x_end_raw, raw_volume_shape[2])

        # Extract raw image patch
        # .astype(np.uint8) ensures consistent type for preprocessing
        raw_image_patch = raw_array[z_start_raw:z_end_raw, y_start_raw:y_end_raw, x_start_raw:x_end_raw].astype(
            np.uint8)

        # Preprocess the extracted image patch
        processed_image_patch = self._preprocess_image(raw_image_patch, self.original_resolution_nm)

        # Crop to exact target patch size in case of rounding errors from scaling
        processed_image_patch = processed_image_patch[:self.patch_size[0], :self.patch_size[1], :self.patch_size[2]]

        # Convert to torch tensor and add channel dimension (C, Z, Y, X)
        re_X = torch.from_numpy(processed_image_patch).unsqueeze(0)

        # --- CRITICAL CHANGE: Conditionally load and process labels ---
        re_Y = None  # Default to None
        if self.expect_labels:
            if label_key is None:  # This should ideally not happen if expect_labels is True, based on __init__ logic
                raise RuntimeError("Label key is None while expect_labels is True. Dataset initialization issue.")

            raw_label_patch = zarr_group[label_key][z_start_raw:z_end_raw, y_start_raw:y_end_raw,
                              x_start_raw:x_end_raw].astype(np.uint64)
            processed_label_patch = self._preprocess_label(raw_label_patch, self.original_resolution_nm)

            processed_label_patch = processed_label_patch[:self.patch_size[0], :self.patch_size[1], :self.patch_size[2]]
            re_Y = torch.from_numpy(processed_label_patch).unsqueeze(0)

        # Return (image, label) if labels expected, else (image,)
        if re_Y is not None:
            return re_X, re_Y
        else:
            return re_X
