# Reset background to zero

# File: engine/utils/utils.py (or a new file like engine/utils/zarr_cleaner.py)

# File: engine/utils/utils.py

import zarr
import numpy as np
import os
import glob
from tqdm import tqdm  # For progress bar
from typing import List, Tuple

# Import GZip compressor
import zarr.storage
import zarr.codecs  # Needed for zarr.GZip()


def clean_zarr_labels(
        input_zarr_dirs: List[str],
        output_base_dir: str,
        label_key: str,  # e.g., 'volumes/labels/mito_ids' or 'volumes/labels/neuron_ids'
        sentinel_value: int = 18446744073709551613,  # The large value to replace
        output_chunk_size: Tuple[int, int, int] = (128, 128, 128),  # Recommended: match your patch_size
        overwrite_existing: bool = False
) -> List[str]:
    """
    Loads Zarr label data, replaces a specific large sentinel value with 0,
    and saves the cleaned data to new Zarr files using GZip compression.

    Args:
        input_zarr_dirs (list[str]): List of paths to directories containing input Zarr files.
        output_base_dir (str): Base directory where cleaned Zarr files will be saved.
        label_key (str): The internal key within the Zarr group pointing to the label array
                         (e.g., 'volumes/labels/mito_ids').
        sentinel_value (int): The specific large integer value to replace with 0.
        output_chunk_size (tuple): Chunking strategy for the output Zarr arrays.
                                   Recommend using your training patch size for efficiency.
        overwrite_existing (bool): If True, overwrite existing output Zarr files.

    Returns:
        list[str]: A list of paths to the newly created cleaned Zarr files.
    """
    os.makedirs(output_base_dir, exist_ok=True)
    cleaned_zarr_paths = []

    all_input_zarr_paths = []
    for dir_path in input_zarr_dirs:
        all_input_zarr_paths.extend(glob.glob(os.path.join(dir_path, '*.zarr')))
        all_input_zarr_paths.extend(
            [d for d in glob.glob(os.path.join(dir_path, '*')) if os.path.isdir(d) and '.zarr' not in d])

    if not all_input_zarr_paths:
        print(f"No Zarr files found in input directories: {input_zarr_dirs}")
        return []

    print(f"Starting Zarr cleaning for label key: '{label_key}'")
    print(f"Replacing sentinel value: {sentinel_value}")

    # Define GZip compressor
    gzip_compressor = zarr.GZip()

    for input_zarr_path in tqdm(all_input_zarr_paths, desc="Cleaning Zarr Volumes"):
        volume_name = os.path.basename(input_zarr_path)
        output_zarr_path = os.path.join(output_base_dir, volume_name)

        if os.path.exists(output_zarr_path) and not overwrite_existing:
            print(f"  Skipping {volume_name}: Output already exists and overwrite_existing is False.")
            cleaned_zarr_paths.append(output_zarr_path)
            continue

        try:
            input_zarr_group = zarr.open(input_zarr_path, mode='r')

            if label_key not in input_zarr_group:
                print(f"  Warning: Label key '{label_key}' not found in {volume_name}. Skipping this volume.")
                continue

            original_label_array = input_zarr_group[label_key]

            raw_key = 'volumes/raw'  # Standard raw data key

            output_zarr_group = zarr.open_group(output_zarr_path, mode='w')
            print(f"  Processing {volume_name}...")

            # Copy raw data (if it exists) to the new Zarr file
            if raw_key in input_zarr_group:
                print(f"    Copying raw data '{raw_key}'...")
                original_raw_array = input_zarr_group[raw_key]

                output_raw_array = output_zarr_group.create_dataset(
                    raw_key,
                    shape=original_raw_array.shape,
                    dtype=original_raw_array.dtype,
                    chunks=original_raw_array.chunks,  # Keep original chunking for raw data
                    compressor=gzip_compressor  # Use GZip compressor
                )

                # Iterate over chunks and copy.
                if original_raw_array.nbytes < (4 * 1024 ** 3):  # E.g., < 4GB, load whole for faster copy
                    output_raw_array[:] = original_raw_array[:]  # Load all into memory and copy
                else:  # Copy chunk by chunk using manual slicing for very large arrays
                    raw_chunk_shape = original_raw_array.chunks
                    for z_start in range(0, original_raw_array.shape[0], raw_chunk_shape[0]):
                        z_end = min(z_start + raw_chunk_shape[0], original_raw_array.shape[0])
                        for y_start in range(0, original_raw_array.shape[1], raw_chunk_shape[1]):
                            y_end = min(y_start + raw_chunk_shape[1], original_raw_array.shape[1])
                            for x_start in range(0, original_raw_array.shape[2], raw_chunk_shape[2]):
                                x_end = min(x_start + raw_chunk_shape[2], original_raw_array.shape[2])

                                chunk_slice = (slice(z_start, z_end), slice(y_start, y_end), slice(x_start, x_end))
                                output_raw_array[chunk_slice] = original_raw_array[chunk_slice]
                print(f"    Raw data copied.")

            # Process and save cleaned label data
            print(f"    Cleaning and saving label data '{label_key}'...")
            output_label_array = output_zarr_group.create_dataset(
                label_key,
                shape=original_label_array.shape,
                dtype=original_label_array.dtype,  # Keep original dtype (uint64) for now
                chunks=output_chunk_size,
                compressor=gzip_compressor  # Use GZip compressor
            )

            # Iterate over chunks by manually computing slices
            label_chunk_shape = original_label_array.chunks  # Use original label chunk shape for reading

            for z_start in range(0, original_label_array.shape[0], label_chunk_shape[0]):
                z_end = min(z_start + label_chunk_shape[0], original_label_array.shape[0])
                for y_start in range(0, original_label_array.shape[1], label_chunk_shape[1]):
                    y_end = min(y_start + label_chunk_shape[1], original_label_array.shape[1])
                    for x_start in range(0, original_label_array.shape[2], label_chunk_shape[2]):
                        x_end = min(x_start + label_chunk_shape[2], original_label_array.shape[2])

                        chunk_slice = (slice(z_start, z_end), slice(y_start, y_end), slice(x_start, x_end))
                        chunk_data = original_label_array[chunk_slice]

                        # Perform replacement on the chunk
                        cleaned_chunk = np.where(chunk_data == sentinel_value, 0, chunk_data)

                        # Write the cleaned chunk to the output label array
                        output_label_array[chunk_slice] = cleaned_chunk

            print(f"    Label data cleaned and saved for {volume_name}.")
            cleaned_zarr_paths.append(output_zarr_path)

        except Exception as e:
            print(f"  Error processing {volume_name}: {e}")

    print("\nZarr cleaning process finished.")
    return cleaned_zarr_paths


# Define your input and output directories
input_train_zarr_dirs = [
    "/media/samia/DATA/mounts/fibserver1/smohinta_data/catena_data/HEMI_NEURONS_FOR_MITO/data_3d/train"]
output_cleaned_train_dir = "/media/samia/DATA/mounts/fibserver1/smohinta_data/catena_data/HEMI_NEURONS_FOR_MITO/data_3d/train_cleaned"
label_key_to_clean = 'volumes/labels/neuron_ids'  # Or 'volumes/labels/mito_ids'
sentinel_val = 18446744073709551613  # The specific value you want to replace

# Recommended: Match your training patch_size
output_chunk_size_for_cleaned_zarr = (128, 128, 128)

# Call the cleaning function
cleaned_train_zarr_paths = clean_zarr_labels(
    input_zarr_dirs=input_train_zarr_dirs,
    output_base_dir=output_cleaned_train_dir,
    label_key=label_key_to_clean,
    sentinel_value=sentinel_val,
    output_chunk_size=output_chunk_size_for_cleaned_zarr,
    overwrite_existing=True  # Set to True to re-run cleaning
)

print(f"\nCleaned Zarr files created at: {cleaned_train_zarr_paths}")

# Now, when you set up your EMDataset for training, point it to the cleaned data:
# train_zarr_dirs = [output_cleaned_train_dir] # Or use cleaned_train_zarr_paths directly if it's a list of files
# test_zarr_dirs (if applicable) = ["/path/to/your/test_data_cleaned"] # Clean test data too if needed
