import numpy as np
from scipy.stats import entropy
from skimage.util import view_as_windows
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tifffile import imread, imwrite
from tqdm import tqdm


# Define this function at the top level of your module
def calculate_entropy_for_slice(slice_data):
    """Calculate entropy for a slice of windows"""
    i, windows_slice = slice_data
    slice_shape = windows_slice.shape[:2]
    result = np.zeros(slice_shape)

    for j in range(slice_shape[0]):
        for k in range(slice_shape[1]):
            window = windows_slice[j, k]
            hist, _ = np.histogram(window, bins=256, range=(0, 256), density=True)
            # Filter out zeros to avoid warnings
            hist = hist[hist > 0]
            if len(hist) > 0:
                result[j, k] = entropy(hist)

    return i, result


def calculate_entropy_3d(volume, window_size, n_jobs=None):
    """Calculate Shannon entropy for a 3D volume using parallel processing"""
    # Ensure window size is a tuple of three odd numbers
    if not (isinstance(window_size, tuple) and len(window_size) == 3 and all(w % 2 == 1 for w in window_size)):
        raise ValueError("Window size must be a tuple of three odd numbers.")

    # Pad the volume to handle borders
    pad_width = tuple((w // 2, w // 2) for w in window_size)
    padded_volume = np.pad(volume, pad_width, mode='reflect')

    # Create sliding windows
    windows = view_as_windows(padded_volume, window_size)

    # Determine number of processes to use
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)

    # Prepare data for parallel processing
    slice_data = [(i, windows[i]) for i in range(windows.shape[0])]

    # Calculate entropy in parallel
    entropy_volume = np.zeros(volume.shape)

    # Use a different approach that works better with multiprocessing
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for i, result in executor.map(calculate_entropy_for_slice, slice_data):
            entropy_volume[i] = result

    return entropy_volume


def create_entropy_mask(volume, window_size, threshold, n_jobs=None):
    """Create a mask based on entropy threshold"""
    volume = np.squeeze(volume)  # squeeze any singleton channel dimension
    entropy_volume = calculate_entropy_3d(volume, window_size, n_jobs)
    mask = np.where(entropy_volume > threshold, 1, 0)
    return mask, entropy_volume  # Return both for visualization


# Alternative approach without multiprocessing (slower but more reliable)
def calculate_entropy_3d_sequential(volume, window_size):
    """Calculate Shannon entropy for a 3D volume without parallel processing"""
    # Ensure window size is a tuple of three odd numbers
    if not (isinstance(window_size, tuple) and len(window_size) == 3 and all(w % 2 == 1 for w in window_size)):
        raise ValueError("Window size must be a tuple of three odd numbers.")

    # Pad the volume to handle borders
    pad_width = tuple((w // 2, w // 2) for w in window_size)
    padded_volume = np.pad(volume, pad_width, mode='reflect')

    # Create sliding windows
    windows = view_as_windows(padded_volume, window_size)

    # Calculate entropy for each window
    entropy_volume = np.zeros(volume.shape)

    # Process in batches to show progress
    total_slices = windows.shape[0]
    for i in tqdm(range(total_slices)):
        if i % 10 == 0:  # Print progress every 10 slices
            print(f"Processing slice {i}/{total_slices}")

        for j in range(windows.shape[1]):
            for k in range(windows.shape[2]):
                window = windows[i, j, k]
                hist, _ = np.histogram(window, bins=256, range=(0, 256), density=True)
                # Filter out zeros
                hist = hist[hist > 0]
                if len(hist) > 0:
                    entropy_volume[i, j, k] = entropy(hist)

    return entropy_volume


def create_entropy_mask_sequential(volume, window_size, threshold):
    """Create a mask based on entropy threshold using sequential processing"""
    volume = np.squeeze(volume)  # squeeze any singleton channel dimension
    entropy_volume = calculate_entropy_3d_sequential(volume, window_size)
    mask = np.where(entropy_volume > threshold, 1, 0)
    return mask, entropy_volume  # Return both for visualization


def calculate_entropy_2d(slice_data):
    """Calculate entropy for a 2D slice using a sliding window approach"""
    slice_idx, slice_img, window_size_2d = slice_data

    # Ensure window size is a tuple of two odd numbers
    if not (isinstance(window_size_2d, tuple) and len(window_size_2d) == 2 and all(w % 2 == 1 for w in window_size_2d)):
        raise ValueError("Window size must be a tuple of two odd numbers.")

    # Pad the slice to handle borders
    pad_width = tuple((w // 2, w // 2) for w in window_size_2d)
    padded_slice = np.pad(slice_img, pad_width, mode='reflect')

    # Create sliding windows
    windows = view_as_windows(padded_slice, window_size_2d)

    # Calculate entropy for each window
    entropy_slice = np.zeros(slice_img.shape)

    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            window = windows[i, j]
            hist, _ = np.histogram(window, bins=256, range=(0, 256), density=True)
            hist = hist[hist > 0]
            if len(hist) > 0:
                entropy_slice[i, j] = entropy(hist)

    return slice_idx, entropy_slice


def calculate_entropy_3d_slice_parallel(volume, window_size, n_jobs=None):
    """Calculate Shannon entropy for a 3D volume by processing 2D slices in parallel"""
    # Ensure window size is a tuple of three odd numbers
    if not (isinstance(window_size, tuple) and len(window_size) == 3 and all(w % 2 == 1 for w in window_size)):
        raise ValueError("Window size must be a tuple of three odd numbers.")

    # Extract 2D window size (height, width)
    window_size_2d = window_size[1:]

    # Determine number of processes to use
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)

    # Prepare data for parallel processing - each slice is processed independently
    slice_data = [(i, volume[i], window_size_2d) for i in range(volume.shape[0])]

    # Calculate entropy in parallel
    entropy_volume = np.zeros(volume.shape)

    print(f"Processing {len(slice_data)} slices in parallel with {n_jobs} workers...")
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(tqdm(executor.map(calculate_entropy_2d, slice_data), total=len(slice_data)))

    # Collect results
    for slice_idx, entropy_slice in results:
        entropy_volume[slice_idx] = entropy_slice

    return entropy_volume


def create_entropy_mask_slice_parallel(volume, window_size, threshold, n_jobs=None):
    """Create a mask based on entropy threshold using slice-parallel processing"""
    volume = np.squeeze(volume)  # squeeze any singleton channel dimension
    entropy_volume = calculate_entropy_3d_slice_parallel(volume, window_size, n_jobs)
    mask = np.where(entropy_volume > threshold, 1, 0)
    return mask, entropy_volume  # Return both for visualization


if __name__ == "__main__":
    # Define parameters
    window_size = (5, 5, 5)  # Ensure this is a tuple of three odd numbers
    threshold = 0.5  # Adjust this threshold based on your needs

    # Load your volume data here
    data_path = "/Volumes/SamiaSan/Camb/EM_masking/Mask_data/sam_vol_for_mask.tiff"
    volume = imread(data_path)

    if len(volume.shape) > 3:
        # Squeeze any singleton channel dimension
        volume = np.squeeze(volume)

    # Clipping the intensities
    # Calculate and print the intensity range before clipping
    min_intensity_before = np.min(volume)
    max_intensity_before = np.max(volume)
    print(f"Intensity range before clipping: {min_intensity_before} to {max_intensity_before}")

    # Define the clipping range
    clip_min = 100
    clip_max = max_intensity_before

    # Clip the intensities
    clipped_volume = np.clip(volume, clip_min, clip_max)

    # Calculate and print the intensity range after clipping
    min_intensity_after = np.min(clipped_volume)
    max_intensity_after = np.max(clipped_volume)
    print(f"Intensity range after clipping: {min_intensity_after} to {max_intensity_after}")

    # Visualize a few slices of the volume
    slices = [70, 100, 200]  # Example slice indices

    # Create a figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, len(slices), figsize=(15, 10))

    # Plot the original volume slices
    for i, slice_idx in enumerate(slices):
        axes[0, i].imshow(volume[:, :, slice_idx], cmap='gray')
        axes[0, i].set_title(f'Original Slice {slice_idx}')
        axes[0, i].axis('off')

    # Plot the clipped volume slices
    for i, slice_idx in enumerate(slices):
        axes[1, i].imshow(clipped_volume[:, :, slice_idx], cmap='gray')
        axes[1, i].set_title(f'Clipped Slice {slice_idx}')
        axes[1, i].axis('off')

    plt.tight_layout()
    # plt.show()
    plt.savefig("./tmp_clipped.png", dpi=300)

    # # Choose whether to use parallel or sequential processing
    # try:
    #     print("Trying parallel processing...")
    #     entropy_mask, entropy_volume = create_entropy_mask(volume, window_size, threshold, n_jobs=5)
    # except Exception as e:
    #     print(f"Parallel processing failed with error: {e}")
    #     print("Falling back to sequential processing...")
    #     entropy_mask, entropy_volume = create_entropy_mask_sequential(volume, window_size, threshold)

    # Choose whether to use parallel or sequential processing
    try:
        print("Trying slice-parallel processing (faster)...")
        entropy_mask, entropy_volume = create_entropy_mask_slice_parallel(clipped_volume, window_size, threshold,
                                                                          n_jobs=8)
    except Exception as e:
        print(f"Slice-parallel processing failed with error: {e}")
        try:
            print("Trying standard parallel processing...")
            entropy_mask, entropy_volume = create_entropy_mask(clipped_volume, window_size, threshold, n_jobs=5)
        except Exception as e:
            print(f"Parallel processing failed with error: {e}")
            print("Falling back to sequential processing...")
            entropy_mask, entropy_volume = create_entropy_mask_sequential(clipped_volume, window_size, threshold)

    # Visualization code
    slices = [70, 100, 200]  # Example slice indices

    fig, axes = plt.subplots(2, len(slices), figsize=(15, 10))

    # Plot the original volume slices
    for i, slice_idx in enumerate(slices):
        axes[0, i].imshow(volume[:, :, slice_idx], cmap='gray')
        axes[0, i].set_title(f'Original Slice {slice_idx}')
        axes[0, i].axis('off')

    # Plot the entropy mask slices
    for i, slice_idx in enumerate(slices):
        axes[1, i].imshow(entropy_mask[:, :, slice_idx], cmap='gray')
        axes[1, i].set_title(f'Mask Slice {slice_idx}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()
