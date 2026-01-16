import zarr
import numpy as np
from scipy.ndimage import center_of_mass
import glob
import os
from tqdm import tqdm
import argparse


def create_mito_locations(data_dir, resolution=None):
    """
    Scans for Zarr containers in the data directory, reads the neuron_ids
    segmentation, calculates the center of mass for each unique ID (mitochondrion),
    and saves these locations to a new dataset 'volumes/labels/mito_locations'.

    Args:
        data_dir (str): The path to the directory containing the Zarr files.
    """
    samples = glob.glob(os.path.join(data_dir, "*.zarr"))
    print(f"Found {len(samples)} samples to process in '{data_dir}'.")

    if not samples:
        print("No Zarr files found. Please check the data_dir path.")
        return

    for sample_path in tqdm(samples, desc="Processing samples"):
        try:
            # Open Zarr file in append mode to allow writing
            zarr_file = zarr.open(sample_path, mode='a')

            if resolution is None:
                try:
                    resolution = zarr_file['volumes/raw'].attrs["resolution"]
                except Exception as e:
                    print("Resolution must be input or be set in volumes/raw in the zarr file")

            # 1. Check for and read the segmentation data
            if 'volumes/labels/neuron_ids' not in zarr_file:
                tqdm.write(f"Skipping {os.path.basename(sample_path)}: 'volumes/labels/neuron_ids' not found.")
                continue

            labels = zarr_file['volumes/labels/neuron_ids'][:]
            tqdm.write(f"Processing {os.path.basename(sample_path)} with shape {labels.shape}")

            # 2. Find unique labels (mitochondria IDs), excluding background (label 0)
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != 0]

            if len(unique_labels) == 0:
                tqdm.write(f"No mitochondria found in {os.path.basename(sample_path)}, skipping.")
                continue

            # 3. Calculate center of mass for each mitochondrion label
            # The output of center_of_mass is a list of tuples, one for each label
            # We convert it to an Nx3 numpy array of (z, y, x) voxel coordinates
            centers = center_of_mass(labels, labels, unique_labels)
            locations_in_voxels = np.array(centers)
            locations_in_nm = locations_in_voxels * resolution

            if locations_in_voxels.size == 0:
                tqdm.write(f"No locations computed for {os.path.basename(sample_path)}, skipping write.")
                continue

            # 4. Store the locations in a new dataset
            dataset_path = 'volumes/labels/mito_locations'
            if dataset_path in zarr_file:
                tqdm.write(f"'{dataset_path}' already exists in {os.path.basename(sample_path)}. Overwriting.")
                del zarr_file[dataset_path]

            # Store as int to be used as index
            zarr_file.create_dataset(
                dataset_path,
                data=locations_in_nm.astype(int), # save the nm locations for gunpowder
                chunks=(100, 3),  # Chunking for potentially faster read access
                compressor=zarr.codecs.GZip(level=5) # this has changed in newer versions of zarr to GzipCodec
            )

            tqdm.write(
                f"Saved {len(locations_in_nm)} mitochondria locations to {os.path.basename(sample_path)}/{dataset_path}")

        except Exception as e:
            tqdm.write(f"An error occurred while processing {sample_path}: {e}")


if __name__ == '__main__':
    """
    How to run:
    
    python create_mito_locations.py /path/to/your/zarr/train_data
    
    The path should be the one containing your training .zarr files, e.g.,
    `.../data_3d/train` from your config.
    """
    parser = argparse.ArgumentParser(description="Create mitochondria location datasets in Zarr files.")
    parser.add_argument('data_dir', type=str, help='Directory containing Zarr training samples.')
    args = parser.parse_args()

    create_mito_locations(args.data_dir)
