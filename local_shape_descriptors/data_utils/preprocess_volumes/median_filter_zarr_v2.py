import argparse
import zarr
import dask.array as da
import numpy as np
from scipy.ndimage import median_filter
from tqdm import tqdm

def rechunk_to_xy_slices(input_path, intermediate_path):
    """
    Memory-efficient rechunking using Dask
    """
    # Open input volume with Dask
    input_volume = da.from_zarr(input_path, component="volumes/raw")

    # Rechunk to 1,x,y slices using Dask
    rechunked_volume = input_volume.rechunk(
        chunks=(1, input_volume.shape[1], input_volume.shape[2])
    )

    # Compute and save to intermediate Zarr
    intermediate_group = zarr.open(intermediate_path, mode='w')
    intermediate_volume = intermediate_group.create_dataset(
        'rechunked_volume',
        shape=input_volume.shape,
        dtype=input_volume.dtype,
        chunks=(1, input_volume.shape[1], input_volume.shape[2])
    )

    # Incremental writing to avoid full memory load
    for i in tqdm(range(0, input_volume.shape[0], 64)):
        chunk = rechunked_volume[i:i + 64].compute()
        intermediate_volume[i:i + 64] = chunk

    return intermediate_volume


def apply_z_median_filter(intermediate_path, output_path, kernel=3):
    """
    Apply median filter along z-slices with Dask
    """
    # Open intermediate volume
    intermediate_volume = da.from_zarr(intermediate_path, component='rechunked_volume')

    # Create output Zarr group
    output_group = zarr.open(output_path, mode='w')
    filtered_volume = output_group.create_dataset(
        'filtered_volume',
        shape=intermediate_volume.shape,
        dtype=intermediate_volume.dtype,
        chunks=(1, intermediate_volume.shape[1], intermediate_volume.shape[2])
    )

    # Dask-based z-slice median filtering
    def process_z_slice(z, kernel=3):
        """
        Process single z-slice with a customizable kernel size neighborhood

        Parameters:
        - z: index of the slice to process
        - kernel: size of the neighborhood (must be odd integer)

        Returns:
        - Processed slice after median filtering and averaging
        """
        # Ensure kernel is odd
        if kernel % 2 == 0:
            raise ValueError("Kernel size must be odd")

        # Calculate half kernel size (for window calculations)
        half_k = kernel // 2

        # Calculate slice range based on kernel size
        z_start = max(0, z - half_k)
        z_end = min(intermediate_volume.shape[0], z + half_k + 1)

        # Get neighborhood slices
        z_slice_neighborhood = intermediate_volume[z_start:z_end]

        # Handle edge cases where we couldn't get the full kernel size
        if z_slice_neighborhood.shape[0] < kernel:
            # We're at an edge and couldn't get enough slices
            # Just use what we have - the algorithm will adapt
            pass

        # Apply median filter and average
        return median_filter(z_slice_neighborhood, size=(z_slice_neighborhood.shape[0], 1, 1)).mean(axis=0)

    # def process_z_slice(z):
    #     """
    #     Process single z-slice with 3-slice neighborhood
    #     """
    #     if z == 0:
    #         # First slice - use first three slices
    #         z_slice_neighborhood = intermediate_volume[0:3]
    #     elif z == intermediate_volume.shape[0] - 1:
    #         # Last slice - use last three slices
    #         z_slice_neighborhood = intermediate_volume[z - 2:z + 1]
    #     else:
    #         # Middle slices - use surrounding 3 slices
    #         z_slice_neighborhood = intermediate_volume[z - 1:z + 2]
    #
    #     # Apply median filter and average
    #     return median_filter(z_slice_neighborhood, size=(3, 1, 1)).mean(axis=0)

    # Incremental processing and writing
    for i in tqdm(range(0, intermediate_volume.shape[0], 64)):
        end = min(i + 64, intermediate_volume.shape[0])

        # Compute filtered slices
        filtered_chunk = da.stack([
            process_z_slice(z, kernel=kernel) for z in range(i, end)
        ]).compute()

        # Write to output volume
        filtered_volume[i:end] = filtered_chunk

    return filtered_volume


def rechunk_to_original(filtered_path, final_output_path, kernel=3):
    """
    Memory-efficient rechunking back to original chunk size
    """
    # Open filtered volume with Dask
    filtered_volume = da.from_zarr(filtered_path, component="filtered_volume")

    # Rechunk to 64,64,64
    rechunked_volume = filtered_volume.rechunk(
        chunks=(64, 64, 64)
    )

    # Create final output Zarr group
    final_output_group = zarr.open(final_output_path, mode='w')
    final_volume = final_output_group.create_dataset(
        'final_volume',
        shape=filtered_volume.shape,
        dtype=filtered_volume.dtype,
        chunks=(64, 64, 64)
    )

    # Incremental writing
    for i in tqdm(range(0, filtered_volume.shape[0], 64)):
        chunk = rechunked_volume[i:i + 64].compute()
        final_volume[i:i + 64] = chunk

    return final_volume


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Memory-Efficient Z-Slice Median Filtering')
    parser.add_argument('input_path', type=str, help='Input Zarr volume path')
    parser.add_argument('output_path', type=str, help='Output Zarr volume path')
    parser.add_argument('--kernel', type=int, default=3, help='Output Zarr volume path')
    args = parser.parse_args()

    # Intermediate paths
    intermediate_path = f"{args.input_path}_intermediate.zarr"
    filtered_path = f"{args.input_path}_filtered.zarr"

    # Workflow steps
    print("Step 1: Rechunking to XY slices")
    rechunked_volume = rechunk_to_xy_slices(args.input_path, intermediate_path)

    print("Step 2: Applying Z-slice Median Filter")
    filtered_volume = apply_z_median_filter(intermediate_path, filtered_path, kernel=args.kernel)

    print("Step 3: Rechunking to Final Volume")
    final_volume = rechunk_to_original(filtered_path, args.output_path)

    print(f"Filtered volume saved to: {args.output_path}")
    print(f"Final volume shape: {final_volume.shape}")
    print(f"Final volume chunks: {final_volume.chunks}")


if __name__ == '__main__':
    main()
