""""
use conda env funkelsd

Run with:
# Default threadpool processing
python em_volume_filter.py input.zarr output.zarr

# Specify Dask parallelism
python em_volume_filter.py input.zarr output.zarr --parallel-method dask

# Custom filter size and patch shape
python em_volume_filter.py input.zarr output.zarr \
    --filter-size 5 5 5 \
    --patch-shape 512 512 64 \
    --max-workers 16
"""

import argparse
import zarr
import numpy as np
import dask.array as da
from concurrent.futures import ThreadPoolExecutor
import os
from scipy.ndimage import median_filter
import itertools


class ZarrMedianFilterProcessor:
    def __init__(self,
                 input_zarr_path,
                 output_zarr_path,
                 filter_size=(3, 3, 3),
                 patch_shape=None,
                 overlap=(1, 1, 1),
                 max_workers=None,
                 parallel_method='threadpool'):
        """
        Process large Zarr volumes with median filtering using configurable parallelism.

        Parameters:
        -----------
        input_zarr_path : str
            Path to input Zarr volume
        output_zarr_path : str
            Path to save filtered Zarr volume
        filter_size : tuple, optional
            Size of median filter kernel. Default (3,3,3)
        patch_shape : tuple, optional
            Shape of patches to process. If None, auto-determined
        overlap : tuple, optional
            Overlap between patches to prevent edge artifacts
        max_workers : int, optional
            Number of threads/workers to use
        parallel_method : str, optional
            Parallelization method: 'threadpool' or 'dask'
        """
        # Open input volume using Dask for efficient I/O
        self.input_volume = da.from_zarr(input_zarr_path, component="volumes/raw")
        self.volume_shape = self.input_volume.shape

        # Prepare output volume
        self.output_group = zarr.open(output_zarr_path, mode='a')
        self.output_volume = self.output_group.create_dataset(
            'volumes/raw_filtered',
            shape=self.volume_shape,
            dtype=self.input_volume.dtype,
            chunks=self.input_volume.chunks
        )

        # Configure processing parameters
        self.filter_size = filter_size
        self.overlap = overlap
        self.parallel_method = parallel_method

        # Determine patch shape
        if patch_shape is None:
            # Automatically determine reasonable patch size
            self.patch_shape = tuple(
                min(dim, max(filter_size[i] * 10, 256))
                for i, dim in enumerate(self.volume_shape)
            )
        else:
            self.patch_shape = patch_shape

        # Configure workers
        self.max_workers = max_workers or min(8, (os.cpu_count() or 1) * 2)

    def _get_patch_coordinates(self):
        """
        Generate patch coordinates with overlap handling
        """
        coords = []
        for dim in range(3):
            dim_coords = []
            start = 0
            while start < self.volume_shape[dim]:
                end = min(start + self.patch_shape[dim], self.volume_shape[dim])
                dim_coords.append((start, end))
                # Move start with overlap consideration
                start = end - self.overlap[dim]
            coords.append(dim_coords)

        return list(itertools.product(*coords))

    def _process_patch_threadpool(self, patch_start, patch_end):
        """
        Process a single patch with median filtering using ThreadPool
        """
        # Extract patch from input volume
        patch_slice = tuple(
            slice(start, end)
            for start, end in zip(patch_start, patch_end)
        )

        # Materialize the patch data (load into memory)
        patch_data = self.input_volume[patch_slice].compute()

        # Apply median filter
        filtered_patch = median_filter(patch_data, size=self.filter_size)

        # Write to output volume
        self.output_volume[patch_slice] = filtered_patch

    def _process_patch_dask(self, patch_start, patch_end):
        """
        Process a single patch with median filtering using Dask
        """
        # Extract patch from input volume
        patch_slice = tuple(
            slice(start, end)
            for start, end in zip(patch_start, patch_end)
        )

        # Apply median filter using Dask's map_overlap
        filtered_patch = da.map_overlap(
            lambda x: median_filter(x, size=self.filter_size),
            self.input_volume[patch_slice],
            depth=tuple(f // 2 for f in self.filter_size),
            boundary='reflect'
        )

        # Compute and write to output volume
        filtered_patch_data = filtered_patch.compute()
        self.output_volume[patch_slice] = filtered_patch_data

    def process(self):
        """
        Process entire volume in patches using selected parallelism method
        """
        # Get patch coordinates
        patch_coords = self._get_patch_coordinates()

        # Choose processing method
        if self.parallel_method == 'threadpool':
            # ThreadPool processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                list(executor.map(
                    lambda coords: self._process_patch_threadpool(
                        tuple(start for start, _ in coords),
                        tuple(end for _, end in coords)
                    ),
                    patch_coords
                ))
        elif self.parallel_method == 'dask':
            # Dask processing
            for coords in patch_coords:
                start = tuple(start for start, _ in coords)
                end = tuple(end for _, end in coords)
                self._process_patch_dask(start, end)
        else:
            raise ValueError(f"Unsupported parallel method: {self.parallel_method}")

        return self.output_volume


def parse_arguments():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description='Median Filter EM Volume')
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to input Zarr volume'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='Path to output filtered Zarr volume'
    )
    parser.add_argument(
        '--filter-size',
        type=int,
        nargs=3,
        default=[3, 3, 3],
        help='Size of median filter kernel (default: 3 3 3)'
    )
    parser.add_argument(
        '--patch-shape',
        type=int,
        nargs=3,
        default=None,
        help='Patch shape for processing (default: auto-determined)'
    )
    parser.add_argument(
        '--parallel-method',
        type=str,
        choices=['threadpool', 'dask'],
        default='threadpool',
        help='Parallelization method (default: threadpool)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Maximum number of workers/threads (default: auto)'
    )

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Prepare patch shape (convert to None if not specified)
    patch_shape = tuple(args.patch_shape) if args.patch_shape else None

    # Initialize and process
    processor = ZarrMedianFilterProcessor(
        args.input_path,
        args.output_path,
        filter_size=tuple(args.filter_size),
        patch_shape=patch_shape,
        parallel_method=args.parallel_method,
        max_workers=args.max_workers
    )

    filtered_volume = processor.process()
    print(f"Filtered volume shape: {filtered_volume.shape}")
    print(f"Filtered volume saved to: {args.output_path}")
    print(f"Parallelization method: {args.parallel_method}")


if __name__ == '__main__':
    main()
