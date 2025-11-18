"""
Run:
python read_sharded_data_parallel_chunkwise.py \
  -f  /ceph.groups/mzlatic.grp/code/connectomic-tools/data/SAM_3G/processed_volumes/brain_crop_1 -of /net/fibserver1/ \
  -of /net/fibserver1/data/raw/SAM_3G/samia/original_samia.n5 \
  -format n5 \
  -chunk_size 256,256,64 \
    -bbox_start 4011,4190,2374 \
     -bbox_end 5000,5000,2380 \
  -max_chunk_size 256,256,64 \
  -max_workers 4


python read_sharded_data_parallel_chunkwise.py \
  -f  /ceph.groups/mzlatic.grp/code/connectomic-tools/data/SAM_3G/processed_volumes/brain_crop_1 -of /net/fibserver1/ \
  -of /net/fibserver1/data/raw/SAM_3G/samia/original_samia.n5 \
  -format n5 \
  -chunk_size 64,256,256 \
    -bbox_start 0,0,0 \
     -bbox_end 14238,15000,8000 \
  -max_chunk_size 256,256,64 \
  -max_workers 32
"""
import cloudvolume
from cloudvolume import CloudVolume
import numpy as np
import argparse
import zarr
from concurrent.futures import ProcessPoolExecutor
import os
from functools import partial
from tqdm import tqdm
import time
import warnings
import traceback
import math

warnings.filterwarnings("ignore")


def resolution_tuple(resolution_str):
    try:
        if isinstance(resolution_str, list) and len(resolution_str) == 1:
            resolution_str = resolution_str[0]
        parts = resolution_str.split(',')
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("Resolution must contain exactly three integers separated by commas.")
        return tuple(int(part.strip()) for part in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("Resolution must contain only integers separated by commas.")


def write_to_zarr(outfile, resolution, shape, transpose=True, offset=(0, 0, 0), format='zarr', chunks=(64, 64, 64), dtype=np.uint8):
    print(f"Debug - Initializing {format} file at {outfile}")
    print(f"Debug - Data shape to create: {shape}")

    try:
        if format == 'n5':
            store = zarr.N5FSStore(outfile)
            file_ = zarr.open(store, mode='a')
        else:
            file_ = zarr.open(outfile, "a")

        # When we create the dataset, we need to consider the transposition if it will be applied
        final_shape = shape
        if transpose:
            # If we'll be transposing the data during chunk writing, account for that in the shape
            final_shape = (shape[2], shape[1], shape[0])
            print(f"Debug - Transposed shape will be: {final_shape}")

        print(f"Debug - Creating dataset with shape {final_shape} and chunks {chunks}")
        # Create the dataset with chunking enabled - ensure we're passing the right shape
        if "volumes/s0" not in file_:
            file_.create_dataset("volumes/s0",
                                 chunks=chunks,
                                 overwrite=True,
                                 shape=final_shape,
                                 dtype=dtype,
                                 compression='gzip') # use compression = zarr.GZip(level=5) if simple 'gzip'does not work 

            file_["volumes/s0"].attrs["resolution"] = resolution
            file_["volumes/s0"].attrs["offset"] = offset
            print(f"Debug - Dataset created with attributes set")
        else:
            print(f"Debug - Dataset volumes/s0 already exists")

        print(f"Debug - File structure initialized successfully at: {outfile}")
        return file_
    except Exception as e:
        print(f"Error in write_to_zarr: {str(e)}")
        traceback.print_exc()
        return None


def write_chunk(chunk_data, chunk_bbox, output_bbox, outfile, resolution, transpose=True, offset=(0, 0, 0), format='zarr'):
    """
    Write a chunk of data to the output file.

    Parameters:
    - chunk_data: The data to write
    - chunk_bbox: The original bounding box of the chunk in source space
    - output_bbox: The overall bounding box in source space that we want to map to the output file
    - outfile: Path to output file
    - resolution: Resolution of the data
    - transpose: Whether to transpose from XYZ to ZYX
    - offset: Offset for the output file
    - format: File format ('zarr' or 'n5')
    """
    try:
        print(f"Debug - Starting write_chunk for source bbox: {chunk_bbox}")
        print(f"Debug - Chunk data shape: {chunk_data.shape}, dtype: {chunk_data.dtype}")

        if format == 'n5':
            store = zarr.N5FSStore(outfile)
            file_ = zarr.open(store, mode='a')
        else:
            file_ = zarr.open(outfile, mode='a')

        # Check if the dataset exists
        if "volumes/s0" not in file_:
            print(f"Error: volumes/s0 not found in {outfile}")
            return False

        raw = file_["volumes/s0"]

        # First transpose the data if needed - do this BEFORE calculating coordinates
        if transpose:
            chunk_data = np.transpose(chunk_data, (2, 1, 0))
            print(f"Debug - Data shape after transpose: {chunk_data.shape}")

        # Calculate the coordinate mapping from source bbox to output file coordinates
        src_bbox_start = output_bbox[0]  # Start of the overall source bounding box

        # Calculate where this chunk should be placed in the output array
        # by determining its position relative to the overall bounding box
        relative_x_start = chunk_bbox.minpt[0] - src_bbox_start[0]
        relative_y_start = chunk_bbox.minpt[1] - src_bbox_start[1]
        relative_z_start = chunk_bbox.minpt[2] - src_bbox_start[2]

        relative_x_end = relative_x_start + (chunk_bbox.maxpt[0] - chunk_bbox.minpt[0])
        relative_y_end = relative_y_start + (chunk_bbox.maxpt[1] - chunk_bbox.minpt[1])
        relative_z_end = relative_z_start + (chunk_bbox.maxpt[2] - chunk_bbox.minpt[2])

        # Make sure we don't go out of bounds
        relative_x_end = min(relative_x_end, raw.shape[2] if transpose else raw.shape[0])
        relative_y_end = min(relative_y_end, raw.shape[1])
        relative_z_end = min(relative_z_end, raw.shape[0] if transpose else raw.shape[2])

        # Skip if completely out of bounds
        if relative_x_start >= raw.shape[2] if transpose else raw.shape[0]:
            print(f"Warning: X dimension completely out of bounds. Skipping chunk.")
            return False
        if relative_y_start >= raw.shape[1]:
            print(f"Warning: Y dimension completely out of bounds. Skipping chunk.")
            return False
        if relative_z_start >= raw.shape[0] if transpose else raw.shape[2]:
            print(f"Warning: Z dimension completely out of bounds. Skipping chunk.")
            return False

        # Ensure we don't have negative coordinates
        relative_x_start = max(0, relative_x_start)
        relative_y_start = max(0, relative_y_start)
        relative_z_start = max(0, relative_z_start)

        # Create the slice coordinates in the appropriate format depending on transpose setting
        if transpose:
            # When transposing from XYZ to ZYX
            chunk_coords = (
                slice(relative_z_start, relative_z_end),  # Z
                slice(relative_y_start, relative_y_end),  # Y
                slice(relative_x_start, relative_x_end)   # X
            )
        else:
            chunk_coords = (
                slice(relative_x_start, relative_x_end),  # X
                slice(relative_y_start, relative_y_end),  # Y
                slice(relative_z_start, relative_z_end)   # Z
            )

        print(f"Debug - Output coordinates: {chunk_coords}")

        # Calculate how much of the chunk data we need to use
        # (in case we're at the edge of the output volume)
        data_x_slice = slice(0, relative_x_end - relative_x_start)
        data_y_slice = slice(0, relative_y_end - relative_y_start)
        data_z_slice = slice(0, relative_z_end - relative_z_start)

        if transpose:
            data_slices = (data_z_slice, data_y_slice, data_x_slice)
        else:
            data_slices = (data_x_slice, data_y_slice, data_z_slice)

        print(f"Debug - Data slices: {data_slices}")

        # Extract the relevant portion of data
        trimmed_data = chunk_data[data_slices]
        print(f"Debug - Trimmed data shape: {trimmed_data.shape}")

        # Check if we have valid data
        if np.prod(trimmed_data.shape) == 0:
            print("Error: Trimmed data has zero elements. Skipping this chunk.")
            return False

        try:
            # Convert to required dtype
            data_to_write = trimmed_data.astype(np.uint8)
            # Write the data
            raw[chunk_coords] = data_to_write
            print(f"Debug - Write operation completed successfully")
            return True
        except Exception as e:
            print(f"Error during write operation: {str(e)}")
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"Exception in write_chunk: {str(e)}")
        traceback.print_exc()
        return False


def download_and_process_chunk(vol, bbox_chunk, output_bbox, output_file, res, format='zarr', transpose=True, offset=(0, 0, 0), mip=0):
    """Download and process a single chunk directly"""
    try:
        print(f"Downloading chunk with bounds: {bbox_chunk}")
        files = vol.download(bbox_chunk, mip=mip)
        data = np.squeeze(files.data)
        print(f"Downloaded chunk of shape {data.shape}, min: {data.min()}, max: {data.max()}")

        # Write the chunk
        success = write_chunk(
            data,
            bbox_chunk,
            output_bbox,
            output_file,
            res,
            transpose=transpose,
            offset=offset,
            format=format
        )

        return success
    except Exception as e:
        print(f"Error in download_and_process_chunk: {str(e)}")
        traceback.print_exc()
        return False


def read_shards_as_cloudvolume(filename, args, bbox_start=(0, 0, 0), bbox_end=(1024, 1024, 1024), mip=0):
    """Process large data by downloading and processing smaller chunks directly"""
    start_time = time.time()
    vol = CloudVolume(f"precomputed://file://{filename}", fill_missing=True)
    print(f"Volume info {vol.info}")
    print(f"Volume shape {vol.shape}")
    print(f"Volume grid size {vol.image.grid_size()}")

    # Calculate the full shape for initialization
    full_shape = (
        bbox_end[0] - bbox_start[0],
        bbox_end[1] - bbox_start[1],
        bbox_end[2] - bbox_start[2]
    )

    print(f"Full bounding box: {bbox_start} to {bbox_end}, shape: {full_shape}")
    print(f"This would require approximately {full_shape[0] * full_shape[1] * full_shape[2] / (1024**3):.2f} GB of memory")

    # Initialize the output file with the correct shape
    chunk_size = args.chunk_size if hasattr(args, 'chunk_size') else (64, 64, 64)
    print("Initializing output file...")
    write_to_zarr(outfile=args.of,
                  shape=full_shape,  # Pass the shape, not the data
                  resolution=resolution_tuple(args.res),
                  transpose=args.trans,
                  offset=resolution_tuple(args.offset),
                  format=args.format,
                  chunks=chunk_size)

    # Calculate processing chunks based on max_chunk_size
    max_chunk_size = args.max_chunk_size if hasattr(args, 'max_chunk_size') else (512, 512, 512)

    # Calculate number of chunks needed in each dimension
    chunks_x = math.ceil((bbox_end[0] - bbox_start[0]) / max_chunk_size[0])
    chunks_y = math.ceil((bbox_end[1] - bbox_start[1]) / max_chunk_size[1])
    chunks_z = math.ceil((bbox_end[2] - bbox_start[2]) / max_chunk_size[2])

    print(f"Processing in {chunks_x}x{chunks_y}x{chunks_z} chunks")

    # Create bounding boxes for each processing chunk
    chunk_bboxes = []
    for x in range(chunks_x):
        x_start = bbox_start[0] + x * max_chunk_size[0]
        x_end = min(bbox_start[0] + (x + 1) * max_chunk_size[0], bbox_end[0])

        for y in range(chunks_y):
            y_start = bbox_start[1] + y * max_chunk_size[1]
            y_end = min(bbox_start[1] + (y + 1) * max_chunk_size[1], bbox_end[1])

            for z in range(chunks_z):
                z_start = bbox_start[2] + z * max_chunk_size[2]
                z_end = min(bbox_start[2] + (z + 1) * max_chunk_size[2], bbox_end[2])

                chunk_bbox = cloudvolume.Bbox((x_start, y_start, z_start), (x_end, y_end, z_end))
                chunk_bboxes.append(chunk_bbox)

    # Store the overall output bounding box for coordinate mapping
    output_bbox = (bbox_start, bbox_end)

    total_chunks = len(chunk_bboxes)
    print(f"Processing {total_chunks} chunks in total")

    # Option for sequential processing for debugging
    debug_sequential = True if hasattr(args, 'debug') and args.debug else False

    if debug_sequential:
        print("Running in sequential debug mode")
        for i, bbox_chunk in enumerate(chunk_bboxes):
            print(f"Processing chunk {i+1}/{total_chunks}")
            success = download_and_process_chunk(
                vol,
                bbox_chunk,
                output_bbox,
                args.of,
                resolution_tuple(args.res),
                format=args.format,
                transpose=args.trans,
                offset=resolution_tuple(args.offset),
                mip=mip
            )
            if not success:
                print(f"Failed to process chunk {i+1}")
    else:
        # Process chunks in parallel
        process_chunk_partial = partial(
            download_and_process_chunk,
            vol,
            output_bbox=output_bbox,
            output_file=args.of,
            res=resolution_tuple(args.res),
            format=args.format,
            transpose=args.trans,
            offset=resolution_tuple(args.offset),
            mip=mip
        )

        max_workers = min(os.cpu_count(), args.max_workers) if hasattr(args, 'max_workers') else os.cpu_count()
        print(f"Using {max_workers} parallel workers")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for bbox_chunk in chunk_bboxes:
                future = executor.submit(process_chunk_partial, bbox_chunk)
                futures.append(future)

            # Monitor progress
            with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
                completed = 0
                while completed < total_chunks:
                    done = sum(1 for f in futures if f.done())
                    pbar.update(done - completed)
                    completed = done
                    time.sleep(0.1)

    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Average time per chunk: {elapsed_time / total_chunks:.2f} seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', required=True, help="Input path to the neuroglancer multiscale file")
    parser.add_argument('-of', required=True, help="Output path for the zarr/n5 file")
    parser.add_argument('-mip', default=0, type=int, help="Scale level to save as zarr")
    parser.add_argument('-trans', default=True, type=bool, help="Transpose data from xyz to zyx")
    parser.add_argument('-offset', nargs='+', default="0,0,0", help="Offset as Z,Y,X")
    parser.add_argument('-res', nargs='+', default="8,8,8", help="Resolution as Z,Y,X")
    parser.add_argument('-bbox_start', nargs='+', default="0,0,0", help="Boundary box start as X,Y,Z")
    parser.add_argument('-bbox_end', nargs='+', default="4416,2912,2848", help="Boundary box end as X,Y,Z")
    parser.add_argument('-format', default='zarr', choices=['zarr', 'n5'], help="Output format")
    parser.add_argument('-chunk_size', nargs='+', default="64,64,64", help="Chunk size for zarr/n5 as Z,Y,X")
    parser.add_argument('-max_chunk_size', nargs='+', default="512,512,512",
                        help="Maximum size of chunks to download at once as X,Y,Z")
    parser.add_argument('-max_workers', type=int, default=os.cpu_count(),
                        help="Maximum number of parallel workers")
    parser.add_argument('-debug', action='store_true', help="Run in sequential debug mode")

    args = parser.parse_args()

    # Convert tuple arguments
    args.chunk_size = resolution_tuple(args.chunk_size)
    args.max_chunk_size = resolution_tuple(args.max_chunk_size)

    # Start processing
    read_shards_as_cloudvolume(args.f, args,
                               bbox_start=resolution_tuple(args.bbox_start),
                               bbox_end=resolution_tuple(args.bbox_end),
                               mip=args.mip)


if __name__ == '__main__':
    main()
