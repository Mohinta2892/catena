"""Runs with:

python read_sharded_data.py -f /path/to/input -of /path/to/output.n5 -format n5 -chunk_size 64,64,64

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

def write_to_zarr(outfile, data, resolution, transpose=True, offset=(0, 0, 0), format='zarr', chunks=(64, 64, 64)):
    if format == 'n5':
        store = zarr.N5Store(outfile)
        file_ = zarr.open(store, mode='a')
    else:
        file_ = zarr.open(outfile, "a")

    # per our convention we should save the data as zyx
    if transpose:
        data = np.transpose(data, (2, 1, 0))

    # Create the dataset with chunking enabled
    file_.create_dataset("volumes/s0", 
                        data=data,
                        chunks=chunks,
                        overwrite=True)
    
    file_["volumes/raw"].attrs["resolution"] = resolution
    file_["volumes/raw"].attrs["offset"] = offset

    print(f" Saved here: {outfile}")

def write_chunk(chunk_data, chunk_coords, outfile, resolution, transpose=True, offset=(0, 0, 0), format='zarr'):
    if format == 'n5':
        store = zarr.N5Store(outfile)
        file_ = zarr.open(store, mode='a')
    else:
        file_ = zarr.open(outfile, mode='a')
    
    if transpose:
        chunk_data = np.transpose(chunk_data, (2, 1, 0))
    
    raw = file_["volumes/s0"]
    raw[chunk_coords] = chunk_data
    return True

def read_shards_as_cloudvolume(filename, args, bbox_start=(0, 0, 0), bbox_end=(1024, 1024, 1024), mip=1):
    start_time = time.time()
    vol = CloudVolume(f"precomputed://file://{filename}", fill_missing=True)
    print(f"volume info {vol.info}")
    print(f"volume shape {vol.shape}")
    print(f"volume grid size {vol.image.grid_size()}")

    print("Downloading data...")
    bbox = cloudvolume.Bbox(bbox_start, bbox_end)
    files = vol.download(bbox, mip=mip)
    data = np.squeeze(files.data)
    resolution = files.resolution

    # Initialize the output file with metadata
    chunk_size = args.chunk_size if hasattr(args, 'chunk_size') else (64, 64, 64)
    print("Initializing output file...")
    write_to_zarr(outfile=args.of, 
                  data=np.zeros_like(data), 
                  resolution=resolution_tuple(args.res),
                  transpose=args.trans,
                  offset=resolution_tuple(args.offset),
                  format=args.format,
                  chunks=chunk_size)

    # Calculate chunks for parallel processing
    z_chunks = range(0, data.shape[0], chunk_size[0])
    y_chunks = range(0, data.shape[1], chunk_size[1])
    x_chunks = range(0, data.shape[2], chunk_size[2])

    # Prepare chunk coordinates and data
    chunks_to_process = []
    for z in z_chunks:
        for y in y_chunks:
            for x in x_chunks:
                z_end = min(z + chunk_size[0], data.shape[0])
                y_end = min(y + chunk_size[1], data.shape[1])
                x_end = min(x + chunk_size[2], data.shape[2])
                
                chunk_data = data[z:z_end, y:y_end, x:x_end]
                chunk_coords = (slice(z, z_end), slice(y, y_end), slice(x, x_end))
                chunks_to_process.append((chunk_data, chunk_coords))

    total_chunks = len(chunks_to_process)
    print(f"\nProcessing {total_chunks} chunks in parallel...")
    print(f"Estimated memory usage: {data.nbytes / 1e9:.2f} GB")
    
    # Process chunks in parallel with progress bar
    write_chunk_partial = partial(
        write_chunk,
        outfile=args.of,
        resolution=resolution_tuple(args.res),
        transpose=args.trans,
        offset=resolution_tuple(args.offset),
        format=args.format
    )

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for chunk_data, chunk_coords in chunks_to_process:
            future = executor.submit(write_chunk_partial, chunk_data, chunk_coords)
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
    print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
    print(f"Average time per chunk: {elapsed_time/total_chunks:.2f} seconds")

def main():
    parser = argparse.ArgumentParser()
    # ... existing code ...
    parser.add_argument('-format', default='zarr', choices=['zarr', 'n5'], 
                       help="Format to save the data (zarr or n5)")
    parser.add_argument('-chunk_size', nargs='+', default="64,64,64",
                       help="Chunk size for parallel processing as Z,Y,X")
    
    args = parser.parse_args()
    
    # Convert chunk_size to tuple if provided
    if hasattr(args, 'chunk_size'):
        args.chunk_size = resolution_tuple(args.chunk_size)

    read_shards_as_cloudvolume(args.f, args, 
                              bbox_start=resolution_tuple(args.bbox_start),
                              bbox_end=resolution_tuple(args.bbox_end), 
                              mip=args.mip)

if __name__ == '__main__':
    main()