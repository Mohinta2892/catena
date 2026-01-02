import os
import sys
import time
import numpy as np
import tensorstore as ts
import zarr
import multiprocessing as mp
from tqdm import tqdm
from common_seg import (
    TaskManager, parse_bounds_from_filename,
    INPUT_FILENAME, INPUT_ZARR_PATH, OUTPUT_PATH, DB_PATH,
    MODE, RESOLUTION, INPUT_CHUNK_SIZE, OUTPUT_INNER_CHUNK
)

# ================= CONFIGURATION =================

NUM_WORKERS = 16 

# ================= WORKER FUNCTIONS =================

global_source_zarr = None
global_dataset_dest = None
global_offset_xyz = None
global_mode = None

def worker_initializer(zarr_path, ts_spec, offset_xyz, mode):
    global global_source_zarr, global_dataset_dest, global_offset_xyz, global_mode
    
    # 1. Open Source
    try:
        global_source_zarr = zarr.open(zarr_path, mode='r')
    except Exception as e:
        print(f"!! Worker failed to open Zarr: {e}")

    # 2. Open Dest
    try:
        global_dataset_dest = ts.open(ts_spec, open=True, create=False).result()
    except Exception as e:
        print(f"!! Worker failed to open TensorStore: {e}")

    global_offset_xyz = offset_xyz
    global_mode = mode

def process_chunk(task):
    z, y, x = task
    src = global_source_zarr
    dest = global_dataset_dest
    off_xyz = global_offset_xyz
    mode = global_mode
    cz, cy, cx = INPUT_CHUNK_SIZE

    try:
        # --- READ ---
        z_end = min(z + cz, src.shape[0])
        y_end = min(y + cy, src.shape[1])
        x_end = min(x + cx, src.shape[2])

        # Read ZYX
        data_chunk = src[z:z_end, y:y_end, x:x_end]

        # --- TRANSFORM ---
        # 1. Transpose ZYX -> XYZ
        data_chunk = np.transpose(data_chunk, (2, 1, 0))
        
        # 2. Add Channel Dim: XYZ -> XYZC
        data_chunk = data_chunk[..., np.newaxis]

        # 3. Cast Type
        if mode == 'EM':
            if data_chunk.dtype != np.uint8:
                data_chunk = data_chunk.astype(np.uint8)
        else:
            if data_chunk.dtype != np.uint64:
                data_chunk = data_chunk.astype(np.uint64)

        # --- WRITE ---
        x_start_g = x + off_xyz[0]
        y_start_g = y + off_xyz[1]
        z_start_g = z + off_xyz[2]

        x_end_g = x_start_g + (x_end - x)
        y_end_g = y_start_g + (y_end - y)
        z_end_g = z_start_g + (z_end - z)

        # FIX: Use 0:1 instead of 0 to preserve 4D shape for the write
        # dest expects [x, y, z, c]
        dest[
            x_start_g : x_end_g,
            y_start_g : y_end_g,
            z_start_g : z_end_g,
            0:1 
        ].write(data_chunk).result()

        return (task, 'success', None)

    except Exception as e:
        return (task, 'failed', str(e))

# ================= MAIN CONTROLLER =================

def get_base_spec(shape_xyz, voxel_offset_xyz):
    
    sharding_spec = {
        '@type': 'neuroglancer_uint64_sharded_v1',
        'data_encoding': 'gzip',
        'hash': 'identity',
        'minishard_bits': 6,
        'minishard_index_encoding': 'gzip',
        'preshift_bits': 9,
        'shard_bits': 13,
    }

    if MODE == 'EM':
        encoding = 'jpeg'
        block_size = None
        dtype_val = 'uint8'
    else:
        encoding = 'compressed_segmentation'
        block_size = [8, 8, 8]
        dtype_val = 'uint64'

    # Domain Bounds
    inclusive_min = [voxel_offset_xyz[0], voxel_offset_xyz[1], voxel_offset_xyz[2], 0]
    exclusive_max = [
        voxel_offset_xyz[0] + shape_xyz[0],
        voxel_offset_xyz[1] + shape_xyz[1],
        voxel_offset_xyz[2] + shape_xyz[2],
        1
    ]

    scale_meta = {
        'size': shape_xyz, 
        'chunk_size': OUTPUT_INNER_CHUNK,
        'encoding': encoding,
        'key': '8.0x8.0x8.0',
        'resolution': RESOLUTION,
        'voxel_offset': voxel_offset_xyz,
        'sharding': sharding_spec
    }

    if block_size:
        scale_meta['compressed_segmentation_block_size'] = block_size

    spec = {
        'driver': 'neuroglancer_precomputed',
        'kvstore': {'driver': 'file', 'path': OUTPUT_PATH},
        'context': {
            'cache_pool': {'total_bytes_limit': 100_000_000},
            'data_copy_concurrency': {'limit': 1},
        },
        'scale_metadata': scale_meta,
        'dtype': dtype_val,
        'schema': {
            'domain': {
                'inclusive_min': inclusive_min,
                'exclusive_max': exclusive_max,
                'labels': ['x', 'y', 'z', 'channel']
            }
        }
    }
    
    return spec

def run_multiprocess_conversion():
    # 1. Setup & DB
    db = TaskManager(DB_PATH)
    offsets = parse_bounds_from_filename(INPUT_FILENAME)
    
    # 2. Get Input Info
    temp_zarr = zarr.open(INPUT_ZARR_PATH, mode='r')
    shape_zyx = temp_zarr.shape
    shape_xyz = [shape_zyx[2], shape_zyx[1], shape_zyx[0]]
    offset_xyz = [offsets[2], offsets[1], offsets[0]]

    # 3. Prepare Specs
    ts_spec = get_base_spec(shape_xyz, offset_xyz)
    
    # 4. Initialize Output (Main Process)
    print("Initializing TensorStore volume...")
    try:
        ts.open(ts_spec, create=True, delete_existing=False).result()
    except Exception as e:
        print(f"CRITICAL ERROR initializing TensorStore: {e}")
        sys.exit(1)

    # 5. Fetch Tasks
    print("Fetching pending tasks from DB...")
    db.cursor.execute("SELECT z, y, x FROM chunks WHERE status='pending'")
    tasks = db.cursor.fetchall()
    
    if not tasks:
        print("No pending tasks found. Run scout.py first.")
        return

    # ================= TEST ONE CHUNK =================
    print("\n--- RUNNING SINGLE CHUNK TEST ---")
    test_task = tasks[0]
    print(f"Testing chunk: {test_task}")
    
    # Initialize globals manually for the main process test
    worker_initializer(INPUT_ZARR_PATH, ts_spec, offset_xyz, MODE)
    
    # Run synchronously
    test_result = process_chunk(test_task)
    
    if test_result[1] == 'failed':
        print(f"\nFATAL ERROR in single chunk test: {test_result[2]}")
        print("Fix this error before running multiprocessing.")
        sys.exit(1)
    else:
        print("Single chunk test SUCCESS! Volume should have data now.")
        print("Check folder size now before proceeding.")
        # Optional: update the DB for this one task
        db.update_status(test_result[0], 'success')
        # Remove it from the list so we don't process it again in the pool
        tasks = tasks[1:]
    
    print("---------------------------------\n")
    # ==================================================

    print(f"Starting Multiprocessing Pool with {NUM_WORKERS} workers.")
    print(f"Remaining tasks: {len(tasks)}")

    # 6. Start Pool
    with mp.Pool(
        processes=NUM_WORKERS, 
        initializer=worker_initializer, 
        initargs=(INPUT_ZARR_PATH, ts_spec, offset_xyz, MODE)
    ) as pool:
        
        results = pool.imap_unordered(process_chunk, tasks, chunksize=1)
        batch_updates = []
        
        # Verbose progress bar
        pbar = tqdm(results, total=len(tasks))
        
        for task, status, err_msg in pbar:
            if status == 'success':
                batch_updates.append((status, task[0], task[1], task[2]))
            else:
                # PRINT THE ERROR
                tqdm.write(f"ERROR on {task}: {err_msg}")
                batch_updates.append((status, task[0], task[1], task[2]))
            
            if len(batch_updates) >= 100:
                db.cursor.executemany(
                    "UPDATE chunks SET status=? WHERE z=? AND y=? AND x=?", 
                    batch_updates
                )
                db.conn.commit()
                batch_updates = []

        if batch_updates:
            db.cursor.executemany(
                "UPDATE chunks SET status=? WHERE z=? AND y=? AND x=?", 
                batch_updates
            )
            db.conn.commit()

    print("Multiprocessing conversion complete.")
    print("Final Stats:", db.get_stats())

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    run_multiprocess_conversion()
