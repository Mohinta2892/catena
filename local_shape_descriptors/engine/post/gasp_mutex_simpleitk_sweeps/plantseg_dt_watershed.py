import argparse
import sqlite3
import itertools
import numpy as np
import zarr
import tensorstore as ts
import tifffile
import multiprocessing as mp
import os
from tqdm import tqdm
from plantseg.functionals.segmentation import dt_watershed

# --- Configuration ---
# Processing Chunk: Large blocks for efficient Python overhead (Z, Y, X)
DEFAULT_PROC_CHUNK = (100, 512, 512)  
DEFAULT_PADDING = (10, 64, 64)
DB_NAME = "processing_state.db"

# Output Chunk: Small blocks for efficient Neuroglancer rendering (X, Y, Z)
TS_INNER_CHUNK = [64, 64, 64] 

# --- Globals for Workers ---
global_zarr_path = None
global_zarr_key = None
global_ts_spec = None
global_reduction = None
global_proc_chunk = None
global_padding = None
global_shape_spatial = None
global_offset = None
global_invert = False  # <--- New Global

# --- Database Management ---
def init_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            z_start INT, y_start INT, x_start INT,
            z_end INT, y_end INT, x_end INT,
            status TEXT DEFAULT 'pending'
        )
    ''')
    conn.commit()
    return conn

def scout_active_chunks(shape, chunk_size, mask_path=None):
    grid = [range(0, s, c) for s, c in zip(shape, chunk_size)]
    all_chunks = list(itertools.product(*grid))
    
    if not mask_path:
        return all_chunks

    print(f"Scouting mask at {mask_path}...")
    try:
        mask_vol = tifffile.imread(mask_path)
        if mask_vol.ndim != 3:
            print(f"Warning: Mask dimensions {mask_vol.shape} do not match 3D expectation. Ignoring mask.")
            return all_chunks
    except Exception as e:
        print(f"Error reading mask TIFF: {e}. Proceeding without mask.")
        return all_chunks

    active_chunks = []
    print("Filtering chunks based on mask content...")
    
    for (z, y, x) in tqdm(all_chunks, desc="Scouting"):
        z_e = min(z + chunk_size[0], shape[0])
        y_e = min(y + chunk_size[1], shape[1])
        x_e = min(x + chunk_size[2], shape[2])
        
        mz_e = min(z_e, mask_vol.shape[0])
        my_e = min(y_e, mask_vol.shape[1])
        mx_e = min(x_e, mask_vol.shape[2])
        
        if z >= mask_vol.shape[0] or y >= mask_vol.shape[1] or x >= mask_vol.shape[2]:
            continue

        mask_chunk = mask_vol[z:mz_e, y:my_e, x:mx_e]
        
        if np.any(mask_chunk):
            active_chunks.append((z, y, x))
            
    print(f"Scout complete. {len(active_chunks)} / {len(all_chunks)} chunks selected.")
    return active_chunks

def populate_db(conn, shape, chunk_size, mask_path=None):
    c = conn.cursor()
    c.execute('SELECT count(*) FROM chunks')
    if c.fetchone()[0] > 0:
        print("Database already populated. Resuming...")
        return

    active_coords = scout_active_chunks(shape, chunk_size, mask_path)
    
    entries = []
    for (z, y, x) in active_coords:
        z_e = min(z + chunk_size[0], shape[0])
        y_e = min(y + chunk_size[1], shape[1])
        x_e = min(x + chunk_size[2], shape[2])
        
        chunk_id = f"{z}_{y}_{x}"
        entries.append((chunk_id, z, y, x, z_e, y_e, x_e))

    print(f"Inserting {len(entries)} chunks into database...")
    c.executemany('INSERT INTO chunks (id, z_start, y_start, x_start, z_end, y_end, x_end) VALUES (?,?,?,?,?,?,?)', entries)
    conn.commit()

# --- TensorStore Spec ---
def get_sharded_spec(path, shape, resolution, offset_zyx, dtype="uint64"):
    # Convert ZYX (numpy) to XYZ (Neuroglancer)
    shape_xyz = [shape[2], shape[1], shape[0]]
    offset_xyz = [offset_zyx[2], offset_zyx[1], offset_zyx[0]]
    
    sharding_spec = {
        '@type': 'neuroglancer_uint64_sharded_v1',
        'data_encoding': 'gzip',
        'hash': 'identity',
        'minishard_bits': 6,
        'minishard_index_encoding': 'gzip',
        'preshift_bits': 9,
        'shard_bits': 13,
    }

    inclusive_min = [*offset_xyz, 0]
    exclusive_max = [offset_xyz[0] + shape_xyz[0], 
                     offset_xyz[1] + shape_xyz[1], 
                     offset_xyz[2] + shape_xyz[2], 
                     1]

    return {
        'driver': 'neuroglancer_precomputed',
        'kvstore': {
            'driver': 'file',
            'path': path,
        },
        'scale_metadata': {
            'size': shape_xyz,
            'chunk_size': TS_INNER_CHUNK,
            'resolution': resolution,
            'voxel_offset': offset_xyz,
            'sharding': sharding_spec,
            'encoding': 'compressed_segmentation',
            'compressed_segmentation_block_size': [8, 8, 8]
        },
        'dtype': dtype,
        'schema': {
            'domain': {
                'labels': ['x', 'y', 'z', 'channel'],
                'inclusive_min': inclusive_min,
                'exclusive_max': exclusive_max
            }
        },
        'context': {
            'cache_pool': {'total_bytes_limit': 100_000_000},
            'data_copy_concurrency': {'limit': 4},
        },
    }

# --- Worker Functions ---
def worker_initializer(zarr_path, zarr_key, ts_spec, reduction, proc_chunk, padding, shape_spatial, offset, invert):
    global global_zarr_path, global_zarr_key, global_ts_spec
    global global_reduction, global_proc_chunk, global_padding, global_shape_spatial, global_offset, global_invert
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    global_zarr_path = zarr_path
    global_zarr_key = zarr_key
    global_ts_spec = ts_spec
    global_reduction = reduction
    global_proc_chunk = proc_chunk
    global_padding = padding
    global_shape_spatial = shape_spatial
    global_offset = offset
    global_invert = invert  # <--- Store Invert Flag

def process_chunk_task(task):
    cid, z, y, x, ze, ye, xe = task
    
    try:
        z_in = zarr.open(global_zarr_path, mode='r')
        ds_in = z_in[global_zarr_key] if global_zarr_key else z_in
        dataset = ts.open(global_ts_spec, open=True, create=False).result()

        # 1. Padding
        z_p = max(0, z - global_padding[0])
        y_p = max(0, y - global_padding[1])
        x_p = max(0, x - global_padding[2])
        
        ze_p = min(global_shape_spatial[0], ze + global_padding[0])
        ye_p = min(global_shape_spatial[1], ye + global_padding[1])
        xe_p = min(global_shape_spatial[2], xe + global_padding[2])

        # 2. Load & Reduce
        full_shape = ds_in.shape
        
        if len(full_shape) == 3:
            pmap_chunk = ds_in[z_p:ze_p, y_p:ye_p, x_p:xe_p]
        elif len(full_shape) == 4:
            if full_shape[0] == 3: # (C, Z, Y, X)
                data_chunk = ds_in[:, z_p:ze_p, y_p:ye_p, x_p:xe_p]
                axis = 0
            else: # (Z, Y, X, C)
                data_chunk = ds_in[z_p:ze_p, y_p:ye_p, x_p:xe_p, :]
                axis = 3
            
            if global_reduction == 'max': pmap_chunk = np.max(data_chunk, axis=axis)
            elif global_reduction == 'mean': pmap_chunk = np.mean(data_chunk, axis=axis)
            elif global_reduction == 'median': pmap_chunk = np.median(data_chunk, axis=axis)
        else:
             return (cid, 'failed', f"Unexpected shape {full_shape}")

        pmap_chunk = pmap_chunk.astype(np.float32)

        # 3. Inversion (Optional)
        if global_invert:
            # Assumes data is 0..1. Turns (1=Inside) into (0=Inside) for Watershed
            pmap_chunk = 1.0 - pmap_chunk

        # 4. Watershed
        ws_chunk = dt_watershed(
            pmap_chunk, 
            threshold=0.5,     
            sigma_seeds=1.0, 
            stacked=False,     
            sigma_weights=2.0, 
            min_size=100
        )

        # 5. Global ID Offset
        linear_idx = (z // global_proc_chunk[0]) * 10000 + \
                     (y // global_proc_chunk[1]) * 100 + \
                     (x // global_proc_chunk[2])
        id_offset = int(linear_idx) * 10_000_000 
        
        ws_chunk = ws_chunk.astype(np.uint64)
        ws_chunk[ws_chunk > 0] += id_offset

        # 6. Crop Halo
        valid_z_start = z - z_p
        valid_y_start = y - y_p
        valid_x_start = x - x_p
        
        final_seg = ws_chunk[
            valid_z_start : valid_z_start + (ze - z),
            valid_y_start : valid_y_start + (ye - y),
            valid_x_start : valid_x_start + (xe - x)
        ]

        # 7. Write to TensorStore (XYZC) with Global Offset
        data_to_write = final_seg.transpose(2, 1, 0)[..., np.newaxis]
        
        off_z, off_y, off_x = global_offset
        
        gx_start = x + off_x
        gx_end = xe + off_x
        gy_start = y + off_y
        gy_end = ye + off_y
        gz_start = z + off_z
        gz_end = ze + off_z

        dataset[gx_start:gx_end, gy_start:gy_end, gz_start:gz_end, 0:1].write(data_to_write).result()

        return (cid, 'success', None)

    except Exception as e:
        return (cid, 'failed', str(e))

# --- Main Controller ---
def run_parallel(args):
    # 1. Open Input
    z_in = zarr.open(args.input, mode='r')
    ds_in = z_in[args.key] if args.key else z_in
    full_shape = ds_in.shape
    
    # 2. Detect Shape
    if len(full_shape) == 4:
        if full_shape[0] == 3: spatial_shape = full_shape[1:]
        elif full_shape[-1] == 3: spatial_shape = full_shape[:-1]
        else: raise ValueError(f"Unclear dimensions: {full_shape}")
    elif len(full_shape) == 3:
        spatial_shape = full_shape
    else:
        raise ValueError(f"Unsupported dimensions: {full_shape}")

    print(f"Volume Shape: {spatial_shape}")
    
    # 3. Setup DB
    conn = init_db(DB_NAME)
    populate_db(conn, spatial_shape, DEFAULT_PROC_CHUNK, args.mask)
    
    # 4. Parse Options
    if args.offset:
        offset_zyx = tuple(map(int, args.offset.split(',')))
    else:
        offset_zyx = (0, 0, 0)
        
    res = [int(x) for x in args.resolution.split(',')]
    
    # 5. Initialize TensorStore
    spec = get_sharded_spec(args.output, spatial_shape, res, offset_zyx)
    print(f"Initializing Sharded TS at {args.output}...")
    if args.invert:
        print("!! INVERSION ENABLED (1.0 - data) !!")
        
    ts.open(spec, create=True, open=True).result()

    # 6. Process
    cursor = conn.cursor()
    cursor.execute("SELECT id, z_start, y_start, x_start, z_end, y_end, x_end FROM chunks WHERE status='pending'")
    tasks = cursor.fetchall()
    
    if not tasks:
        print("No pending tasks.")
        return

    print(f"Processing {len(tasks)} chunks...")
    
    init_args = (
        args.input, args.key, spec, args.reduction, 
        DEFAULT_PROC_CHUNK, DEFAULT_PADDING, spatial_shape, offset_zyx, args.invert
    )

    with mp.Pool(processes=args.workers, initializer=worker_initializer, initargs=init_args) as pool:
        results = pool.imap_unordered(process_chunk_task, tasks, chunksize=1)
        batch_updates = []
        pbar = tqdm(results, total=len(tasks))
        
        for cid, status, error in pbar:
            if status == 'success':
                batch_updates.append((cid,))
            else:
                tqdm.write(f"Error in chunk {cid}: {error}")
            
            if len(batch_updates) >= 50:
                conn.executemany("UPDATE chunks SET status='done' WHERE id=?", batch_updates)
                conn.commit()
                batch_updates = []
        
        if batch_updates:
            conn.executemany("UPDATE chunks SET status='done' WHERE id=?", batch_updates)
            conn.commit()

    conn.close()
    print("Processing complete.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Parallel Chunked PlantSeg Watershed (Sharded)")
    parser.add_argument("--input", required=True, help="Path to zarr container")
    parser.add_argument("--key", required=True, help="Path inside zarr")
    parser.add_argument("--output", required=True, help="Output path for precomputed volume")
    parser.add_argument("--mask", help="Optional path to mask TIFF file")
    parser.add_argument("--reduction", default="max", choices=["max", "mean", "median"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resolution", default="8,8,8", help="Voxel size (x,y,z)")
    parser.add_argument("--offset", default=None, help="Global offset (z,y,x) e.g. '1000,2000,3000'")
    parser.add_argument("--invert", action='store_true', help="Invert affinities (1 - pmap) before watershed")
    
    args = parser.parse_args()
    
    run_parallel(args)
