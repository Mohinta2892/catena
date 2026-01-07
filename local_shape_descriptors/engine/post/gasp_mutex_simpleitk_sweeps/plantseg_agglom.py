import argparse
import sqlite3
import itertools
import os
import pickle
import multiprocessing as mp
import numpy as np
import zarr
import tensorstore as ts
from tqdm import tqdm

# --- PlantSeg / Elf Imports ---
try:
    import nifty.graph as ngraph
    # We use elf wrappers to handle the specific nifty factory names automatically
    from elf.segmentation.multicut import multicut_kernighan_lin, multicut_gaec
    from elf.segmentation.lifted_multicut import lifted_multicut_gaec
except ImportError:
    raise ImportError("Elf/Nifty not found. Please ensure you are in the PlantSeg environment.")

# --- Configuration ---
CHUNK_SIZE = (100, 512, 512)
DB_NAME = "agglomeration_state.db"
TEMP_GRAPH_DIR = "temp_graph_parts"

# --- Database Management ---
def init_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS graph_tasks (
            id TEXT PRIMARY KEY,
            z INT, y INT, x INT,
            z_e INT, y_e INT, x_e INT,
            status TEXT DEFAULT 'pending',
            pickle_path TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS write_tasks (
            id TEXT PRIMARY KEY,
            z INT, y INT, x INT,
            z_e INT, y_e INT, x_e INT,
            status TEXT DEFAULT 'pending'
        )
    ''')
    conn.commit()
    return conn

def populate_db(conn, shape, chunk_size, table_name):
    c = conn.cursor()
    c.execute(f'SELECT count(*) FROM {table_name}')
    if c.fetchone()[0] > 0:
        return 

    grid = [range(0, s, c) for s, c in zip(shape, chunk_size)]
    chunks = list(itertools.product(*grid))
    
    entries = []
    for (z, y, x) in chunks:
        z_e = min(z + chunk_size[0], shape[0])
        y_e = min(y + chunk_size[1], shape[1])
        x_e = min(x + chunk_size[2], shape[2])
        chunk_id = f"{z}_{y}_{x}"
        entries.append((chunk_id, z, y, x, z_e, y_e, x_e))

    c.executemany(f'INSERT INTO {table_name} (id, z, y, x, z_e, y_e, x_e) VALUES (?,?,?,?,?,?,?)', entries)
    conn.commit()
    print(f"Populated {table_name} with {len(entries)} tasks.")

# --- TensorStore Specs ---
def get_input_spec(path):
    return {'driver': 'neuroglancer_precomputed', 'kvstore': {'driver': 'file', 'path': path}}

def get_output_sharded_spec(path, shape, chunk_size, resolution, offset_xyz, dtype="uint64"):
    shape_xyz = [shape[2], shape[1], shape[0]]
    chunk_xyz = [chunk_size[2], chunk_size[1], chunk_size[0]]
    
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
        'kvstore': {'driver': 'file', 'path': path},
        'scale_metadata': {
            'size': shape_xyz,
            'chunk_size': chunk_xyz,
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

# --- Phase 1: Graph Extraction Worker ---

def extract_edges_wrapper(args):
    return extract_edges_worker(*args)

def extract_edges_worker(task, aff_path, aff_key, seg_path, temp_dir, offset_zyx, invert_aff):
    try:
        cid, z, y, x, ze, ye, xe = task
        
        # 1. Open Inputs
        z_root = zarr.open(aff_path, mode='r')
        ds_aff = z_root[aff_key]
        spec = get_input_spec(seg_path)
        ds_seg = ts.open(spec, open=True).result()

        full_shape = ds_aff.shape
        if len(full_shape) == 4 and full_shape[0] == 3:
            channel_dim = 0
            spatial_shape = full_shape[1:]
        elif len(full_shape) == 4 and full_shape[-1] == 3:
            channel_dim = 3
            spatial_shape = full_shape[:-1]
        else:
             channel_dim = 0 
             spatial_shape = full_shape

        # 2. Coordinate Handling
        off_z, off_y, off_x = offset_zyx
        gz, gy, gx = z + off_z, y + off_y, x + off_x
        gze, gye, gxe = ze + off_z, ye + off_y, xe + off_x
        
        # Read Seg (Global Coords)
        seg_data = ds_seg[gx:gxe+1, gy:gye+1, gz:gze+1, 0:1].read().result()
        seg_data = seg_data.squeeze().transpose(2, 1, 0) 

        # Read Affinities (Local Coords)
        ze_p = min(spatial_shape[0], ze + 1)
        ye_p = min(spatial_shape[1], ye + 1)
        xe_p = min(spatial_shape[2], xe + 1)
        
        if channel_dim == 0:
            aff_data = ds_aff[:, z:ze_p, y:ye_p, x:xe_p]
        else:
            aff_data = ds_aff[z:ze_p, y:ye_p, x:xe_p, :]
            aff_data = np.moveaxis(aff_data, -1, 0) 
        
        # Clip to valid intersection
        d_z = min(seg_data.shape[0], aff_data.shape[1])
        d_y = min(seg_data.shape[1], aff_data.shape[2])
        d_x = min(seg_data.shape[2], aff_data.shape[3])
        
        seg_data = seg_data[:d_z, :d_y, :d_x]
        aff_data = aff_data[:, :d_z, :d_y, :d_x]

        # 3. Normalization & Inversion
        if aff_data.max() > 1.1:
            aff_data = aff_data.astype(np.float32) / 255.0
        else:
            aff_data = aff_data.astype(np.float32)

        if invert_aff:
            # Turns (1=Inside) into (0=Inside) for cost calculation
            aff_data = 1.0 - aff_data
            
        # 4. Compute Edges
        edges = []
        weights = []

        def process_axis(slice_main, slice_shift, aff_idx):
            u = seg_data[slice_main]
            v = seg_data[slice_shift]
            mask = (u != v) & (u != 0) & (v != 0)
            if np.any(mask):
                u_valid = u[mask]
                v_valid = v[mask]
                aff_valid = aff_data[aff_idx][slice_main][mask]
                stacked = np.stack([u_valid, v_valid], axis=1)
                stacked.sort(axis=1)
                return stacked, aff_valid
            return None, None

        if seg_data.shape[0] > 1:
            e, w = process_axis(slice(0, -1), slice(1, None), 0)
            if e is not None: edges.append(e); weights.append(w)
        if seg_data.shape[1] > 1:
            e, w = process_axis((slice(None), slice(0, -1), slice(None)), (slice(None), slice(1, None), slice(None)), 1)
            if e is not None: edges.append(e); weights.append(w)
        if seg_data.shape[2] > 1:
            e, w = process_axis((slice(None), slice(None), slice(0, -1)), (slice(None), slice(None), slice(1, None)), 2)
            if e is not None: edges.append(e); weights.append(w)

        if not edges:
            return (cid, 'skipped', None)

        all_edges = np.concatenate(edges)
        all_weights = np.concatenate(weights)
        
        unique_edges, inverse = np.unique(all_edges, axis=0, return_inverse=True)
        sum_weights = np.bincount(inverse, weights=all_weights)
        edge_counts = np.bincount(inverse)
        
        out_file = os.path.join(temp_dir, f"{cid}.pkl")
        with open(out_file, 'wb') as f:
            pickle.dump({'edges': unique_edges, 'sum': sum_weights, 'count': edge_counts}, f)

        return (cid, 'success', out_file)

    except Exception as e:
        return (cid, 'failed', str(e))

# --- Phase 2: Solvers (Using Elf) ---

def solve_multicut(graph, costs):
    print("Solver: Multicut (Kernighan-Lin)")
    # Using elf wrapper
    return multicut_kernighan_lin(graph, costs)

def solve_gasp(graph, costs):
    print("Solver: GASP (GAEC)")
    # Using elf wrapper
    return multicut_gaec(graph, costs)

def solve_mutex(graph, costs, dense_edges):
    print("Solver: Mutex (Lifted GAEC)")
    lifted_edges = dense_edges.astype(np.uint64)
    # Using elf wrapper
    return lifted_multicut_gaec(graph, costs, lifted_edges, costs)

# --- Phase 3: Relabeling ---

global_nodes = None
global_labels = None

def relabel_init(nodes_path, labels_path):
    global global_nodes, global_labels
    global_nodes = np.load(nodes_path, mmap_mode='r')   
    global_labels = np.load(labels_path, mmap_mode='r') 

def relabel_worker_wrapper(args):
    return relabel_worker(*args)

def relabel_worker(task, seg_path, out_path, shape, chunk_size, resolution, offset_zyx):
    try:
        cid, z, y, x, ze, ye, xe = task
        off_z, off_y, off_x = offset_zyx
        gz, gy, gx = z + off_z, y + off_y, x + off_x
        gze, gye, gxe = ze + off_z, ye + off_y, xe + off_x
        
        spec_in = get_input_spec(seg_path)
        ds_in = ts.open(spec_in, open=True).result()
        
        offset_xyz = [off_x, off_y, off_z]
        spec_out = get_output_sharded_spec(out_path, shape, chunk_size, resolution, offset_xyz)
        ds_out = ts.open(spec_out, open=True, create=False).result()

        arr = ds_in[gx:gxe, gy:gye, gz:gze, 0:1].read().result()
        arr = arr.squeeze().transpose(2, 1, 0)

        idx = np.searchsorted(global_nodes, arr)
        idx_clipped = np.clip(idx, 0, len(global_nodes) - 1)
        mask = (global_nodes[idx_clipped] == arr)
        
        out_arr = arr.copy() 
        out_arr[mask] = global_labels[idx_clipped[mask]]

        out_arr = out_arr.transpose(2, 1, 0)[..., np.newaxis]
        ds_out[gx:gxe, gy:gye, gz:gze, 0:1].write(out_arr).result()

        return (cid, 'success')
    except Exception as e:
        return (cid, 'failed', str(e))

# --- Main Logic ---
def run_agglomeration(args):
    conn = init_db(DB_NAME)
    os.makedirs(TEMP_GRAPH_DIR, exist_ok=True)
    
    if args.offset:
        offset_zyx = tuple(map(int, args.offset.split(',')))
    else:
        print("WARNING: No offset provided. Using 0,0,0.")
        offset_zyx = (0, 0, 0)

    z_aff = zarr.open(args.affinities, mode='r')
    ds_aff = z_aff[args.aff_key]
    full_shape = ds_aff.shape
    if len(full_shape) == 4 and full_shape[0] == 3: spatial_shape = full_shape[1:]
    elif len(full_shape) == 4: spatial_shape = full_shape[:-1]
    else: spatial_shape = full_shape

    populate_db(conn, spatial_shape, CHUNK_SIZE, 'graph_tasks')
    populate_db(conn, spatial_shape, CHUNK_SIZE, 'write_tasks')

    # --- Phase 1: Extract ---
    print("--- Phase 1: Edge Extraction ---")
    c = conn.cursor()
    c.execute("SELECT id, z, y, x, z_e, y_e, x_e FROM graph_tasks WHERE status='pending'")
    tasks = c.fetchall()

    if tasks:
        with mp.Pool(args.workers) as pool:
            worker_args = [(t, args.affinities, args.aff_key, args.supervoxels, TEMP_GRAPH_DIR, offset_zyx, args.invert) for t in tasks]
            results = pool.imap_unordered(extract_edges_wrapper, worker_args)
            for res in tqdm(results, total=len(tasks), desc="Extracting"):
                cid, status, val = res
                if status == 'success':
                    conn.execute("UPDATE graph_tasks SET status='done', pickle_path=? WHERE id=?", (val, cid))
                elif status == 'skipped':
                     conn.execute("UPDATE graph_tasks SET status='done' WHERE id=?", (cid,))
                conn.commit()
    else:
        print("Phase 1 complete.")

    # --- Phase 2: Solve ---
    print(f"--- Phase 2: Global Agglomeration ({args.method}) ---")
    c.execute("SELECT pickle_path FROM graph_tasks WHERE status='done' AND pickle_path IS NOT NULL")
    files = c.fetchall()
    
    global_edges = []
    global_weights = []
    global_counts = []
    
    print("Loading graph parts...")
    for (fpath,) in tqdm(files):
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
            global_edges.append(data['edges'])
            global_weights.append(data['sum'])
            global_counts.append(data['count'])
            
    if not global_edges:
        print("No edges found. Exiting.")
        return

    all_edges = np.concatenate(global_edges)
    all_sum = np.concatenate(global_weights)
    all_count = np.concatenate(global_counts)
    
    print("Deduplicating edges...")
    unique_sparse_edges, inverse = np.unique(all_edges, axis=0, return_inverse=True)
    final_weights = np.bincount(inverse, weights=all_sum) / np.bincount(inverse, weights=all_count)
    
    print("Remapping IDs...")
    unique_nodes = np.unique(unique_sparse_edges)
    num_nodes = len(unique_nodes)
    u_mapped = np.searchsorted(unique_nodes, unique_sparse_edges[:, 0])
    v_mapped = np.searchsorted(unique_nodes, unique_sparse_edges[:, 1])
    dense_edges = np.stack([u_mapped, v_mapped], axis=1)

    print(f"Graph: {num_nodes} nodes, {len(dense_edges)} edges.")
    
    # Cost Logic: 
    # Assumes input affinities are "Boundary Probabilities" (0=Inside, 1=Boundary)
    # If using --invert, the worker converts 1=Inside to 0=Inside.
    # So `final_weights` are 0 for inside (merge) and 1 for boundary (split).
    # Merge Prob = 1.0 - weight.
    # Inside (0) -> Prob 1.0 -> Cost +0.5. Merge.
    merge_probs = 1.0 - final_weights
    costs = merge_probs - args.beta
    
    graph = ngraph.UndirectedGraph(num_nodes)
    graph.insertEdges(dense_edges.astype(np.uint64))
    
    if args.method == 'multicut':
        dense_labels = solve_multicut(graph, costs)
    elif args.method == 'gasp':
        dense_labels = solve_gasp(graph, costs)
    elif args.method == 'mutex':
        dense_labels = solve_mutex(graph, costs, dense_edges)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    nodes_file = "unique_nodes.npy"
    labels_file = "dense_labels.npy"
    np.save(nodes_file, unique_nodes)   
    np.save(labels_file, dense_labels)  
    
    # --- Phase 3: Write ---
    print("--- Phase 3: Writing Output ---")
    res_list = [int(x) for x in args.resolution.split(',')]
    offset_xyz = [offset_zyx[2], offset_zyx[1], offset_zyx[0]]
    spec_out = get_output_sharded_spec(args.output, spatial_shape, CHUNK_SIZE, res_list, offset_xyz)
    ts.open(spec_out, create=True, open=True).result()

    c.execute("SELECT id, z, y, x, z_e, y_e, x_e FROM write_tasks WHERE status='pending'")
    tasks = c.fetchall()
    
    if tasks:
        with mp.Pool(args.workers, initializer=relabel_init, initargs=(nodes_file, labels_file)) as pool:
            worker_args = [(t, args.supervoxels, args.output, spatial_shape, CHUNK_SIZE, res_list, offset_zyx) for t in tasks]
            results = pool.imap_unordered(relabel_worker_wrapper, worker_args)
            
            for res in tqdm(results, total=len(tasks), desc="Writing"):
                cid, status = res
                if status == 'success':
                    conn.execute("UPDATE write_tasks SET status='done' WHERE id=?", (cid,))
                    conn.commit()
                else:
                    print(f"Error {cid}: {status}")

    conn.close()
    print("Agglomeration Complete.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--affinities", required=True)
    parser.add_argument("--aff_key", required=True)
    parser.add_argument("--supervoxels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resolution", default="8,8,8")
    parser.add_argument("--method", default="gasp", choices=["gasp", "multicut", "mutex"])
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--offset", default=None, help="Global offset (z,y,x)")
    parser.add_argument("--invert", action='store_true', help="Invert affinities during edge extraction.")
    
    args = parser.parse_args()
    
    run_agglomeration(args)
