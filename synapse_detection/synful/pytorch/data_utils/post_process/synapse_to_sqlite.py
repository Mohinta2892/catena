"""
GPU2:
conda activate syn_save
"""
import sqlite3
import os
import itertools
import logging
import zarr
import numpy as np
import concurrent.futures
from tqdm import tqdm
import sys

# ================= CONFIGURATION =================
# Double check this path matches your network mount exactly
SYNAPSE_DIR = '/net/zstore1/smohinta/synful/scripts/predict/output_predict_on_train/octo_cns_setup_03_octo_cube_all3_same_preid_256_300000/syn_cc_thr095000_sum'
EM_ZARR_PATH = "/net/fibserver1/raw/smohinta_data/catena_data/OCTO/data_3d/test/octo_cns_s0_z0-34745_y0-21471_x0-20486.zarr"
EM_DATASET = "volumes/raw"
OUTPUT_DB_NAME = '/mnt/graid/synapse_detection/octo_cns_setup_03_octo_cube_all3_same_preid_256_300000/synapse_predictions.db'

# Worker settings
NUM_WORKERS = 16  # Crank this up since workers are now very lightweight
FILES_PER_TASK = 50 # Number of .npz files to process in one task (Batching)
VOXEL_SIZE = (8, 8, 8)
SCORE_THRESHOLD = 0
# =================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def is_in_roi(point, roi_offset, roi_shape):
    """Pure python check if point (z,y,x) is in roi."""
    z, y, x = point
    oz, oy, ox = roi_offset
    sz, sy, sx = roi_shape
    return (oz <= z < oz + sz) and (oy <= y < oy + sy) and (ox <= x < ox + sx)

def process_file_batch(file_paths, roi_offset, roi_shape, score_thr):
    """
    Worker Function: Processes a specific list of .npz files.
    """
    results = []

    for f_path in file_paths:
        try:
            # Fast fail if file vanished
            if not os.path.exists(f_path):
                continue

            data = np.load(f_path)
            # Accessing arrays is fast, no computation yet
            ids = data['ids']

            if len(ids) == 0:
                continue

            locations = data['positions']
            scores = data['scores']

            for ii, id_val in enumerate(ids):
                if scores[ii] <= score_thr:
                    continue

                loc_pre = locations[ii, 0]
                loc_post = locations[ii, 1]

                # Check ROI
                if is_in_roi(loc_post, roi_offset, roi_shape):
                    results.append((
                        int(id_val),
                        float(scores[ii]),
                        float(loc_pre[0]), float(loc_pre[1]), float(loc_pre[2]),
                        float(loc_post[0]), float(loc_post[1]), float(loc_post[2])
                    ))
        except Exception:
            # If one file is corrupt, skip it, don't crash the batch
            continue

    return results

def create_tables(conn):
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS pre_sites (id INTEGER PRIMARY KEY, z REAL, y REAL, x REAL, score REAL)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_pre_loc ON pre_sites (z, y, x)')
    c.execute('CREATE TABLE IF NOT EXISTS post_sites (id INTEGER PRIMARY KEY, z REAL, y REAL, x REAL)')
    c.execute('CREATE TABLE IF NOT EXISTS pre_post_mapping (pre_id INTEGER, post_id INTEGER, FOREIGN KEY(pre_id) REFERENCES pre_sites(id), FOREIGN KEY(post_id) REFERENCES post_sites(id))')
    conn.commit()

def get_existing_pre_locations(conn):
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pre_sites'")
    if not c.fetchone():
        return set()
    logging.info("Loading existing locations for deduplication...")
    c.execute("SELECT z, y, x FROM pre_sites")
    return set((r[0], r[1], r[2]) for r in c.fetchall())

def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def main():
    # 1. Setup DB
    os.makedirs(os.path.dirname(OUTPUT_DB_NAME), exist_ok=True)
    conn = sqlite3.connect(OUTPUT_DB_NAME)
    conn.execute('PRAGMA journal_mode=WAL;') # Crucial for speed
    conn.execute('PRAGMA synchronous = NORMAL;') # Safer speed vs full sync
    create_tables(conn)

    existing_locs = get_existing_pre_locations(conn)
    logging.info(f"Resume capability: {len(existing_locs)} synapses already in DB.")

    # 2. Get ROI
    logging.info(f"Reading ROI from {EM_ZARR_PATH}...")
    try:
        f = zarr.open(EM_ZARR_PATH, mode='r')
        shape = f[EM_DATASET].shape
        roi_shape_tuple = (shape[0]*VOXEL_SIZE[0], shape[1]*VOXEL_SIZE[1], shape[2]*VOXEL_SIZE[2])
        roi_offset_tuple = (0, 0, 0)
    except Exception as e:
        logging.error(f"Could not read Zarr: {e}")
        return

    # 3. Setup ID Generators
    c = conn.cursor()
    c.execute("SELECT MAX(id) FROM pre_sites")
    max_pre = c.fetchone()[0] or 0
    c.execute("SELECT MAX(id) FROM post_sites")
    max_post = c.fetchone()[0] or 0
    start_id = max(max_pre, max_post) + 1
    ngid = itertools.count(start=start_id)

    # 4. STREAMING EXECUTION
    # Instead of finding all 1 million files first (slow), we find Z-slices,
    # then finding Y-slices, then files, and yield them instantly to workers.

    logging.info(f"Scanning {SYNAPSE_DIR}...")

    if not os.path.exists(SYNAPSE_DIR):
        raise FileNotFoundError(f"Path not found: {SYNAPSE_DIR}")

    # Generator that yields batches of file paths
    def file_batch_generator():
        # Get Z folders
        with os.scandir(SYNAPSE_DIR) as it:
            z_folders = [f.path for f in it if f.is_dir()]

        # Sort Z folders (optional, but nice)
        z_folders.sort(key=lambda p: int(os.path.basename(p)) if os.path.basename(p).isdigit() else p)

        current_batch = []

        for z_path in z_folders:
            # Get Y folders
            with os.scandir(z_path) as it_y:
                y_folders = [f.path for f in it_y if f.is_dir()]

            for y_path in y_folders:
                # Get NPZ files
                with os.scandir(y_path) as it_f:
                    # We get full paths immediately
                    files = [f.path for f in it_f if f.is_file() and f.name.endswith('.npz')]

                # Add to batch
                for f in files:
                    current_batch.append(f)
                    if len(current_batch) >= FILES_PER_TASK:
                        yield current_batch
                        current_batch = []

        # Yield remainder
        if current_batch:
            yield current_batch

    # 5. Process Pool
    total_added = 0
    total_skipped = 0

    logging.info(f"Starting pipeline with {NUM_WORKERS} workers processing batches of {FILES_PER_TASK} files...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:

        # We submit tasks as the generator yields them.
        # To avoid overfilling memory with pending tasks, we keep a set of running futures.
        futures = set()
        file_gen = file_batch_generator()

        # Progress bar (unknown total, so it just counts tasks)
        pbar = tqdm(desc="Processing File Batches", unit="batch")

        try:
            while True:
                # 1. Fill the pool with tasks until we hit a limit (e.g. 2x workers)
                # This keeps the pipeline full without loading all 1M files into RAM
                while len(futures) < NUM_WORKERS * 4:
                    try:
                        batch = next(file_gen)
                        f = executor.submit(
                            process_file_batch,
                            batch,
                            roi_offset_tuple,
                            roi_shape_tuple,
                            SCORE_THRESHOLD
                        )
                        futures.add(f)
                    except StopIteration:
                        break

                if not futures:
                    break

                # 2. Wait for at least one result
                done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

                # 3. Process Results
                for future in done:
                    pbar.update(1)
                    try:
                        batch_synapses = future.result()
                        if not batch_synapses:
                            continue

                        pre_buffer, post_buffer, map_buffer = [], [], []

                        for syn_data in batch_synapses:
                            _, score, pz, py, px, poz, poy, pox = syn_data

                            if (pz, py, px) in existing_locs:
                                total_skipped += 1
                                continue

                            pre_id = next(ngid)
                            post_id = next(ngid)

                            pre_buffer.append((pre_id, pz, py, px, score))
                            post_buffer.append((post_id, poz, poy, pox))
                            map_buffer.append((pre_id, post_id))
                            existing_locs.add((pz, py, px))

                        if pre_buffer:
                            c.executemany('INSERT INTO pre_sites (id, z, y, x, score) VALUES (?, ?, ?, ?, ?)', pre_buffer)
                            c.executemany('INSERT INTO post_sites (id, z, y, x) VALUES (?, ?, ?, ?)', post_buffer)
                            c.executemany('INSERT INTO pre_post_mapping (pre_id, post_id) VALUES (?, ?)', map_buffer)
                            conn.commit()
                            total_added += len(pre_buffer)
                            pbar.set_description(f"Batches Done (Synapses: {total_added})")

                    except Exception as e:
                        logging.error(f"Batch failed: {e}")

        except KeyboardInterrupt:
            logging.info("Stopping...")
            executor.shutdown(wait=False, cancel_futures=True)

    pbar.close()
    conn.close()
    logging.info(f"Done. Total added: {total_added}. Total skipped: {total_skipped}.")

if __name__ == "__main__":
    main()
