"""
Author: Samia Mohinta, Gemini 3 Pro
Affiliation: University of Cambridge, UK
"""

import sqlite3
import pandas as pd
import numpy as np
import multiprocessing
import os
import sys
import tifffile
from cloudvolume import CloudVolume
from tqdm import tqdm
from collections import defaultdict

# ================= CONFIGURATION =================
SOURCE_DB_PATH = "/mnt/graid/synapse_detection/predictions/octo_cns/octo_cns_setup_03_octo_cube_all3_same_preid_256_300000/synapse_predictions.db"
OUTPUT_DB_PATH = "/mnt/graid/synapse_detection/predictions/octo_cns/octo_cns_setup_03_octo_cube_all3_same_preid_256_300000_filt/final_upload_ready.db"

# SEGMENTATION & MASK
SEG_VOL_PATH = 'file:///mnt/graid/biomedparse/octo/seg_241224_250131b_rsg8_spl'
NEUROPIL_TIF_PATH = '/mnt/graid/synapse_neuron_associations/octo_s6_neurophil_corrected_dilated.tif' 

# RESOLUTION & SCALING
RESOLUTION_NM = np.array([8, 8, 8]) 
MASK_SCALE_FACTOR = 64
MIP_LEVEL = 0
FORCE_ZYX_TO_XYZ = True

# WORKER SETTINGS
NUM_WORKERS = 16 
BATCH_SIZE = 100 
# ===============================================

mask_data_global = None

def pre_flight_checks():
    print("\n--- RUNNING PRE-FLIGHT CHECKS ---")
    if not os.path.exists(SOURCE_DB_PATH):
        print(f"❌ ERROR: Source DB not found at: {SOURCE_DB_PATH}")
        return False
    if not os.path.exists(NEUROPIL_TIF_PATH):
        print(f"❌ ERROR: Neuropil TIFF not found at: {NEUROPIL_TIF_PATH}")
        return False
    return True

def init_output_db():
    os.makedirs(os.path.dirname(OUTPUT_DB_PATH), exist_ok=True)
    if os.path.exists(OUTPUT_DB_PATH):
        try: os.remove(OUTPUT_DB_PATH) 
        except OSError: pass
        
    conn = sqlite3.connect(OUTPUT_DB_PATH)
    c = conn.cursor()
    
    # --- CREATING TABLES COMPATIBLE WITH UPLOAD SCRIPT ---
    
    # 1. pre_sites (Must have 'id' and 'score')
    c.execute('''
        CREATE TABLE pre_sites (
            id INTEGER PRIMARY KEY,
            x INTEGER, y INTEGER, z INTEGER,
            score REAL, 
            segment_id INTEGER
        )
    ''')
    
    # 2. post_sites (Must have 'id')
    c.execute('''
        CREATE TABLE post_sites (
            id INTEGER PRIMARY KEY,
            x INTEGER, y INTEGER, z INTEGER,
            segment_id INTEGER
        )
    ''')
    
    # 3. pre_post_mapping (Must have 'pre_id', 'post_id')
    c.execute('''
        CREATE TABLE pre_post_mapping (
            pre_id INTEGER,
            post_id INTEGER,
            pre_seg_id INTEGER,
            post_seg_id INTEGER
        )
    ''')
    
    c.execute('CREATE INDEX idx_pre_seg ON pre_sites (segment_id)')
    c.execute('CREATE INDEX idx_post_seg ON post_sites (segment_id)')
    c.execute('CREATE INDEX idx_map_pre ON pre_post_mapping (pre_id)')
    conn.commit()
    conn.close()

def load_mask_global():
    global mask_data_global
    print(f"⏳ Loading TIFF mask from {NEUROPIL_TIF_PATH}...")
    try:
        img = tifffile.imread(NEUROPIL_TIF_PATH)
        if FORCE_ZYX_TO_XYZ:
            print("   🔄 Forcing Transpose: (Z, Y, X) -> (X, Y, Z)")
            img = img.transpose(2, 1, 0)
        mask_data_global = img
        return True
    except Exception as e:
        print(f"❌ ERROR Loading TIFF: {e}")
        return False

def worker_process_pre_sites(args):
    """
    Filters by Neuropil Mask AND fetches Score.
    """
    vol_path, mip, df_batch = args
    results = []
    
    try:
        vol = CloudVolume(vol_path, mip=mip, fill_missing=True, progress=False)
        chunk_size = vol.chunk_size
        mask = mask_data_global
        mask_shape = mask.shape
        
        res_nm = RESOLUTION_NM
        mask_scale = res_nm * MASK_SCALE_FACTOR
        
        ids = df_batch['id'].values
        xs_nm = df_batch['x'].values
        ys_nm = df_batch['y'].values
        zs_nm = df_batch['z'].values
        scores = df_batch['score'].values # <--- NEW: Capture Score
        
        # Coords for Segmentation
        xs_vox = (xs_nm // res_nm[0]).astype(np.int32)
        ys_vox = (ys_nm // res_nm[1]).astype(np.int32)
        zs_vox = (zs_nm // res_nm[2]).astype(np.int32)
        
        # Coords for Mask
        xs_mask = (xs_nm // mask_scale[0]).astype(np.int32)
        ys_mask = (ys_nm // mask_scale[1]).astype(np.int32)
        zs_mask = (zs_nm // mask_scale[2]).astype(np.int32)
        
        # Grouping
        cx = (xs_vox // chunk_size[0]) * chunk_size[0]
        cy = (ys_vox // chunk_size[1]) * chunk_size[1]
        cz = (zs_vox // chunk_size[2]) * chunk_size[2]
        
        chunk_map = defaultdict(list)
        for i in range(len(ids)):
            chunk_map[(cx[i], cy[i], cz[i])].append(i)
            
        for (c_x, c_y, c_z), indices in chunk_map.items():
            bbox = np.s_[c_x:c_x+chunk_size[0], c_y:c_y+chunk_size[1], c_z:c_z+chunk_size[2]]
            try:
                chunk_data = vol[bbox].squeeze()
            except Exception:
                continue

            for idx in indices:
                # 1. Check Neuropil Mask
                mx, my, mz = xs_mask[idx], ys_mask[idx], zs_mask[idx]
                if not (0 <= mx < mask_shape[0] and 0 <= my < mask_shape[1] and 0 <= mz < mask_shape[2]): continue
                if mask[mx, my, mz] != 255: continue
                
                # 2. Check Segmentation
                lx, ly, lz = int(xs_vox[idx]-c_x), int(ys_vox[idx]-c_y), int(zs_vox[idx]-c_z)
                if (0 <= lx < chunk_data.shape[0] and 0 <= ly < chunk_data.shape[1] and 0 <= lz < chunk_data.shape[2]):
                    seg_id = chunk_data[lx, ly, lz]
                    
                    if seg_id > 0:
                        results.append((
                            int(ids[idx]), 
                            int(xs_nm[idx]), int(ys_nm[idx]), int(zs_nm[idx]), 
                            float(scores[idx]), # <--- NEW: Save Score
                            int(seg_id)
                        ))
        return results
    except Exception:
        return []

def worker_process_post_sites(args):
    # Same as before, just mapping Post Sites
    vol_path, mip, df_batch = args
    results = []
    
    try:
        vol = CloudVolume(vol_path, mip=mip, fill_missing=True, progress=False)
        chunk_size = vol.chunk_size
        res_nm = RESOLUTION_NM
        
        ids = df_batch['id'].values
        xs_nm = df_batch['x'].values
        ys_nm = df_batch['y'].values
        zs_nm = df_batch['z'].values
        
        xs_vox = (xs_nm // res_nm[0]).astype(np.int32)
        ys_vox = (ys_nm // res_nm[1]).astype(np.int32)
        zs_vox = (zs_nm // res_nm[2]).astype(np.int32)
        
        cx = (xs_vox // chunk_size[0]) * chunk_size[0]
        cy = (ys_vox // chunk_size[1]) * chunk_size[1]
        cz = (zs_vox // chunk_size[2]) * chunk_size[2]
        
        chunk_map = defaultdict(list)
        for i in range(len(ids)):
            chunk_map[(cx[i], cy[i], cz[i])].append(i)
            
        for (c_x, c_y, c_z), indices in chunk_map.items():
            bbox = np.s_[c_x:c_x+chunk_size[0], c_y:c_y+chunk_size[1], c_z:c_z+chunk_size[2]]
            try:
                chunk_data = vol[bbox].squeeze()
            except Exception:
                continue
                
            for idx in indices:
                lx, ly, lz = int(xs_vox[idx]-c_x), int(ys_vox[idx]-c_y), int(zs_vox[idx]-c_z)
                if (0 <= lx < chunk_data.shape[0] and 0 <= ly < chunk_data.shape[1] and 0 <= lz < chunk_data.shape[2]):
                    seg_id = chunk_data[lx, ly, lz]
                    if seg_id > 0:
                        results.append((
                            int(ids[idx]), 
                            int(xs_nm[idx]), int(ys_nm[idx]), int(zs_nm[idx]), 
                            int(seg_id)
                        ))
        return results
    except Exception:
        return []

def db_writer_listener(queue):
    conn = sqlite3.connect(OUTPUT_DB_PATH)
    conn.execute('PRAGMA journal_mode=WAL;') 
    conn.execute('PRAGMA synchronous=NORMAL;')
    cursor = conn.cursor()
    while True:
        record = queue.get()
        if record == 'KILL': break
        table, data = record
        try:
            # Dynamic placeholders based on column count
            placeholders = ','.join(['?'] * len(data[0]))
            cursor.executemany(f'INSERT OR REPLACE INTO {table} VALUES ({placeholders})', data)
            conn.commit()
        except Exception as e: print(f"DB Error: {e}")
    conn.close()

def main():
    if not pre_flight_checks(): sys.exit(1)
    if not load_mask_global(): sys.exit(1)
    init_output_db()
    
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    writer = multiprocessing.Process(target=db_writer_listener, args=(queue,))
    writer.start()
    
    # --- STEP 1: PRE SITES (NOW WITH SCORE) ---
    print("\n--- STEP 1: Processing Pre-Sites (Neuropil Filter + Score) ---")
    conn = sqlite3.connect(SOURCE_DB_PATH)
    # Fetch Score here
    df_pre = pd.read_sql_query("SELECT id, x, y, z, score FROM pre_sites", conn)
    conn.close()
    
    df_pre.sort_values(by=['z', 'y', 'x'], inplace=True)
    batches = [df_pre.iloc[i:i+BATCH_SIZE] for i in range(0, len(df_pre), BATCH_SIZE)]
    tasks = [(SEG_VOL_PATH, MIP_LEVEL, b) for b in batches]
    
    from multiprocessing.pool import ThreadPool
    valid_pre_ids = set()
    
    with ThreadPool(NUM_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(worker_process_pre_sites, tasks), total=len(tasks)):
            if result:
                for r in result: valid_pre_ids.add(r[0])
                queue.put(('pre_sites', result)) # Saving to 'pre_sites' table
            
    print(f"Valid Pre-sites: {len(valid_pre_ids):,}")

    # --- STEP 2: POST SITES ---
    print("\n--- STEP 2: Processing Post-Sites ---")
    conn = sqlite3.connect(SOURCE_DB_PATH)
    df_map = pd.read_sql_query("SELECT pre_id, post_id FROM pre_post_mapping", conn)
    conn.close()
    
    # Filter mapping by valid pre-sites
    df_map_valid = df_map[df_map['pre_id'].isin(valid_pre_ids)]
    relevant_post_ids = set(df_map_valid['post_id'].unique())
    
    conn = sqlite3.connect(SOURCE_DB_PATH)
    df_post = pd.read_sql_query("SELECT id, x, y, z FROM post_sites", conn)
    conn.close()
    
    df_post = df_post[df_post['id'].isin(relevant_post_ids)]
    df_post.sort_values(by=['z', 'y', 'x'], inplace=True)
    
    batches = [df_post.iloc[i:i+BATCH_SIZE] for i in range(0, len(df_post), BATCH_SIZE)]
    tasks = [(SEG_VOL_PATH, MIP_LEVEL, b) for b in batches]
    
    with ThreadPool(NUM_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(worker_process_post_sites, tasks), total=len(tasks)):
            if result: queue.put(('post_sites', result)) # Saving to 'post_sites' table

    queue.put('KILL')
    writer.join()

    # --- STEP 3: FINAL JOIN (AUTAPSES) ---
    print("\n--- STEP 3: Finalizing & Removing Autapses ---")
    conn_out = sqlite3.connect(OUTPUT_DB_PATH)
    cursor = conn_out.cursor()
    
    print("Uploading valid pairs to temporary table...")
    df_map_valid.to_sql('temp_pairs', conn_out, if_exists='replace', index=False)
    
    print("Creating 'pre_post_mapping' table (Removing Autapses)...")
    # Join with standard table names 'pre_sites' and 'post_sites'
    cursor.execute('''
        INSERT INTO pre_post_mapping (pre_id, post_id, pre_seg_id, post_seg_id)
        SELECT 
            t.pre_id, 
            t.post_id,
            pre.segment_id,
            post.segment_id
        FROM temp_pairs t
        INNER JOIN pre_sites pre ON t.pre_id = pre.id
        INNER JOIN post_sites post ON t.post_id = post.id
        WHERE pre.segment_id != post.segment_id
    ''')
    
    cursor.execute("DROP TABLE temp_pairs")
    conn_out.commit()
    conn_out.close()
    
    print(f"\n SUCCESS. Database ready for upload at: {OUTPUT_DB_PATH}")

if __name__ == "__main__":
    main()
