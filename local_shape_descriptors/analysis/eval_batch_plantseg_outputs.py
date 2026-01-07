## beast: conda activate funkelsd
import os
import sqlite3
import pandas as pd
import numpy as np
import skimage.metrics
import tifffile
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

def setup_db(db_path):
    """Initializes the SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            folder_name TEXT PRIMARY KEY,
            rand_error REAL,
            precision REAL,
            recall REAL,
            false_splits REAL,
            false_merges REAL
        )
    ''')
    conn.commit()
    conn.close()

def get_processed_folders(db_path):
    """Returns a set of folder names already processed."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT folder_name FROM metrics')
    rows = cursor.fetchall()
    conn.close()
    return {row[0] for row in rows}

def process_single_folder(folder_path, gt_data):
    """
    Worker function: Performs the heavy computation.
    Returns results to the main process for DB writing.
    """
    folder_name = os.path.basename(folder_path)
    post_proc_path = os.path.join(folder_path, "PostProcessing")
    
    result = {
        'folder_name': folder_name,
        'rand_error': None,
        'precision': None,
        'recall': None,
        'false_splits': None,
        'false_merges': None
    }

    try:
        if not os.path.exists(post_proc_path):
            return result
            
        tiff_files = [f for f in os.listdir(post_proc_path) if f.endswith(('.tiff', '.tif'))]
        if not tiff_files:
            return result

        pred_path = os.path.join(post_proc_path, tiff_files[0])
        seg = tifffile.imread(pred_path)

        # Standard skimage metrics
        splits, merges = skimage.metrics.variation_of_information(gt_data, seg, ignore_labels=(0,))
        error, precision, recall = skimage.metrics.adapted_rand_error(gt_data, seg, ignore_labels=(0,))

        result.update({
            'rand_error': round(float(error), 6),
            'precision': round(float(precision), 6),
            'recall': round(float(recall), 6),
            'false_splits': round(float(splits), 6),
            'false_merges': round(float(merges), 6)
        })

    except Exception as e:
        # We don't print inside the parallel loop to keep the progress bar clean
        pass
    
    return result

def save_results_to_db(db_path, results):
    """Sequential write to database from the main process."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for res in results:
        cursor.execute('''
            INSERT OR REPLACE INTO metrics 
            (folder_name, rand_error, precision, recall, false_splits, false_merges)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            res['folder_name'], res['rand_error'], res['precision'], 
            res['recall'], res['false_splits'], res['false_merges']
        ))
    conn.commit()
    conn.close()

def main(root_dir, gt_path, db_name="eval_results.db", n_jobs=-1):
    db_path = os.path.join(root_dir, db_name)
    setup_db(db_path)
    
    print("--- Loading Ground Truth ---")
    gt_data = tifffile.imread(gt_path)
    
    # Identify folders containing 'beta'
    all_folders = [
        os.path.join(root_dir, d) for d in os.listdir(root_dir) 
        if os.path.isdir(os.path.join(root_dir, d)) and "beta" in d
    ]
    
    processed = get_processed_folders(db_path)
    folders_to_run = [f for f in all_folders if os.path.basename(f) not in processed]
    
    if not folders_to_run:
        print("All folders already processed or no valid folders found.")
    else:
        print(f"--- Processing {len(folders_to_run)} Configurations ---")
        
        # Parallel execution with Progress Bar
        # We use a generator to update tqdm as tasks complete
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_folder)(f, gt_data) 
            for f in tqdm(folders_to_run, desc="Evaluating Segments", unit="folder")
        )
        
        print("\n--- Finalizing Database ---")
        save_results_to_db(db_path, results)

    # Always generate/update CSV from current DB state
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM metrics", conn)
    conn.close()
    
    csv_path = os.path.join(root_dir, "evaluation_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Success! Metrics exported to: {csv_path}")

if __name__ == "__main__":
    # USER CONFIG
    ROOT_FOLDER = "/media/samia/DATA/plant-seg/segmentation-runs/data-20-200-200-3Drun"
    GT_FILE = "/media/samia/DATA/plant-seg/segmentation-runs/data_gt/ottoLabels_manualCuration_22-Nov-2022.tif"
    
    main(ROOT_FOLDER, GT_FILE)
