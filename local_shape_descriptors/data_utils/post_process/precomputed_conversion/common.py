import os
import re
import sqlite3
import numpy as np
from typing import Tuple, List

# ================= USER CONFIGURATION =================

# TOGGLE MODE HERE: 'EM' or 'SEGMENTATION'
MODE = 'EM' 

# INPUTS
#INPUT_FILENAME = "parker_s0_z200-6110_y0-23392_x0-24672.zarr"
INPUT_FILENAME = "parker_s0_z1600-2506_y7384-9077_x5934-7548_f1_clahe.zarr"
# Zarr path: Adjust 'volumes/raw' or 'volumes/final_segmentation' based on MODE
#INPUT_ZARR_PATH = f"/net/fibserver1/raw/smohinta_data/catena_data/PARKER/data_3d/test/{INPUT_FILENAME}/volumes/raw"
INPUT_ZARR_PATH = f"/net/fibserver1/raw/smohinta_data/catena_data/PARKER_F1/data_3d/test/{INPUT_FILENAME}/volumes/raw"

# MASK
#MASK_PATH = "/mnt/graid/mito-seg/catena_data/Parker/data_3d/test/s5_ParkerBrainMask_3.tif"
#MASK_SCALE_FACTOR = 32  
MASK_PATH = "/mnt/graid/mito-seg/catena_data/Parker/data_3d/test/parker_s0_z1600-2506_y7384-9077_x5934-7548_f1_clahe_mask.tif"
MASK_SCALE_FACTOR = 1

# OUTPUT
OUTPUT_PATH = f"/mnt/graid/neuroglancer_data/parker_f1/output_{MODE.lower()}_sharded"
DB_PATH = f"/mnt/graid/neuroglancer_data/parker_f1//state_{MODE.lower()}.db"

# CHUNKING
INPUT_CHUNK_SIZE = (256, 256, 256) # (z, y, x)
OUTPUT_INNER_CHUNK = [64, 64, 64]  # (x, y, z) inside the shard

# RESOLUTION (nm)
RESOLUTION = [8.0, 8.0, 8.0] 

# ================= UTILITIES =================

def parse_bounds_from_filename(filename: str) -> Tuple[int, int, int]:
    """Extracts (z_start, y_start, x_start) from filename."""
    z_match = re.search(r'z(\d+)-', filename)
    y_match = re.search(r'y(\d+)-', filename)
    x_match = re.search(r'x(\d+)-', filename)
    
    if not (z_match and y_match and x_match):
        raise ValueError(f"Could not parse coordinates from filename: {filename}")
        
    return (int(z_match.group(1)), int(y_match.group(1)), int(x_match.group(1)))

class TaskManager:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                z INT, y INT, x INT,
                status TEXT DEFAULT 'pending', 
                PRIMARY KEY (z, y, x)
            )
        ''')
        self.conn.commit()

    def add_tasks(self, coords_list: List[Tuple[int, int, int]]):
        if not coords_list: return
        self.cursor.executemany(
            "INSERT OR IGNORE INTO chunks (z, y, x, status) VALUES (?, ?, ?, 'pending')", 
            coords_list
        )
        self.conn.commit()

    def get_pending_batch(self, limit=1):
        """Fetches a batch of pending tasks."""
        self.cursor.execute("SELECT z, y, x FROM chunks WHERE status='pending' LIMIT ?", (limit,))
        return self.cursor.fetchall()

    def update_status(self, coords, status):
        self.cursor.execute(
            "UPDATE chunks SET status=? WHERE z=? AND y=? AND x=?", 
            (status, coords[0], coords[1], coords[2])
        )
        self.conn.commit()

    def count_pending(self):
        self.cursor.execute("SELECT COUNT(*) FROM chunks WHERE status='pending'")
        return self.cursor.fetchone()[0]

    def get_stats(self):
        self.cursor.execute("SELECT status, COUNT(*) FROM chunks GROUP BY status")
        return dict(self.cursor.fetchall())
