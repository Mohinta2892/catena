import numpy as np
import zarr
import tifffile
from tqdm import tqdm
from common import (
    TaskManager, parse_bounds_from_filename, 
    INPUT_FILENAME, INPUT_ZARR_PATH, MASK_PATH, 
    MASK_SCALE_FACTOR, INPUT_CHUNK_SIZE, DB_PATH
)

def scout_volume():
    # 1. Setup
    db = TaskManager(DB_PATH)
    offsets = parse_bounds_from_filename(INPUT_FILENAME)
    off_z, off_y, off_x = offsets
    
    print(f"Opening Input Zarr: {INPUT_ZARR_PATH}")
    try:
        source_zarr = zarr.open(INPUT_ZARR_PATH, mode='r')
        gz, gy, gx = source_zarr.shape
    except Exception as e:
        print(f"Error opening Zarr: {e}")
        return

    print(f"Opening Mask: {MASK_PATH}")
    mask_arr = tifffile.imread(MASK_PATH)
    
    cz, cy, cx = INPUT_CHUNK_SIZE
    
    # Calculate grid size
    nz = int(np.ceil(gz / cz))
    ny = int(np.ceil(gy / cy))
    nx = int(np.ceil(gx / cx))
    
    total_chunks = nz * ny * nx
    print(f"Scouting Volume: {gz}x{gy}x{gx}")
    print(f"Global Offsets: {offsets}")
    print(f"Total potential chunks: {total_chunks}")
    
    valid_tasks = []
    
    # 2. Iterate
    # We use a flat loop or nested loop with tqdm
    # Nested allows us to batch DB writes per Z-slice if we want
    
    for z_idx in tqdm(range(nz), desc="Scouting Z-slices"):
        for y_idx in range(ny):
            for x_idx in range(nx):
                # Local coords
                z_local = z_idx * cz
                y_local = y_idx * cy
                x_local = x_idx * cx
                
                # Global coords
                z_global = z_local + off_z
                y_global = y_local + off_y
                x_global = x_local + off_x

                # Mask coords (Scale 5 -> divide by 32)
                mz_start = z_global // MASK_SCALE_FACTOR
                my_start = y_global // MASK_SCALE_FACTOR
                mx_start = x_global // MASK_SCALE_FACTOR
                
                mz_end = (z_global + cz) // MASK_SCALE_FACTOR
                my_end = (y_global + cy) // MASK_SCALE_FACTOR
                mx_end = (x_global + cx) // MASK_SCALE_FACTOR
                
                # Bounds check against mask array
                mz_start = max(0, min(mz_start, mask_arr.shape[0]-1))
                my_start = max(0, min(my_start, mask_arr.shape[1]-1))
                mx_start = max(0, min(mx_start, mask_arr.shape[2]-1))
                
                mz_end = max(mz_start + 1, min(mz_end, mask_arr.shape[0]))
                my_end = max(my_start + 1, min(my_end, mask_arr.shape[1]))
                mx_end = max(mx_start + 1, min(mx_end, mask_arr.shape[2]))

                # Check Mask
                mask_slice = mask_arr[mz_start:mz_end, my_start:my_end, mx_start:mx_end]
                
                if np.any(mask_slice):
                    valid_tasks.append((z_local, y_local, x_local))
        
        # Batch insert to keep DB transaction size reasonable
        if valid_tasks:
            db.add_tasks(valid_tasks)
            valid_tasks = []

    print("Scouting Complete.")
    print("DB Stats:", db.get_stats())

if __name__ == "__main__":
    scout_volume()
