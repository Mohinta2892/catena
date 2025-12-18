import pandas as pd
import numpy as np
import os

def create_proofread_volume(csv_path, npy_path, output_path, id_column='ID'):
    """
    Loads a segmentation volume and filters it to keep ONLY the IDs listed in the CSV.
    All other IDs in the volume are set to 0.
    
    Args:
        csv_path (str): Path to the .csv file containing allowed instance IDs.
        npy_path (str): Path to the seg-modified.npy file.
        output_path (str): Path where the final filtered .npy file will be saved.
        id_column (str): The column name in the CSV containing the IDs.
    """
    
    # 1. Load the Valid IDs
    print(f"Loading CSV from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        if id_column not in df.columns:
            raise ValueError(f"Column '{id_column}' not found. Available: {list(df.columns)}")
        
        # Get unique IDs and ensure they match the numpy data type (usually integers)
        valid_ids = df[id_column].unique()
        print(f"Found {len(valid_ids)} unique valid IDs in CSV.")
        
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return

    # 2. Load the Segmentation Volume
    print(f"Loading Segmentation NPY from {npy_path}...")
    try:
        # We load the full array into memory to perform the fast vectorized operation
        segmentation_data = np.load(npy_path)
    except FileNotFoundError:
        print(f"Error: NPY file not found at {npy_path}")
        return

    # 3. Filter the Volume
    print("Filtering volume... (This may take a moment for large files)")
    
    # np.isin creates a boolean mask (True/False) of the same shape as segmentation_data.
    # True where the pixel value is in valid_ids, False otherwise.
    mask = np.isin(segmentation_data, valid_ids)
    
    # Apply the mask: Keep original data where mask is True, set to 0 where False.
    # We maintain the original data type (e.g., uint16, int32) to save space.
    filtered_volume = np.where(mask, segmentation_data, 0).astype(segmentation_data.dtype)

    # 4. Save the Result
    print(f"Saving filtered volume to {output_path}...")
    np.save(output_path, filtered_volume)
    
    print("Done!")

# --- Configuration ---
INPUT_CSV = '/media/samia/DATA/mounts/gpu2/neuron-seg/seg2link_proofreadneurons_v2.csv'
INPUT_NPY = '/media/samia/DATA/mounts/gpu2/neuron-seg/seg-modified.npy'
OUTPUT_FILE = '/media/samia/DATA/mounts/gpu2/neuron-seg/seg-proofread_30Jul24_v2.npy'
ID_COLUMN_NAME = 'ID' 

# --- Execution ---
if __name__ == "__main__":
    create_proofread_volume(INPUT_CSV, INPUT_NPY, OUTPUT_FILE, ID_COLUMN_NAME)
