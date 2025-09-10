# File: cropAndSaveSegmentation.py

import os
import numpy as np
import zarr
import tifffile
from caveclient import CAVEclient
from cloudvolume.lib import Bbox
from skimage.segmentation import relabel_sequential

def crop_and_save_segmentation(bbox_dims_xyz, output_dir):
    """
    Crops a segmentation volume from the Flywire datastack at a specific
    timestamp, reformats it, and saves it as Zarr and TIFF files.

    Args:
        bbox_dims_xyz (list): A list of 6 integers defining the bounding
                              box in XYZ format: [x_start, y_start, z_start, x_end, y_end, z_end].
        output_dir (str): The directory where the output files will be saved.
    """
    # ---------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------
    flywire_token = "x"
    datastack_name = 'zlatic_octo_8x8x8_full_200525_datastack'
    server_address = 'https://global.connectomics.braininbrain.org'
    # **Timestamp to query a specific state of the database**
    timestamp = 1750665448

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    base_filename = f"segmentation_crop_{bbox_dims_xyz[0]}-{bbox_dims_xyz[3]}_{bbox_dims_xyz[1]}-{bbox_dims_xyz[4]}_{bbox_dims_xyz[2]}-{bbox_dims_xyz[5]}"
    zarr_output_path = os.path.join(output_dir, f"{base_filename}.zarr")
    tiff_output_path = os.path.join(output_dir, f"{base_filename}.tiff")

    print("--- Script Configuration ---")
    print(f"Data Stack: {datastack_name}")
    print(f"Query Timestamp: {timestamp}") # Added for clarity
    print(f"Input BBox (XYZ): {bbox_dims_xyz}")
    print(f"Output Directory: {output_dir}")
    print(f"Zarr Output: {zarr_output_path}")
    print(f"TIFF Output: {tiff_output_path}")
    print("-" * 28 + "\n")

    # ---------------------------------------
    # 2. Connect to the data source
    # ---------------------------------------
    print("--> Connecting to CAVEclient...")
    client = CAVEclient(server_address=server_address, datastack_name=datastack_name, auth_token=flywire_token)

    # Get the full-resolution (mip=0) agglomerated segmentation source
    # **The timestamp is now correctly passed here to ensure version reproducibility**
    src_vol = client.info.segmentation_cloudvolume(
        agglomerate=True,
        mip=0,
        timestamp=timestamp
    )
    print("--> Connection successful.\n")

    # ---------------------------------------
    # 3. Download the data for the specified BBox
    # ---------------------------------------
    # Create a Bbox object from the user-provided dimensions
    print(f"Data shape {src_vol.shape}")
    x_start, y_start, z_start, x_end, y_end, z_end = bbox_dims_xyz
    crop_bbox = Bbox([x_start, y_start, z_start], [x_end, y_end, z_end])

    print(f"--> Downloading data for bounding box: {crop_bbox}...")
    # Directly download the volume slice corresponding to the bbox
    segmentation_data_xyzc = src_vol[crop_bbox]

    # The data has a channel dimension, which we can remove
    segmentation_data_xyz = np.squeeze(segmentation_data_xyzc)
    print(f"--> Download complete. Data shape (XYZ): {segmentation_data_xyz.shape}\n")

    # ---------------------------------------
    # 4. Transpose data from XYZ to ZYX format
    # ---------------------------------------
    print("--> Transposing data from XYZ to ZYX format...")
    # Original axes: 0=X, 1=Y, 2=Z
    # Target axes order: 2=Z, 1=Y, 0=X
    segmentation_data_zyx = np.transpose(segmentation_data_xyz, (2, 1, 0))
    print(f"--> Transposition complete. New shape (ZYX): {segmentation_data_zyx.shape}\n")
    
    # --- FIX: Flip the X-axis to correct the orientation ---
    # The X-axis is now the last axis (axis=2) in the ZYX numpy array.
    # We flip it to match the visual orientation seen in other viewers.
    print("--> Flipping horizontal axis (X) for correct orientation...")
    segmentation_data_zyx = np.flip(segmentation_data_zyx, axis=2)
    print("--> Flip complete.\n")

    # ---------------------------------------
    # 4. Relabel data for better visualization
    # ---------------------------------------
    print("--> Relabelling data...")
    #segmentation_data_zyx_relab, _, _= relabel_sequential(segmentation_data_zyx)
    segmentation_data_xyz_relab, _, _= relabel_sequential(segmentation_data_xyz)
    print(f"--> Relabelling complete.")
    
    # ---------------------------------------
    # 5. Save the cropped data
    # ---------------------------------------
    # Save as a Zarr array
    print(f"--> Saving to Zarr at {zarr_output_path}...")
    zarr.save(zarr_output_path, segmentation_data_xyz_relab, path="volumes/segmentation_rsg8") # pass either relabelled seg or original
    print("--> Zarr save complete.\n")

    # Save as a multi-page TIFF file
    print(f"--> Saving to TIFF at {tiff_output_path}...")
    tifffile.imwrite(tiff_output_path, segmentation_data_xyz_relab)
    print("--> TIFF save complete.\n")

    print("All operations finished successfully!")


if __name__ == '__main__':
    # ====================================================================
    # USER INPUT: Define the bounding box and output directory here
    # ====================================================================

    # Bounding box dimensions in XYZ format: [X_start, Y_start, Z_start, X_end, Y_end, Z_end]
    BBOX_DIMS_XYZ = [8083, 5878, 4697, 8765, 6542, 5319]
    #octo cube 1:8083_8765_y5878_6542_z4697_5319

    # Directory to save the output files
    OUTPUT_DIR = "/media/samia/DATA/mounts/zstore1/catena/lsd_outputs/FFNS_Octo"

    # ====================================================================

    crop_and_save_segmentation(bbox_dims_xyz=BBOX_DIMS_XYZ, output_dir=OUTPUT_DIR)
