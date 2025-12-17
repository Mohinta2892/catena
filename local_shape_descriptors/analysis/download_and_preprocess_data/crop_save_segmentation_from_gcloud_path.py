"""
Usage:
(ngl) samia@samia-PC-XB10250:/media/samia/DATA/PhD/codebases/proofread_CAVE$ 
python crop_save_segmentation_from_gcloud_path.py  gs://fly-larva-sf/octo/seg_241224_250131b_rsg8_spl --bbox 8083 5878 4697 8765 6542 5319 --mip 0 --output_dir /media/samia/DATA/mounts/zstore1/catena/lsd_outputs/FFNS_Octo

python crop_save_segmentation_from_gcloud_path.py  gs://fly-larva-sf/octo/seg_241224_250131b_rsg8_spl --bbox 12485 6231 3971 13164 6901 4640 --mip 0 --output_dir /media/samia/DATA/mounts/zstore1/catena/lsd_outputs/FFNS_Octo

python crop_save_segmentation_from_gcloud_path.py  gs://fly-larva-sf/octo/seg_241224_250131b_rsg8_spl --bbox 5603 3254 7464 6267 3890 8163 --mip 0 --output_dir /media/samia/DATA/mounts/zstore1/catena/lsd_outputs/FFNS_Octo

python crop_save_segmentation_from_gcloud_path.py  gs://fly-larva-sf/octo/seg_241224_250131b_rsg8_spl --bbox 5388 6586 7392 5900 7098 7904 --mip 0 --output_dir /media/samia/DATA/mounts/zstore1/catena/lsd_outputs/FFNS_Octo

x5388-5900 y6586-7098 z7392-7904
octo_z7392-7904_y6586-7098_x5388-5900_z120-140_y100-300_x100-300_wgt.zarr
OCTO_cube1_v2_8083_8765_y5878_6542_z4697_5319
OCTO_cube2_v2_12485_13164_y6231_6901_z3971_4640
OCTO_cube3_calyx_v2_5603_6267_y3254_3890_z7464_8163

raw:
python crop_save_segmentation_from_gcloud_path.py  gs://fly-larva-sf/octo/clahe/ --bbox 8083 5878 4697 8765 6542 5319 --mip 0 --output_dir /media/samia/DATA/mounts/zstore1/catena/lsd_outputs/FFNS_Octo_EM

Octo AL:
minCoords = [12029, 10063, 4495]
maxCoords = [13724, 11455, 5120]

python crop_save_segmentation_from_gcloud_path.py  gs://fly-larva-sf/octo/seg_241224_250131b_rsg8_spl --bbox 12029 10063 4495 13724 11455 5120 --mip 0 --output_dir /media/samia/DATA/mounts/zstore1/catena/lsd_outputs/FFNS_Octo



"""

import os
import argparse
import numpy as np
import zarr
import tifffile
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox
from skimage.segmentation import relabel_sequential

def crop_and_save_from_path(precomputed_path, bbox_dims_xyz, output_dir, mip):
    """
    Crops a segmentation volume from a precomputed path using CloudVolume,
    reformats it from XYZ to ZYX, and saves it as Zarr and TIFF files.

    Args:
        precomputed_path (str): The full precomputed path (e.g., 'gs://bucket/path/to/data').
        bbox_dims_xyz (list): A list of 6 integers defining the bounding
                              box in XYZ format: [x_start, y_start, z_start, x_end, y_end, z_end].
        output_dir (str): The directory where the output files will be saved.
        mip (int): The resolution level (mip) to download from.
    """
    # ---------------------------------------
    # 1. Setup and Validation
    # ---------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = (
        f"segmentation_crop_mip{mip}_"
        f"{bbox_dims_xyz[0]}-{bbox_dims_xyz[3]}_"
        f"{bbox_dims_xyz[1]}-{bbox_dims_xyz[4]}_"
        f"{bbox_dims_xyz[2]}-{bbox_dims_xyz[5]}"
    )
    zarr_output_path = os.path.join(output_dir, f"{base_filename}.zarr")
    tiff_output_path = os.path.join(output_dir, f"{base_filename}.tiff")

    print("--- Script Configuration ---")
    print(f"Precomputed Path: {precomputed_path}")
    print(f"MIP Level: {mip}")
    print(f"Input BBox (XYZ): {bbox_dims_xyz}")
    print(f"Output Directory: {output_dir}")
    print(f"Zarr Output: {zarr_output_path}")
    print(f"TIFF Output: {tiff_output_path}")
    print("-" * 28 + "\n")

    # ---------------------------------------
    # 2. Connect to the data source via CloudVolume
    # ---------------------------------------
    try:
        print(f"--> Connecting to CloudVolume at mip {mip}...")
        # Initialize CloudVolume with the specified path and mip level
        vol = CloudVolume(precomputed_path, mip=mip, fill_missing=True, progress=True)
        print(f"--> Connection successful. Volume info:")
        print(f"    - Resolution: {vol.resolution}")
        print(f"    - Size: {vol.bounds.size()}")
        print(f"    - Data Type: {vol.dtype}\n")
    except Exception as e:
        print(f"❌ Error: Could not connect to the CloudVolume path: {precomputed_path}")
        print(f"   Please check if the path is correct and you have the necessary permissions.")
        print(f"   Details: {e}")
        return

    # ---------------------------------------
    # 3. Download the data for the specified BBox
    # ---------------------------------------
    crop_bbox = Bbox.from_list(bbox_dims_xyz)
    
    print(f"--> Downloading data for bounding box: {crop_bbox}...")
    # Directly download the volume slice. This returns a numpy array.
    # The format is (X, Y, Z, Channels)
    segmentation_data_xyzc = vol[crop_bbox]

    # Remove the channel dimension, which is usually 1 for segmentation
    segmentation_data_xyz = np.squeeze(segmentation_data_xyzc)
    print(f"--> Download complete. Data shape (XYZ): {segmentation_data_xyz.shape}\n")

    # ---------------------------------------
    # 4. Transpose data from XYZ to ZYX format
    # ---------------------------------------
    print("--> Transposing data from XYZ to ZYX format...")
    # Original axes: 0=X, 1=Y, 2=Z -> Target axes: 2=Z, 1=Y, 0=X
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
    #print("--> Relabelling data...")
    segmentation_data_zyx_relab, _, _= relabel_sequential(segmentation_data_zyx) # pass either relabelled seg or original
    #segmentation_data_xyz_relab, _, _= relabel_sequential(segmentation_data_xyz)
    #print(f"--> Relabelling complete.")
    

    # ---------------------------------------
    # 5. Save the cropped data
    # ---------------------------------------
    print(f"--> Saving to Zarr at {zarr_output_path}...")
    zarr.save(zarr_output_path, segmentation_data_zyx_relab, path="volumes/segmentation_rsg8") # path within grp should be changed
    print("--> Zarr save complete.\n")

    print(f"--> Saving to TIFF at {tiff_output_path}...")
    tifffile.imwrite(tiff_output_path, segmentation_data_zyx_relab)
    print("--> TIFF save complete.\n")

    print("All operations finished successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Crop a region from a precomputed volume, transpose it to ZYX, and save as Zarr and TIFF.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        'precomputed_path',
        type=str,
        help="The full gs:// or file:// precomputed path to the segmentation data."
    )
    
    parser.add_argument(
        '--bbox',
        type=int,
        nargs=6,
        required=True,
        metavar=('X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2'),
        help="The bounding box for the crop in XYZ format, e.g., --bbox 20000 20000 5000 20512 20512 5512"
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default="./segmentation_crop_output",
        help="The directory to save the output Zarr and TIFF files. (Default: ./segmentation_crop_output)"
    )
    
    parser.add_argument(
        '--mip',
        type=int,
        default=0,
        help="The mip level (resolution) to download. (Default: 0 for full resolution)"
    )

    args = parser.parse_args()

    crop_and_save_from_path(
        precomputed_path=args.precomputed_path,
        bbox_dims_xyz=args.bbox,
        output_dir=args.output_dir,
        mip=args.mip
    )
