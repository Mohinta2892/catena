"""
conda activate nglancer
python viz_ffns_seg.py --seg_zarr /path/to/data.zarr
python viz_ffns_seg.py --seg_zarr /path/to/data.zarr --raw_path /path/to/raw_image.tif
python viz_ffns_seg.py --seg_zarr /media/samia/DATA/mounts/zstore1/catena/lsd_outputs/FFNS_Octo/segmentation_crop_mip0_12029-13724_10063-11455_4495-5120.zarr --raw_path /media/samia/DATA/mounts/zstore1/catena/lsd_outputs/FFNS_Octo_EM/segmentation_crop_mip0_8083-8765_5878-6542_4697-5319.zarr
"""

import neuroglancer
import zarr
import tifffile
import numpy as np
import dask.array as da
import argparse
import os
import sys

def load_raw_volume(path):
    """
    Loads raw data from either a TIFF file or a Zarr container.
    Expected Zarr path: volumes/raw
    """
    if not os.path.exists(path):
        print(f"Error: Raw data path not found: {path}")
        sys.exit(1)

    # Check for TIFF extension
    if path.lower().endswith(('.tif', '.tiff')):
        print(f"Loading Raw data from TIFF: {path}")
        try:
            data = tifffile.imread(path)
            return data
        except Exception as e:
            print(f"Failed to load TIFF: {e}")
            sys.exit(1)
    
    # Assume Zarr otherwise
    else:
        print(f"Loading Raw data from Zarr: {path}/volumes/raw")
        try:
            # We use dask.array.from_zarr to ensure the object supports 
            # numpy methods like .transpose() required by Neuroglancer meshing
            dask_arr = da.from_zarr(path, component='volumes/raw')
            return dask_arr
        except Exception as e:
            print(f"Failed to load Raw Zarr: {e}")
            # fallback/debug info
            try:
                z = zarr.open(path, mode='r')
                if 'volumes/raw' not in z:
                    print(f"Available keys in {path}: {list(z.keys())}")
            except:
                pass
            sys.exit(1)

def load_segmentation_volume(path):
    """
    Loads segmentation data from a Zarr container using Dask.
    Expected Zarr path: volumes/segmentation_rsg8
    """
    if not os.path.exists(path):
        print(f"Error: Segmentation path not found: {path}")
        sys.exit(1)

    print(f"Loading Segmentation from Zarr: {path}/volumes/segmentation_rsg8")
    try:
        # Load as Dask array to support on-the-fly meshing (requires .transpose())
        dask_arr = da.from_zarr(path, component='volumes/segmentation_rsg8')
        return dask_arr
    except Exception as e:
        print(f"Failed to load Segmentation Zarr: {e}")
        # Helper: check if 'volumes' exists and list its content for debugging
        try:
            z = zarr.open(path, mode='r')
            if 'volumes' in z:
                 print(f"Content of 'volumes': {list(z['volumes'].keys())}")
        except:
            pass
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Visualize Zarr/Tiff EM data in Neuroglancer.")
    
    # Arguments
    parser.add_argument('--seg_zarr', type=str, required=True, 
                        help='Path to the Zarr file containing volumes/segmentation_rsg8')
    parser.add_argument('--raw_path', type=str, required=False, 
                        help='Path to raw data (Zarr or TIFF). If omitted, uses --seg_zarr path.')
    parser.add_argument('--voxel_size', type=float, nargs=3, default=[8, 8, 8],
                        help='Voxel size in nanometers (x y z). Default: 8 8 8')
    
    args = parser.parse_args()

    # Determine Raw Path (default to seg_zarr if not provided)
    raw_input_path = args.raw_path if args.raw_path else args.seg_zarr

    # 1. Load Data
    raw_vol = load_raw_volume(raw_input_path)
    seg_vol = load_segmentation_volume(args.seg_zarr)

    # 2. Setup Neuroglancer Viewer
    neuroglancer.set_server_bind_address('0.0.0.0')
    viewer = neuroglancer.Viewer()

    # 3. Define Coordinate Space (resolution)
    res = neuroglancer.CoordinateSpace(
        names=['z', 'y', 'x'],
        units='nm',
        scales=[args.voxel_size[2], args.voxel_size[1], args.voxel_size[0]]
    )

    # 4. Add Layers
    with viewer.txn() as s:
        
        # Add Raw Layer
        s.layers['raw'] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(raw_vol, dimensions=res),
            shader="""
#uicontrol float black slider(min=0, max=255, default=0)
#uicontrol float white slider(min=0, max=255, default=255)
void main() {
  float val = toNormalized(getDataValue());
  float b = black / 255.0;
  float w = white / 255.0;
  emitGrayscale((val - b) / (w - b));
}
"""
        )

        # Add Segmentation Layer
        s.layers['segmentation'] = neuroglancer.SegmentationLayer(
            source=neuroglancer.LocalVolume(seg_vol, dimensions=res),
            selected_alpha=0.5,
            object_alpha=0.0 
        )

    print(f"\nVOLUMES LOADED:")
    print(f"Raw shape: {raw_vol.shape}")
    print(f"Seg shape: {seg_vol.shape}")
    print(f"\nViewer running at:\n{viewer}")
    
    print("Press Enter to quit...")
    input()

if __name__ == "__main__":
    main()
