"""
conda activate ngl
Usage Example (Fetch Skeletons/Meshes using an EXISTING local file):
python crop_save_segmentation_skel_from_gcloud.py gs://fly-larva-sf/octo/seg_241224_250131b_rsg8_spl \
    --local_segmentation ./output/segmentation_crop.zarr \
    --save_skeletons --save_meshes \
    --output_dir ./output/morphology

Usage Example (Standard - Download Everything):
python crop_save_segmentation_skel_from_gcloud.py gs://fly-larva-sf/octo/seg_241224_250131b_rsg8_spl \
    --bbox 8083 5878 4697 8765 6542 5319 \
    --save_skeletons --save_meshes
    
python crop_save_segmentation_skel_from_gcloud.py  precomputed://file:///media/samia/DATA/mounts/gpu2/biomedparse/octo/seg_241224_250131b_rsg8_spl --bbox 12029 10063 4495 13724 11455 5120 --mip 0 --output_dir /media/samia/DATA/mounts/zstore1/catena/lsd_outputs/FFNS_Octo_Skel --save_skeletons --save_meshes
"""


import os
import argparse
import numpy as np
import zarr
import tifffile
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox
from tqdm import tqdm
import concurrent.futures

def save_skeleton_to_swc(skel, filepath, scalar_offset=[0,0,0]):
    """
    Saves a CloudVolume Skeleton object to a standard .swc file.
    """
    with open(filepath, 'w') as f:
        f.write(f"# Skeleton ID: {skel.id}\n")
        f.write("# id type x y z radius parent\n")
        
        parents = np.full(len(skel.vertices), -1, dtype=int)
        
        for edge in skel.edges:
            src, dst = edge
            if parents[dst] == -1 and src != dst:
                parents[dst] = src
            elif parents[src] == -1 and src != dst:
                parents[src] = dst
        
        for i, vertex in enumerate(skel.vertices):
            x, y, z = vertex + scalar_offset
            r = skel.radii[i] if skel.radii is not None else 10.0
            p = parents[i]
            parent_id = p + 1 if p != -1 else -1
            f.write(f"{i+1} 0 {x:.2f} {y:.2f} {z:.2f} {r:.2f} {parent_id}\n")

def save_mesh_to_obj(mesh, filepath):
    """
    Saves a CloudVolume Mesh object to a standard .obj file.
    """
    with open(filepath, 'w') as f:
        f.write(f"# Mesh ID: {mesh.id}\n")
        for v in mesh.vertices:
            f.write(f"v {v[0]:.2f} {v[1]:.2f} {v[2]:.2f}\n")
        for face in mesh.faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def get_unique_ids_chunked(data_array):
    unique_ids = set()
    print(f"--> Calculating unique IDs chunkwise (Input shape: {data_array.shape})...")
    
    try:
        iterator = tqdm(range(data_array.shape[0]), desc="Finding IDs", unit="slice")
    except ImportError:
        iterator = range(data_array.shape[0])

    for i in iterator:
        slice_uniques = np.unique(data_array[i])
        unique_ids.update(slice_uniques)
        
    if 0 in unique_ids:
        unique_ids.remove(0)
        
    return np.array(sorted(list(unique_ids)))

def load_local_data(path):
    print(f"--> Loading local segmentation from: {path}")
    if path.endswith('.zarr'):
        try:
            z = zarr.open(path, mode='r')
            if isinstance(z, zarr.Group):
                if 'volumes/segmentation' in z:
                    return z['volumes/segmentation']
                elif 'volumes/segmentation_rsg8' in z:
                    return z['volumes/segmentation_rsg8']
                else:
                    keys = list(z.array_keys())
                    if keys:
                        return z[keys[0]]
                    else:
                        raise ValueError("Zarr group contains no arrays.")
            return z
        except Exception as e:
            print(f"❌ Error opening Zarr: {e}")
            return None
    elif path.endswith(('.tiff', '.tif')):
        try:
            return tifffile.imread(path)
        except Exception as e:
            print(f"❌ Error opening TIFF: {e}")
            return None
    else:
        print("❌ Error: Unsupported file format. Use .zarr or .tiff")
        return None

def crop_and_save_from_path(precomputed_path, output_dir, mip, bbox_dims_xyz=None, local_seg_path=None, fetch_skeletons=False, fetch_meshes=False):
    
    # ---------------------------------------
    # 1. Setup
    # ---------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    skel_dir = os.path.join(output_dir, "skeletons")
    mesh_dir = os.path.join(output_dir, "meshes")
    
    if fetch_skeletons: os.makedirs(skel_dir, exist_ok=True)
    if fetch_meshes: os.makedirs(mesh_dir, exist_ok=True)

    print("--- Script Configuration ---")
    print(f"Precomputed Path:  {precomputed_path}")
    print(f"Local Segmentation:{local_seg_path if local_seg_path else 'None (Download from Cloud)'}")
    print("-" * 28 + "\n")

    # ---------------------------------------
    # 2. Get Segmentation Data
    # ---------------------------------------
    segmentation_data = None
    if local_seg_path:
        segmentation_data = load_local_data(local_seg_path)
        if segmentation_data is None: return
    else:
        if bbox_dims_xyz is None:
            print("❌ Error: You must provide --bbox if not using a local file.")
            return
        try:
            print(f"--> Connecting to CloudVolume at mip {mip}...")
            vol = CloudVolume(precomputed_path, mip=mip, fill_missing=True, progress=True)
            crop_bbox = Bbox.from_list(bbox_dims_xyz)
            print(f"--> Downloading data for bbox: {crop_bbox}...")
            data_xyzc = vol[crop_bbox]
            segmentation_data = np.squeeze(data_xyzc)
            
            # Save downloaded volume
            data_zyx = np.transpose(segmentation_data, (2, 1, 0))
            base_filename = f"seg_{bbox_dims_xyz[0]}-{bbox_dims_xyz[3]}_{bbox_dims_xyz[1]}-{bbox_dims_xyz[4]}_{bbox_dims_xyz[2]}-{bbox_dims_xyz[5]}"
            zpath = os.path.join(output_dir, f"{base_filename}.zarr")
            tpath = os.path.join(output_dir, f"{base_filename}.tiff")
            zarr.save(zpath, data_zyx, path="volumes/segmentation")
            tifffile.imwrite(tpath, data_zyx)
        except Exception as e:
            print(f"❌ Error during CloudVolume download: {e}")
            return

    # ---------------------------------------
    # 3. Extract IDs
    # ---------------------------------------
    unique_ids = []
    if fetch_skeletons or fetch_meshes:
        if segmentation_data is not None:
            is_huge = False
            # Check nbytes safely
            if hasattr(segmentation_data, 'nbytes') and segmentation_data.nbytes > 100 * 1024**2:
                is_huge = True
            
            if is_huge:
                unique_ids = get_unique_ids_chunked(segmentation_data)
            else:
                loaded_data = segmentation_data[:] 
                unique_ids = np.unique(loaded_data)
                if 0 in unique_ids: unique_ids = unique_ids[unique_ids != 0]

            print(f"--> Found {len(unique_ids)} unique segments.")
        else:
            print("❌ Error: No segmentation data available.")
            return

    # ---------------------------------------
    # 4. Connect for Metadata
    # ---------------------------------------
    try:
        # We create a new CloudVolume instance for metadata fetching
        vol = CloudVolume(precomputed_path, mip=mip, progress=False) # Turn off default progress to avoid conflict
    except Exception as e:
        print(f"❌ Error connecting to CloudVolume: {e}")
        return

    # ---------------------------------------
    # 5. Fetch Skeletons (Fault Tolerant)
    # ---------------------------------------
    if fetch_skeletons and len(unique_ids) > 0:
        print(f"\n--> Fetching {len(unique_ids)} skeletons (skipping missing ones)...")
        ids_to_fetch = [int(x) for x in unique_ids]
        
        # Helper function for threading
        def fetch_single_skeleton(skel_id):
            try:
                # Retrieve scalar ID -> returns object, raises error if missing
                skel = vol.skeleton.get(skel_id)
                out_name = os.path.join(skel_dir, f"{skel.id}.swc")
                save_skeleton_to_swc(skel, out_name)
                return 1 # Success
            except Exception:
                return 0 # Fail/Missing

        # Use ThreadPool to speed up single-file processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            # We map the function over IDs and sum the results (1 for success, 0 for fail)
            results = list(tqdm(executor.map(fetch_single_skeleton, ids_to_fetch), 
                                total=len(ids_to_fetch), 
                                desc="Downloading Skeletons"))
            
        success_count = sum(results)
        print(f"--> Done. Successfully saved {success_count}/{len(unique_ids)} skeletons.")

    # ---------------------------------------
    # 6. Fetch Meshes (Fault Tolerant)
    # ---------------------------------------
    if fetch_meshes and len(unique_ids) > 0:
        print(f"\n--> Fetching {len(unique_ids)} meshes (skipping missing ones)...")
        ids_to_fetch = [int(x) for x in unique_ids]
        
        def fetch_single_mesh(mesh_id):
            try:
                # CloudVolume mesh.get can take a scalar or list. 
                # Scalar returns a dict {id: mesh} usually, or raises error.
                # Safe way: request list of 1.
                mesh_dict = vol.mesh.get([mesh_id])
                if mesh_id in mesh_dict:
                    mesh = mesh_dict[mesh_id]
                    out_name = os.path.join(mesh_dir, f"{mesh.id}.obj")
                    save_mesh_to_obj(mesh, out_name)
                    return 1
                return 0
            except Exception:
                return 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            results = list(tqdm(executor.map(fetch_single_mesh, ids_to_fetch), 
                                total=len(ids_to_fetch), 
                                desc="Downloading Meshes"))
            
        success_count = sum(results)
        print(f"--> Done. Successfully saved {success_count}/{len(unique_ids)} meshes.")

    print("\nAll operations finished successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('precomputed_path', type=str)
    parser.add_argument('--bbox', type=int, nargs=6, required=False)
    parser.add_argument('--local_segmentation', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="./output_morphology")
    parser.add_argument('--mip', type=int, default=0)
    parser.add_argument('--save_skeletons', action='store_true')
    parser.add_argument('--save_meshes', action='store_true')

    args = parser.parse_args()

    if not args.local_segmentation and not args.bbox:
        parser.error("Provide --bbox OR --local_segmentation.")

    crop_and_save_from_path(
        precomputed_path=args.precomputed_path,
        output_dir=args.output_dir,
        mip=args.mip,
        bbox_dims_xyz=args.bbox,
        local_seg_path=args.local_segmentation,
        fetch_skeletons=args.save_skeletons,
        fetch_meshes=args.save_meshes
    )
