"""
Usage:
python compare_segmentation_comprehensive.py --vol_a /mnt/scratch/mounts/zstore1/catena/lsd_outputs/FFNS_Octo/octo_cube1_ffn_8083-8765_5878-6542_4697-5319.tiff --vol_b /mnt/scratch/mounts/fibserver/smohinta_data/lsd_outputs/MTLSD/3d/run-aclsd-together/model_checkpoint_latest/cropped_from_blackwell/octo_cube1_8083_8765_y5878_6542_z4697_5319.zarr --dset_b volumes/segmentation_05 --voxel_size 8 8 8 --output_dir /mnt/scratch/mounts/zstore1/catena/lsd_outputs/FFN_LSD_comparison/

python compare_segmentation_comprehensive.py --vol_a /mnt/scratch/mounts/zstore1/catena/lsd_outputs/FFNS_Octo/octo_cube1_ffn_8083-8765_5878-6542_4697-5319.tiff --vol_b /mnt/scratch/mounts/fibserver/smohinta_data/lsd_outputs/MTLSD/3d/run-aclsd-together/model_checkpoint_latest/cropped_from_blackwell/octo_cube1_8083_8765_y5878_6542_z4697_5319.zarr --dset_b volumes/segmentation_06 --voxel_size 8 8 8 --output_dir /mnt/scratch/mounts/zstore1/catena/lsd_outputs/FFN_LSD_comparison/


python compare_segmentation_comprehensive.py --vol_a /mnt/scratch/mounts/zstore1/catena/lsd_outputs/FFNS_Octo/octo_cube2_ffn_12485-13164_6231-6901_3971-4640.tiff --vol_b /mnt/scratch/mounts/fibserver/smohinta_data/lsd_outputs/MTLSD/3d/run-aclsd-together/model_checkpoint_latest/cropped_from_blackwell/octo_cube2_12485_13164_y6231_6901_z3971_4640.zarr --dset_b volumes/segmentation_05 --voxel_size 8 8 8 --output_dir /mnt/scratch/mounts/zstore1/catena/lsd_outputs/FFN_LSD_comparison/

python compare_segmentation_comprehensive.py --vol_a /mnt/scratch/mounts/zstore1/catena/lsd_outputs/FFNS_Octo/octo_cube2_ffn_12485-13164_6231-6901_3971-4640.tiff --vol_b /mnt/scratch/mounts/fibserver/smohinta_data/lsd_outputs/MTLSD/3d/run-aclsd-together/model_checkpoint_latest/cropped_from_blackwell/octo_cube2_12485_13164_y6231_6901_z3971_4640.zarr --dset_b volumes/segmentation_06 --voxel_size 8 8 8 --output_dir /mnt/scratch/mounts/zstore1/catena/lsd_outputs/FFN_LSD_comparison/


python compare_segmentation_comprehensive.py --vol_a /mnt/scratch/mounts/zstore1/catena/lsd_outputs/FFNS_Octo/octo_cube3_ffn_5603-6267_3254-3890_7464-8163.tiff --vol_b /mnt/scratch/mounts/fibserver/smohinta_data/lsd_outputs/MTLSD/3d/run-aclsd-together/model_checkpoint_latest/cropped_from_blackwell/octo_cube3_calyx_5603_6267_y3254_3890_z7464_8163.zarr --dset_b volumes/segmentation_05 --voxel_size 8 8 8 --output_dir /mnt/scratch/mounts/zstore1/catena/lsd_outputs/FFN_LSD_comparison/

python compare_segmentation_comprehensive.py --vol_a /mnt/scratch/mounts/zstore1/catena/lsd_outputs/FFNS_Octo/octo_cube3_ffn_5603-6267_3254-3890_7464-8163.tiff --vol_b /mnt/scratch/mounts/fibserver/smohinta_data/lsd_outputs/MTLSD/3d/run-aclsd-together/model_checkpoint_latest/cropped_from_blackwell/octo_cube3_calyx_5603_6267_y3254_3890_z7464_8163.zarr --dset_b volumes/segmentation_06 --voxel_size 8 8 8 --output_dir /mnt/scratch/mounts/zstore1/catena/lsd_outputs/FFN_LSD_comparison/
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
import zarr
import tifffile
import numpy as np
import skimage.metrics
import kimimaro
import networkx as nx
from funlib.evaluate import expected_run_length

# Optional import for visualization
try:
    import napari
except ImportError:
    napari = None


def parse_neuron_list(neuron_list_str: str):
    """Parses a comma-separated string of integers into a list."""
    if not neuron_list_str:
        return None
    try:
        return [int(x.strip()) for x in neuron_list_str.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("Neuron ID list must be a comma-separated list of integers.")


def load_volume(path: str, dataset: str = None) -> np.ndarray:
    """
    Loads a volume from a TIFF or Zarr (v2 or v3) file with robust error handling.

    Args:
        path (str): The file path to the volume.
        dataset (str, optional): The dataset key within a Zarr group. Can be a nested
                                 path like 'group/subgroup/array'.

    Returns:
        np.ndarray: The loaded volume as a NumPy array.
    """
    print(f"Loading volume from: {path}")
    if path.endswith(('.tif', '.tiff')):
        return tifffile.imread(path)
    elif path.endswith('.zarr'):
        try:
            print(f"  -> Attempting to open Zarr store (v2/v3 compatible)...")
            store_root = zarr.open(path, mode='r')

            zarr_version = getattr(store_root.store, 'zarr_format', None)
            if zarr_version:
                print(f"  -> Successfully opened as Zarr v{zarr_version} format.")
            else:
                print("  -> Could not determine Zarr version, proceeding...")

            if dataset:
                print(f"  -> Attempting to read dataset: '{dataset}'")
                if dataset not in store_root:
                    all_arrays = []
                    def find_arrays(name, obj):
                        if isinstance(obj, zarr.Array):
                            all_arrays.append(name)
                    store_root.visititems(find_arrays)
                    
                    error_msg = (
                        f"\n\n--- KEY ERROR ---\n"
                        f"Dataset '{dataset}' was NOT found inside '{path}'.\n\n"
                        f"Please check your --dset argument. The available datasets are:\n"
                    )
                    if not all_arrays:
                         error_msg += "  - No datasets found in this Zarr store.\n"
                    for arr_path in sorted(all_arrays):
                        error_msg += f"  - '{arr_path}'\n"
                    error_msg += "-------------------\n"
                    raise KeyError(error_msg)
                
                return store_root[dataset][...]
            
            elif isinstance(store_root, zarr.Array):
                print("  -> Reading as a single Zarr array at the root.")
                return store_root[...]
            
            else:
                all_paths = [name for name, _ in store_root.visititems()]
                raise ValueError(
                    f"Path '{path}' points to a Zarr group, but no dataset was specified. "
                    "Please use --dset_a or --dset_b to select one. "
                    f"Available paths: {sorted(all_paths)}"
                )
        except Exception as e:
            print(f"\nERROR: An unexpected error occurred while loading Zarr store at '{path}'.\nDetails: {e}")
            raise
    else:
        raise ValueError(f"Unsupported file format for path: {path}. Please use .tif, .tiff, or .zarr.")


def calculate_skeleton_lengths_with_voxel_size(
        skeletons,
        position_attribute,
        voxel_size,
        store_edge_length_attribute,
        skeleton_id_attribute):
    """Calculates the length of each edge in the skeleton graph, scaled by voxel size."""
    skeleton_lengths = {}
    for u, v, data in skeletons.edges(data=True):
        skeleton_id = skeletons.nodes[u][skeleton_id_attribute]
        pos_u = np.array(skeletons.nodes[u][position_attribute], dtype=np.float32)
        pos_v = np.array(skeletons.nodes[v][position_attribute], dtype=np.float32)

        scaled_diff = (pos_u - pos_v) * voxel_size
        length = np.linalg.norm(scaled_diff)
        data[store_edge_length_attribute] = length

        if skeleton_id not in skeleton_lengths:
            skeleton_lengths[skeleton_id] = 0
        skeleton_lengths[skeleton_id] += length
    
    return skeleton_lengths


def calculate_nerl(predicted_seg: np.ndarray, gt_seg: np.ndarray, voxel_size: tuple, dust_threshold: int):
    """Calculates the Normalized Expected Run Length (NERL) and ERL."""
    # --- Diagnostic prints ---
    print(f"  -> GT volume for skeletonization: dtype={gt_seg.dtype}, shape={gt_seg.shape}")
    unique_labels = np.unique(gt_seg)
    print(f"  -> Found {len(unique_labels)} unique labels in GT volume (including background).")
    if len(unique_labels) <= 1:
        print("  -> Warning: Ground truth volume is empty or has only one label. Cannot skeletonize. NERL/ERL will be 0.")
        return 0.0, 0.0
    # --- End diagnostics ---
    
    print(f"  -> Skeletonizing ground truth (dust_threshold={dust_threshold})...")
    skeletons = kimimaro.skeletonize(
        gt_seg.astype(np.uint32),
        anisotropy=voxel_size,
        dust_threshold=dust_threshold, # Use the provided dust threshold
        parallel=0,
        progress=True
    )

    # --- Diagnostic print ---
    print(f"  -> Found {len(skeletons)} skeletons.")
    # --- End diagnostics ---

    if not skeletons:
        print("  -> Warning: No skeletons were generated. NERL/ERL will be 0.")
        return 0.0, 0.0

    skeleton_graph = nx.Graph()
    node_id_offset = 0
    for label, skel in skeletons.items():
        if skel.vertices.shape[0] == 0: continue
        num_vertices = skel.vertices.shape[0]
        node_ids = np.arange(num_vertices) + node_id_offset
        for i in range(num_vertices):
            skeleton_graph.add_node(node_ids[i], skeleton_id=label, zyx=skel.vertices[i])
        for u, v in skel.edges:
            skeleton_graph.add_edge(u + node_id_offset, v + node_id_offset)
        node_id_offset += num_vertices

    nodes = list(skeleton_graph.nodes)
    if not nodes:
        print("  -> Warning: Skeleton graph is empty after processing. NERL/ERL are 0.")
        return 0.0, 0.0

    coords = np.array([skeleton_graph.nodes[n]['zyx'] for n in nodes]).astype(np.uint64)
    for d in range(predicted_seg.ndim):
        coords[:, d] = np.clip(coords[:, d], 0, predicted_seg.shape[d] - 1)
        
    segment_ids = predicted_seg[tuple(coords.T)]
    node_segment_lut = {node: segment_id for node, segment_id in zip(nodes, segment_ids)}
    
    skeleton_lengths = calculate_skeleton_lengths_with_voxel_size(
        skeleton_graph, 'zyx', voxel_size, 'length', 'skeleton_id'
    )

    erl = expected_run_length(
        skeletons=skeleton_graph, skeleton_id_attribute='skeleton_id',
        edge_length_attribute='length', node_segment_lut=node_segment_lut,
        skeleton_lengths=skeleton_lengths, return_merge_split_stats=False
    )
    
    total_gt_length = sum(skeleton_lengths.values())
    nerl = erl / total_gt_length if total_gt_length > 0 else 0.0
    
    return nerl, erl


def save_report_to_file(output_path, gt_name, pred_name, metrics, neuron_ids=None):
    """Saves a formatted evaluation report to a text file."""
    report_lines = [
        "="*80, f"--- Segmentation Evaluation Report ---", f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "="*80, f"Ground Truth (GT): {gt_name}", f"Prediction (Pred): {pred_name}"
    ]
    if neuron_ids:
        report_lines.append(f"Evaluated on Neuron IDs: {', '.join(map(str, neuron_ids))}")
    report_lines.append("-" * 80)
    for key, value in metrics.items():
        report_lines.append(f"{key:<25}: {value:<.4f}")
    report_lines.append("=" * 80)
    with open(output_path, 'w') as f:
        f.write("\n".join(report_lines))
    print(f"  -> Report saved to: {output_path}")


def run_bidirectional_comparison(vol_A, name_A, vol_B, name_B, voxel_size, output_dir, neuron_ids, skel_dust_threshold):
    """Calculates metrics, prints a summary, and saves reports."""
    print("\n" + "="*80, f"\n--- Running Bidirectional Evaluation ---")
    print(f"Volume A: {name_A} (Shape: {vol_A.shape})", f"\nVolume B: {name_B} (Shape: {vol_B.shape})", "="*80 + "\n")
    if vol_A.shape != vol_B.shape: print("Warning: Input volumes have different shapes.")

    print(f"--- Treating '{name_A}' as Ground Truth ---")
    splits_A, merges_A = skimage.metrics.variation_of_information(vol_A, vol_B, ignore_labels=(0,))
    arand_err_A, _, _ = skimage.metrics.adapted_rand_error(vol_A, vol_B, ignore_labels=(0,))
    nerl_A, erl_A = calculate_nerl(vol_B, vol_A, voxel_size, skel_dust_threshold)
    metrics_A = {'VI False Splits': splits_A, 'VI False Merges': merges_A, 'Adapted Rand Error': arand_err_A, 'NERL (Normalized)': nerl_A, 'ERL (physical units)': erl_A}
    report_path_A = Path(output_dir) / f"{Path(name_A).stem}_as_GT_vs_{Path(name_B).stem}.txt"
    save_report_to_file(report_path_A, name_A, name_B, metrics_A, neuron_ids)
    print("-" * 50)

    print(f"\n--- Treating '{name_B}' as Ground Truth ---")
    splits_B, merges_B = skimage.metrics.variation_of_information(vol_B, vol_A, ignore_labels=(0,))
    arand_err_B, _, _ = skimage.metrics.adapted_rand_error(vol_B, vol_A, ignore_labels=(0,))
    nerl_B, erl_B = calculate_nerl(vol_A, vol_B, voxel_size, skel_dust_threshold)
    metrics_B = {'VI False Splits': splits_B, 'VI False Merges': merges_B, 'Adapted Rand Error': arand_err_B, 'NERL (Normalized)': nerl_B, 'ERL (physical units)': erl_B}
    report_path_B = Path(output_dir) / f"{Path(name_B).stem}_as_GT_vs_{Path(name_A).stem}.txt"
    save_report_to_file(report_path_B, name_B, name_A, metrics_B, neuron_ids)
    print("-" * 50)

    print("\n\n" + "="*80, "\n--- Bidirectional Comparison Summary ---", "="*80)
    print(f"{'Metric':<25} | {'A as GT (B is Pred)':<25} | {'B as GT (A is Pred)':<25}")
    print("-" * 80)
    print(f"{'VI False Splits':<25} | {splits_A:<25.4f} | {splits_B:<25.4f}")
    print(f"{'VI False Merges':<25} | {merges_A:<25.4f} | {merges_B:<25.4f}")
    print(f"{'Adapted Rand Error':<25} | {arand_err_A:<25.4f} | {arand_err_B:<25.4f}")
    print(f"{'NERL (Normalized)':<25} | {nerl_A:<25.4f} | {nerl_B:<25.4f}")
    print(f"{'ERL (physical units)':<25} | {erl_A:<25.2f} | {erl_B:<25.2f}")
    print("="*80, "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Bidirectional comparison of two segmentation volumes (Zarr or TIFF).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--vol_a', required=True, help="Path to the first segmentation volume (.zarr or .tif).")
    parser.add_argument('--dset_a', default=None, help="Dataset name for the first volume (if Zarr).")
    parser.add_argument('--vol_b', required=True, help="Path to the second segmentation volume (.zarr or .tif).")
    parser.add_argument('--dset_b', default=None, help="Dataset name for the second volume (if Zarr).")
    parser.add_argument('--neuron_ids', default=None, help='Optional: Comma-separated list of neuron IDs to compare (e.g., "10,25").')
    parser.add_argument('--voxel_size', nargs=3, type=float, default=[8.0, 8.0, 8.0], help='Voxel dimensions in nm (Z Y X). Default: 8 8 8')
    parser.add_argument('--output_dir', default='.', help='Directory to save report files. Defaults to current directory.')
    parser.add_argument(
        '--skel_dust_threshold',
        type=int,
        default=1000,
        help='Minimum voxel size for an object to be skeletonized by kimimaro. Default: 1000'
    )
    parser.add_argument('--visualize', action='store_true', help='If set, display volumes in napari.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    vol_A = load_volume(args.vol_a, args.dset_a)
    vol_B = load_volume(args.vol_b, args.dset_b)
    name_A = Path(args.vol_a).name
    name_B = Path(args.vol_b).name

    neuron_ids = parse_neuron_list(args.neuron_ids)
    if neuron_ids:
        print(f"\nFiltering volumes to include only IDs: {neuron_ids}")
        vol_A = vol_A * np.isin(vol_A, neuron_ids)
        vol_B = vol_B * np.isin(vol_B, neuron_ids)
    
    run_bidirectional_comparison(
        vol_A, name_A, vol_B, name_B, 
        tuple(args.voxel_size), 
        args.output_dir, 
        neuron_ids,
        args.skel_dust_threshold
    )

    if args.visualize:
        if napari:
            print("Launching napari viewer...")
            viewer = napari.Viewer()
            viewer.add_labels(vol_A, name=name_A)
            viewer.add_labels(vol_B, name=name_B)
            napari.run()
        else:
            print("\nNapari is not installed. Skipping visualization (pip install napari[all]).")

if __name__ == '__main__':
    main()
