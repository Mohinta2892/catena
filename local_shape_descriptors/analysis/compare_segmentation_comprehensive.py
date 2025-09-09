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
    Loads a volume from a TIFF or Zarr file.

    For Zarr, if a dataset is specified, it's loaded. If not, it attempts
    to load the path as a single Zarr array.

    Args:
        path (str): The file path to the volume.
        dataset (str, optional): The dataset key within a Zarr group.

    Returns:
        np.ndarray: The loaded volume as a NumPy array.
    """
    print(f"Loading volume from: {path}")
    if path.endswith(('.tif', '.tiff')):
        return tifffile.imread(path)
    elif path.endswith('.zarr'):
        try:
            store = zarr.open(path, mode='r')
            if dataset:
                print(f"  -> Reading dataset: '{dataset}'")
                return store[dataset][...]
            # If no dataset is given, check if the store itself is an array
            elif isinstance(store, zarr.Array):
                print("  -> Reading as a single Zarr array.")
                return store[...]
            else:
                raise ValueError(
                    f"Path '{path}' points to a Zarr group. "
                    "Please specify which dataset to load with --dset_a or --dset_b."
                )
        except Exception as e:
            print(f"Error loading Zarr store at {path}: {e}")
            raise
    else:
        raise ValueError(f"Unsupported file format for path: {path}. Please use .tif, .tiff, or .zarr.")


def calculate_skeleton_lengths_with_voxel_size(
        skeletons,
        position_attribute,
        voxel_size,
        store_edge_length_attribute,
        skeleton_id_attribute):
    """
    Calculates the length of each edge in the skeleton graph, scaled by the
    voxel size, and stores it in the specified edge attribute. Also returns
    a dictionary of total length per skeleton ID.
    (Adapted from eval_predictions_comprehensive.py)
    """
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


def calculate_nerl(predicted_seg: np.ndarray, gt_seg: np.ndarray, voxel_size: tuple):
    """
    Calculates the Normalized Expected Run Length (NERL) and ERL.

    The ground truth volume is skeletonized to evaluate the topological
    completeness of the predicted segmentation.

    Args:
        predicted_seg (np.ndarray): The segmentation being evaluated.
        gt_seg (np.ndarray): The ground truth segmentation.
        voxel_size (tuple): The (Z, Y, X) dimensions of a voxel.

    Returns:
        tuple: A tuple containing (NERL, ERL).
    """
    print("  -> Skeletonizing ground truth for NERL/ERL calculation...")
    skeletons = kimimaro.skeletonize(
        gt_seg.astype(np.uint32),
        anisotropy=voxel_size,
        parallel=0,  # Use all available cores
        progress=True
    )

    if not skeletons:
        print("  -> Warning: No skeletons found in ground truth. NERL/ERL are 0.")
        return 0.0, 0.0

    # Combine all skeleton objects into a single networkx graph
    skeleton_graph = nx.Graph()
    node_id_offset = 0
    for label, skel in skeletons.items():
        if skel.vertices.shape[0] == 0:
            continue
        num_vertices = skel.vertices.shape[0]
        node_ids = np.arange(num_vertices) + node_id_offset
        for i in range(num_vertices):
            skeleton_graph.add_node(
                node_ids[i],
                skeleton_id=label,
                zyx=skel.vertices[i]
            )
        for u, v in skel.edges:
            skeleton_graph.add_edge(u + node_id_offset, v + node_id_offset)
        node_id_offset += num_vertices

    nodes = list(skeleton_graph.nodes)
    if not nodes:
        print("  -> Warning: Skeleton graph is empty after processing. NERL/ERL are 0.")
        return 0.0, 0.0

    # Get the segmentation ID at each skeleton node location
    coords = np.array([skeleton_graph.nodes[n]['zyx'] for n in nodes]).astype(np.uint64)
    for d in range(predicted_seg.ndim):
        coords[:, d] = np.clip(coords[:, d], 0, predicted_seg.shape[d] - 1)
        
    segment_ids = predicted_seg[tuple(coords.T)]
    node_segment_lut = {node: segment_id for node, segment_id in zip(nodes, segment_ids)}
    
    # Calculate skeleton edge lengths considering voxel anisotropy
    skeleton_lengths = calculate_skeleton_lengths_with_voxel_size(
        skeleton_graph,
        position_attribute='zyx',
        voxel_size=voxel_size,
        store_edge_length_attribute='length',
        skeleton_id_attribute='skeleton_id'
    )

    # Calculate Expected Run Length
    erl = expected_run_length(
        skeletons=skeleton_graph,
        skeleton_id_attribute='skeleton_id',
        edge_length_attribute='length',
        node_segment_lut=node_segment_lut,
        skeleton_lengths=skeleton_lengths,
        return_merge_split_stats=False
    )
    
    total_gt_length = sum(skeleton_lengths.values())
    nerl = erl / total_gt_length if total_gt_length > 0 else 0.0
    
    return nerl, erl


def save_report_to_file(
    output_path: str,
    gt_name: str,
    pred_name: str,
    metrics: dict,
    neuron_ids: list = None
):
    """Saves a formatted evaluation report to a text file."""
    report_lines = [
        "=" * 80,
        "--- Segmentation Evaluation Report ---",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        f"Ground Truth (GT): {gt_name}",
        f"Prediction (Pred): {pred_name}"
    ]
    if neuron_ids:
        ids_str = ', '.join(map(str, neuron_ids))
        report_lines.append(f"Evaluated on Neuron IDs: {ids_str}")
    report_lines.append("-" * 80)

    for key, value in metrics.items():
        report_lines.append(f"{key:<25}: {value:<.4f}")

    report_lines.append("=" * 80)

    with open(output_path, 'w') as f:
        f.write("\n".join(report_lines))
    
    print(f"  -> Report saved to: {output_path}")


def run_bidirectional_comparison(vol_A, name_A, vol_B, name_B, voxel_size, output_dir, neuron_ids):
    """
    Calculates ARAND, VI, and NERL, prints a summary, and saves detailed
    reports for each direction.
    """
    print("\n" + "="*80)
    print(f"--- Running Bidirectional Evaluation ---")
    print(f"Volume A: {name_A} (Shape: {vol_A.shape})")
    print(f"Volume B: {name_B} (Shape: {vol_B.shape})")
    print("="*80 + "\n")

    if vol_A.shape != vol_B.shape:
        print("Warning: Input volumes have different shapes. Metrics may be unreliable.")

    # --- Direction 1: A is GT, B is Prediction ---
    print(f"--- Treating '{name_A}' as Ground Truth ---")
    splits_A, merges_A = skimage.metrics.variation_of_information(vol_A, vol_B, ignore_labels=(0,))
    arand_err_A, _, _ = skimage.metrics.adapted_rand_error(vol_A, vol_B, ignore_labels=(0,))
    nerl_A, erl_A = calculate_nerl(predicted_seg=vol_B, gt_seg=vol_A, voxel_size=voxel_size)
    
    metrics_A = {
        'VI False Splits': splits_A, 'VI False Merges': merges_A,
        'Adapted Rand Error': arand_err_A, 'NERL (Normalized)': nerl_A,
        'ERL (physical units)': erl_A
    }
    base_name_A = Path(name_A).stem
    base_name_B = Path(name_B).stem
    report_filename_A = f"{base_name_A}_as_GT_vs_{base_name_B}.txt"
    report_path_A = os.path.join(output_dir, report_filename_A)
    save_report_to_file(report_path_A, name_A, name_B, metrics_A, neuron_ids)
    print("-" * 50)

    # --- Direction 2: B is GT, A is Prediction ---
    print(f"\n--- Treating '{name_B}' as Ground Truth ---")
    splits_B, merges_B = skimage.metrics.variation_of_information(vol_B, vol_A, ignore_labels=(0,))
    arand_err_B, _, _ = skimage.metrics.adapted_rand_error(vol_B, vol_A, ignore_labels=(0,))
    nerl_B, erl_B = calculate_nerl(predicted_seg=vol_A, gt_seg=vol_B, voxel_size=voxel_size)

    metrics_B = {
        'VI False Splits': splits_B, 'VI False Merges': merges_B,
        'Adapted Rand Error': arand_err_B, 'NERL (Normalized)': nerl_B,
        'ERL (physical units)': erl_B
    }
    report_filename_B = f"{base_name_B}_as_GT_vs_{base_name_A}.txt"
    report_path_B = os.path.join(output_dir, report_filename_B)
    save_report_to_file(report_path_B, name_B, name_A, metrics_B, neuron_ids)
    print("-" * 50)

    # --- Console Summary Report ---
    print("\n\n" + "="*80)
    print("--- Bidirectional Comparison Summary (Console) ---")
    print("="*80)
    print(f"{'Metric':<25} | {'A as GT (B is Pred)':<25} | {'B as GT (A is Pred)':<25}")
    print("-" * 80)
    print(f"{'VI False Splits':<25} | {splits_A:<25.4f} | {splits_B:<25.4f}")
    print(f"{'VI False Merges':<25} | {merges_A:<25.4f} | {merges_B:<25.4f}")
    print(f"{'Adapted Rand Error':<25} | {arand_err_A:<25.4f} | {arand_err_B:<25.4f}")
    print(f"{'NERL (Normalized)':<25} | {nerl_A:<25.4f} | {nerl_B:<25.4f}")
    print(f"{'ERL (physical units)':<25} | {erl_A:<25.2f} | {erl_B:<25.2f}")
    print("="*80)

    print("\n--- How to Interpret This Report ---")
    print("• A high 'False Splits' when A is GT means an object in A is fragmented into multiple pieces in B.")
    print("• A high 'False Merges' when A is GT means several objects in A are incorrectly merged into one in B.")
    print("• NERL measures topological completeness. A low value indicates the prediction is fragmented.")
    print("• Look for asymmetries. High splits in one direction and high merges in the other strongly indicate a split/merge error.")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Bidirectional comparison of two segmentation volumes (Zarr or TIFF).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--vol_a', required=True, help="Path to the first segmentation volume (.zarr or .tif).")
    parser.add_argument('--dset_a', default=None, help="Dataset name for the first volume (if it's a Zarr group).")
    
    parser.add_argument('--vol_b', required=True, help="Path to the second segmentation volume (.zarr or .tif).")
    parser.add_argument('--dset_b', default=None, help="Dataset name for the second volume (if it's a Zarr group).")

    parser.add_argument(
        '--neuron_ids',
        default=None,
        help='Optional: A comma-separated list of neuron IDs to focus on (e.g., "10,25,30").\n'
             'If not provided, the entire volumes will be compared.'
    )
    parser.add_argument(
        '--voxel_size',
        nargs=3,
        type=float,
        default=[8.0, 8.0, 8.0],
        help='Voxel side lengths in physical units (Z Y X). Default: 8.0 8.0 8.0'
    )
    parser.add_argument(
        '--output_dir',
        default='.',
        help='Directory to save the output report files. Defaults to the current directory.'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='If set, display the two volumes in napari for visual inspection.'
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load volumes
    vol_A = load_volume(args.vol_a, args.dset_a)
    vol_B = load_volume(args.vol_b, args.dset_b)
    
    name_A = Path(args.vol_a).name
    name_B = Path(args.vol_b).name

    # Filter by neuron IDs if provided
    neuron_ids = parse_neuron_list(args.neuron_ids)
    if neuron_ids:
        print(f"\nFiltering volumes to include only IDs: {neuron_ids}")
        vol_A = vol_A * np.isin(vol_A, neuron_ids)
        vol_B = vol_B * np.isin(vol_B, neuron_ids)
    
    # Run the core evaluation
    run_bidirectional_comparison(
        vol_A=vol_A, name_A=name_A,
        vol_B=vol_B, name_B=name_B,
        voxel_size=tuple(args.voxel_size),
        output_dir=args.output_dir,
        neuron_ids=neuron_ids
    )

    # Optional visualization
    if args.visualize:
        if napari:
            print("Launching napari viewer...")
            viewer = napari.Viewer()
            viewer.add_labels(vol_A, name=name_A)
            viewer.add_labels(vol_B, name=name_B)
            napari.run()
        else:
            print("\nNapari is not installed. Skipping visualization.")
            print("To install, run: pip install napari[all]")


if __name__ == '__main__':
    main()


