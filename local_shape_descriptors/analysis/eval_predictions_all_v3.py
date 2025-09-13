#!/usr/bin/env python

"""
run in blackwell:
conda activate lsd_analysis 
pip install git+https://github.com/funkelab/funlib.evaluate.git
conda install -c conda-forge graph-tool
pip install kimimaro

Usage Examples:
# Basic (seg + gt only), skip slow ERL
python eval_predictions_all_v2.py --gt ... --seg ... --voxel_size 8 8 8 --no-erl

python eval_predictions_all_v3.py --gt /mnt/scratch/mounts/ark/dan-samia/lsd/funke/otto/tiff/octo_z7392-7904_y6586-7098_x5388-5900_z120-140_y100-300_x100-300_wgt.zarr --gt_ds volumes/labels/neuron_ids --seg /mnt/scratch/mounts/fibserver/smohinta_data/lsd_outputs/MTLSD/3d/run-aclsd-together/model_checkpoint_latest/cropped_from_blackwell/octo_z7392-7904_y6586-7098_x5388-5900_z120-140_y100-300_x100-300_ffn.zarr --seg_ds volumes/segmentation_rsg8 --voxel_size 8 8 8


# Full evaluation with raw data in a separate file
python eval_predictions_all_v2.py --gt ... --seg ... --aff ... --raw /path/to/raw.zarr --voxel_size 8 8 8

python eval_predictions_all_v3.py --gt /mnt/scratch/mounts/ark/dan-samia/lsd/funke/otto/tiff/octo_z7392-7904_y6586-7098_x5388-5900_z120-140_y100-300_x100-300_wgt.zarr --gt_ds volumes/labels/neuron_ids --seg /mnt/scratch/lsd_outputs/MTLSD/3d/hemi_histomatched_octo/model_checkpoint_300000/otto_z7392-7904_y6586-7098_x5388-5900_z120-140_y100-300_x100-300.zarr --seg_ds volumes/segmentation_05 --aff /mnt/scratch/lsd_outputs/MTLSD/3d/hemi_histomatched_octo/model_checkpoint_300000/otto_z7392-7904_y6586-7098_x5388-5900_z120-140_y100-300_x100-300.zarr --aff_ds volumes/pred_affs --raw_ds volumes/raw --voxel_size 8 8 8
"""

import numpy as np
import skimage
import zarr
import argparse
import os
import kimimaro
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.ndimage import binary_dilation, center_of_mass
from scipy.spatial.distance import directed_hausdorff
from skimage.segmentation import find_boundaries
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
from funlib.segment.arrays import relabel
from funlib.evaluate import expected_run_length
from skimage import graph
from tqdm import tqdm


def get_unique_filename(base_filename):
    """
    Checks if a file exists. If so, appends a number to make it unique.
    """
    if not base_filename.endswith('.txt'):
        base_filename += '.txt'
    
    if not os.path.exists(base_filename):
        return base_filename
    
    base, ext = os.path.splitext(base_filename)
    counter = 1
    while os.path.exists(f"{base}_{counter}{ext}"):
        counter += 1
    return f"{base}_{counter}{ext}"


def save_metrics_to_file(metrics, filename):
    """
    Saves the computed metrics to a text file.
    """
    with open(filename, 'w') as f:
        f.write(f"Segmentation Evaluation Report\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*40 + "\n\n")

        # Define a consistent order for categories
        category_order = [
            'Rand & VI Metrics',
            'Object Detection Metrics',
            'Boundary Metrics',
            'Instance-wise Boundary Metrics',
            'Topological Metrics (ERL)',
            'Topological Metrics (Min-Cut)'
        ]

        for category in category_order:
            if category in metrics:
                values = metrics[category]
                f.write(f"--- {category} ---\n")
                for key, value in values.items():
                    if isinstance(value, dict):
                        f.write(f"  {key}:\n")
                        for sub_key, sub_value in value.items():
                            f.write(f"    {sub_key}: {sub_value:.6f}\n")
                    else:
                        f.write(f"  {key}: {value:.6f}\n")
                f.write("\n")
    print(f"\nMetrics successfully saved to {filename}")


def open_zarr_array(path, dataset):
    """
    Opens a Zarr array, trying v2 and v3 methods for compatibility.
    """
    try:
        root = zarr.open(path, mode='r')
        if dataset in root:
            print(f"Reading '{dataset}' from '{path}' using Zarr v2 method.")
            return root[dataset][...]
    except Exception:
        pass

    try:
        full_path = os.path.join(path, dataset)
        print(f"Attempting to read '{full_path}' using Zarr v3 method.")
        return zarr.open(full_path, mode='r')[...]
    except Exception as e_v3:
        print(f"Fatal: Failed to open Zarr dataset '{dataset}' at path '{path}' with both v2 and v3 methods.")
        raise IOError(f"Could not read Zarr dataset {dataset} from {path}. Error: {e_v3}")


def calculate_skeleton_lengths_with_voxel_size(
        skeletons,
        position_attribute,
        voxel_size,
        store_edge_length_attribute,
        skeleton_id_attribute):
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


def evaluate_erl(seg, gt, voxel_size):
    """
    Calculates ERL and NERL based on skeletonization of the ground truth.
    This is computationally expensive.
    """
    print("Skeletonizing ground truth for ERL calculation...")
    skeletons = kimimaro.skeletonize(
        gt.astype(np.uint32),
        anisotropy=voxel_size,
        parallel=0,
        progress=True
    )

    skeleton_graph = nx.Graph()
    node_id_offset = 0
    for label, skel in skeletons.items():
        if not skel: continue
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
        print("Warning: No skeletons found in ground truth. Skipping ERL.")
        return 0.0, 0.0

    coords = np.array([skeleton_graph.nodes[n]['zyx'] for n in nodes]).astype(np.uint64)
    for d in range(seg.ndim):
        coords[:, d] = np.clip(coords[:, d], 0, seg.shape[d] - 1)
        
    segment_ids = seg[tuple(coords.T)]
    node_segment_lut = {node: segment_id for node, segment_id in zip(nodes, segment_ids)}
    
    skeleton_lengths = calculate_skeleton_lengths_with_voxel_size(
        skeleton_graph,
        position_attribute='zyx',
        voxel_size=voxel_size,
        store_edge_length_attribute='length',
        skeleton_id_attribute='skeleton_id')

    erl = expected_run_length(
        skeletons=skeleton_graph,
        skeleton_id_attribute='skeleton_id',
        edge_length_attribute='length',
        node_segment_lut=node_segment_lut,
        skeleton_lengths=skeleton_lengths,
        return_merge_split_stats=False)
    
    total_gt_length = sum(skeleton_lengths.values())
    nerl = erl / total_gt_length if total_gt_length > 0 else 0
    
    return erl, nerl


def evaluate_min_cut(seg, gt, boundary_map):
    """
    Calculates the Min-Cut Metric.
    """
    print("Calculating Min-Cut metric...")
    rag = graph.rag_mean_color(image=boundary_map, labels=seg.astype(np.uint64))
    
    rag_nodes = [n for n in rag.nodes if n > 0]
    if not rag_nodes:
        return 0.0, 0

    coms = center_of_mass(gt, labels=seg, index=rag_nodes)
    coms = np.array(coms, dtype=int)
    
    for d in range(gt.ndim):
        coms[:, d] = np.clip(coms[:, d], 0, gt.shape[d] - 1)
    
    node_gt_labels = {
        node: gt[tuple(com)] for node, com in zip(rag_nodes, coms)
    }

    mincut_metric = 0.0
    num_splits = 0
    for u, v, data in rag.edges(data=True):
        gt_u = node_gt_labels.get(u, 0)
        gt_v = node_gt_labels.get(v, 0)

        if gt_u != 0 and gt_u == gt_v:
            mincut_metric += data.get('weight', 0)
            num_splits += 1

    return mincut_metric, num_splits


def evaluate_object_level(seg, gt, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    gt_labels = np.unique(gt)[1:]
    seg_labels = np.unique(seg)[1:]
    if len(gt_labels) == 0 or len(seg_labels) == 0:
        return 0.0, 0.0, 0.0, 0.0, []
    iou_matrix = np.zeros((len(gt_labels), len(seg_labels)))
    for i, gt_id in enumerate(tqdm(gt_labels, desc="Calculating IoU Matrix", leave=False)):
        gt_mask = (gt == gt_id)
        for j, seg_id in enumerate(seg_labels):
            seg_mask = (seg == seg_id)
            intersection = np.sum(gt_mask & seg_mask)
            union = np.sum(gt_mask | seg_mask)
            iou_matrix[i, j] = intersection / union if union > 0 else 0
    matches = []
    iou_matrix_tomod = np.copy(iou_matrix)
    for i in tqdm(range(len(gt_labels)), desc="Matching GT objects", leave=False):
        if np.sum(iou_matrix_tomod[i, :]) == 0:
            continue
        best_match_j = np.argmax(iou_matrix_tomod[i, :])
        if iou_matrix_tomod[i, best_match_j] > 0:
            matched_seg_id = seg_labels[best_match_j]
            matches.append((gt_labels[i], matched_seg_id, iou_matrix[i, best_match_j]))
            iou_matrix_tomod[:, best_match_j] = 0
    average_precisions = []
    for threshold in iou_thresholds:
        tp = sum(1 for _, _, iou in matches if iou > threshold)
        fp = len(seg_labels) - tp
        fn = len(gt_labels) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        average_precisions.append(precision)
    mAP = np.mean(average_precisions)
    tp_f1 = sum(1 for _, _, iou in matches if iou > iou_thresholds[0])
    fp_f1 = len(seg_labels) - tp_f1
    fn_f1 = len(gt_labels) - tp_f1
    precision_f1 = tp_f1 / (tp_f1 + fp_f1) if (tp_f1 + fp_f1) > 0 else 0
    recall_f1 = tp_f1 / (tp_f1 + fn_f1) if (tp_f1 + fn_f1) > 0 else 0
    f1 = 2 * (precision_f1 * recall_f1) / (precision_f1 + recall_f1) if (precision_f1 + recall_f1) > 0 else 0
    return mAP, f1, precision_f1, recall_f1, matches


def evaluate_instance_boundary_metrics(seg, gt, matches):
    if not matches:
        return 0.0, 0.0
    boundary_dices = []
    for gt_id, seg_id, iou in tqdm(matches, desc="Instance Boundary Metrics", leave=False):
        if iou > 0.5:
            gt_instance_mask = (gt == gt_id)
            seg_instance_mask = (seg == seg_id)
            gt_boundary = find_boundaries(gt_instance_mask, mode='thick')
            seg_boundary = find_boundaries(seg_instance_mask, mode='thick')
            dice = f1_score(gt_boundary.ravel(), seg_boundary.ravel())
            boundary_dices.append(dice)
    if not boundary_dices:
        return 0.0, 0.0
    return np.mean(boundary_dices), np.std(boundary_dices)


def evaluate_boundaries(gt_instances, pred_boundary_map, affs_threshold, boundary_width):
    pred_boundaries = pred_boundary_map > affs_threshold
    gt_boundaries = find_boundaries(gt_instances, mode='thick', background=0)
    dilated_gt_boundaries = binary_dilation(gt_boundaries, iterations=boundary_width)
    gt_flat = dilated_gt_boundaries.ravel()
    pred_flat = pred_boundaries.ravel()
    jaccard = jaccard_score(gt_flat, pred_flat)
    dice = f1_score(gt_flat, pred_flat)
    precision = precision_score(gt_flat, pred_flat)
    recall = recall_score(gt_flat, pred_flat)
    slice_idx = pred_boundaries.shape[0] // 2
    gt_coords = np.argwhere(dilated_gt_boundaries[slice_idx])
    pred_coords = np.argwhere(pred_boundaries[slice_idx])
    if gt_coords.size == 0 or pred_coords.size == 0:
        hausdorff = -1.0
    else:
        dist1 = directed_hausdorff(gt_coords, pred_coords)[0]
        dist2 = directed_hausdorff(pred_coords, gt_coords)[0]
        hausdorff = max(dist1, dist2)
    return jaccard, dice, precision, recall, hausdorff


def evaluate_segmentation_metrics(seg, gt):
    splits, merges = skimage.metrics.variation_of_information(gt, seg, ignore_labels=(0,))
    error, precision, recall = skimage.metrics.adapted_rand_error(gt, seg, ignore_labels=(0,))
    return splits, merges, error, precision, recall


def plot_segmentation_slices(seg, raw, aff_map, gt, boundary_width, affs_threshold):
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Segmentation and Boundary Comparison', fontsize=16)
    slice_idx = raw.shape[0] // 2
    axs[0, 0].imshow(raw[slice_idx, ...], cmap='gray')
    axs[0, 0].set_title('Raw Data')
    axs[0, 0].axis('off')
    axs[0, 1].imshow(gt[slice_idx, ...], cmap='prism')
    axs[0, 1].set_title('Ground Truth Instances')
    axs[0, 1].axis('off')
    axs[0, 2].imshow(seg[slice_idx, ...], cmap='prism')
    axs[0, 2].set_title('Final Segmentation')
    axs[0, 2].axis('off')
    pred_boundaries = aff_map[slice_idx, ...] > affs_threshold
    axs[1, 0].imshow(pred_boundaries, cmap='gray')
    axs[1, 0].set_title(f'Predicted Boundaries (Thresh > {affs_threshold})')
    axs[1, 0].axis('off')
    gt_boundaries_original = find_boundaries(gt[slice_idx, ...], mode='thick', background=0)
    axs[1, 1].imshow(gt_boundaries_original, cmap='gray')
    axs[1, 1].set_title('Original GT Boundaries')
    axs[1, 1].axis('off')
    dilated_gt_boundaries = binary_dilation(gt_boundaries_original, iterations=boundary_width)
    axs[1, 2].imshow(dilated_gt_boundaries, cmap='gray')
    axs[1, 2].set_title(f'Dilated GT Boundaries (Width: {boundary_width}px)')
    axs[1, 2].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Advanced Segmentation Evaluation Script")
    # --- REQUIRED ARGUMENTS ---
    parser.add_argument('--gt', required=True, help='Path to the ground truth Zarr file.')
    parser.add_argument('--gt_ds', required=True, help='Dataset name for GT labels.')
    parser.add_argument('--seg', required=True, help='Path to the predicted segmentation Zarr file.')
    parser.add_argument('--seg_ds', required=True, help='Dataset name for the segmentation.')
    
    # --- OPTIONAL DATA ARGUMENTS ---
    parser.add_argument('--raw', help='Path to the raw data Zarr file. If not given, assumes raw data is in the GT file.')
    parser.add_argument('--raw_ds', help='Dataset name for raw data. Required if you want to generate plots.')
    parser.add_argument('--aff', help='Path to the predicted affinities Zarr file. Required for boundary and min-cut metrics.')
    parser.add_argument('--aff_ds', default='volumes/pred_affs', help='Dataset name for affinities.')

    # --- PERFORMANCE & METRIC ARGUMENTS ---
    parser.add_argument('--no-erl', action='store_true', help='If set, skips the slow skeletonization and ERL calculation.')
    parser.add_argument('--voxel_size', nargs=3, type=float, help='Voxel side lengths in nanometers (Z Y X).')
    parser.add_argument('--aff_collapse_method', type=str, default='mean', choices=['mean', 'max', 'min', 'median'],
                        help='Method to collapse 4D affinity maps to 3D.')
    args = parser.parse_args()

    # --- Parameters ---
    AFFS_THRESHOLD = 0.5
    BOUNDARY_WIDTH = 1
    IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)
    
    # --- Voxel Size Handling ---
    VOXEL_SIZE = tuple(args.voxel_size) if args.voxel_size else (1, 1, 1)
    if not args.voxel_size:
        print("Warning: No --voxel_size provided. Assuming isotropic 1x1x1 nm voxels for ERL.")
        
    # --- Load Data ---
    print("Loading data...")
    seg = open_zarr_array(args.seg, args.seg_ds)
    gt = open_zarr_array(args.gt, args.gt_ds)
    
    raw = None
    if args.raw_ds:
        raw_path = args.raw if args.raw else args.gt
        print(f"Loading raw data from: {raw_path}")
        try:
            raw = open_zarr_array(raw_path, args.raw_ds)
        except IOError as e:
            print(f"Could not load raw data. Plotting will be skipped. Error: {e}")
    else:
        print("Warning: No --raw_ds provided. Plotting will be skipped.")

    boundary_map_3d = None
    if args.aff:
        try:
            aff = open_zarr_array(args.aff, args.aff_ds)
            if aff.ndim == 4:
                print(f"Collapsing {aff.shape[0]}-ch aff map to 3D using '{args.aff_collapse_method}'...")
                collapse_func = getattr(np, args.aff_collapse_method)
                boundary_map_3d = collapse_func(aff, axis=0)
            elif aff.ndim == 3:
                boundary_map_3d = aff
            else:
                raise ValueError(f"Unexpected affinity map dimension: {aff.ndim}")
        except IOError as e:
             print(f"Could not load affinity map. Boundary and min-cut metrics will be skipped. Error: {e}")
    else:
        print("\nWarning: No affinity map provided (--aff). Skipping boundary and min-cut metrics.")

    print("Data loaded successfully.")
    voxel_volume_nm3 = np.prod(VOXEL_SIZE)
    total_volume_um3 = (gt.size * voxel_volume_nm3) / (1000**3)
    print(f"\nVerification: GT shape {gt.shape}, voxel size {VOXEL_SIZE} nm -> Total volume {total_volume_um3:.4f} µm³.")

    all_metrics = {}

    # --- Run Evaluations ---
    print("\nCalculating metrics...")
    splits, merges, arand_err, arand_prec, arand_rec = evaluate_segmentation_metrics(seg, gt)
    all_metrics['Rand & VI Metrics'] = {
        'Adapted Rand Error': arand_err, 'Adapted Rand Precision': arand_prec, 'Adapted Rand Recall': arand_rec,
        'Variation of Information (Splits)': splits, 'Variation of Information (Merges)': merges
    }

    mAP, f1, obj_prec, obj_rec, matches = evaluate_object_level(seg, gt, IOU_THRESHOLDS)
    all_metrics['Object Detection Metrics'] = {
        'Mean Average Precision (mAP)': mAP, f'Object-level F1 Score (IoU > {IOU_THRESHOLDS[0]})': f1,
        f'Object-level Precision (IoU > {IOU_THRESHOLDS[0]})': obj_prec, f'Object-level Recall (IoU > {IOU_THRESHOLDS[0]})': obj_rec
    }
    
    if not args.no_erl:
        erl, nerl = evaluate_erl(seg, gt, VOXEL_SIZE)
        all_metrics['Topological Metrics (ERL)'] = {
            'Expected Run Length (ERL, nm)': erl, 'Normalized ERL (NERL, fraction)': nerl
        }

    if boundary_map_3d is not None:
        b_jaccard, b_dice, b_prec, b_recall, hausdorff = evaluate_boundaries(gt, boundary_map_3d, AFFS_THRESHOLD, BOUNDARY_WIDTH)
        all_metrics['Boundary Metrics'] = {
            'Jaccard (IoU)': b_jaccard, 'Dice Coefficient': b_dice, 'Precision': b_prec, 'Recall': b_recall,
            f'Hausdorff Distance (slice, width={BOUNDARY_WIDTH}px)': hausdorff
        }
        mincut, num_splits = evaluate_min_cut(seg, gt, boundary_map_3d)
        all_metrics['Topological Metrics (Min-Cut)'] = {
            'Min-Cut Metric': mincut, 'Number of Splits (Absolute)': num_splits,
            'Splits per um^3': num_splits / total_volume_um3 if total_volume_um3 > 0 else 0
        }

    mean_ibd, std_ibd = evaluate_instance_boundary_metrics(seg, gt, matches)
    all_metrics['Instance-wise Boundary Metrics'] = {
        'Mean Boundary Dice': mean_ibd, 'Std Dev Boundary Dice': std_ibd
    }
    
    print("Metrics calculated.")

    # --- Print & Save ---
    print("\n--- EVALUATION RESULTS ---")
    for category in all_metrics:
        print(f"\n--- {category} ---\n")
        for key, value in all_metrics[category].items():
            print(f"  {key}: {value:.6f}")
    
    filename_prompt = input("\nEnter a base name for the results text file (e.g., 'my_experiment'): ")
    if filename_prompt:
        unique_filename = get_unique_filename(filename_prompt)
        save_metrics_to_file(all_metrics, unique_filename)

    if boundary_map_3d is not None and raw is not None:
        print("\nGenerating visualization...")
        plot_segmentation_slices(seg, raw, boundary_map_3d, gt, BOUNDARY_WIDTH, AFFS_THRESHOLD)
    else:
        print("\nSkipping visualization as raw data or affinity map was not provided.")

if __name__ == '__main__':
    main()
