"""
Runs evaluation between ground-truth and predicted semantic masks and instances from engine.utils.metrics.py.

Run like:
python eval.py --gt_instance_path /media/samia/DATA/mounts/fibserver1/smohinta_data/catena_data/HEMI_MITO/data_3d/test/hemi_x11051_y17920_z21948_clahe.zarr
--semantic_pred_path /media/samia/DATA/PhD/codebases/MitoEM/inference_results/merged_predictions_hemi_test.tif
--instance_pred_path /media/samia/DATA/PhD/codebases/MitoEM/repaired_instance_results/repaired_semantic_to_instance_output.zarr
"""

import sys
import os
import argparse
import numpy as np
import torch
import zarr
import tifffile as tff
from tqdm import tqdm
from skimage import measure
from typing import List, Tuple, Dict, Any

# Add the project root to sys.path to import from engine package
# Assumes eval_script.py is at the root level alongside 'engine'
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import metrics from your engine.utils.metrics
from engine.utils.metrics import (
    DiceCoefficient,
    MeanIoU, MeanIoUBinary,
    BlobsAveragePrecision,
)

# Import utility for removing small instances
from engine.utils.utils import remove_small_instances


# --- Helper function for loading volumes ---
def load_volume(path: str, key: str = None):
    """
    Loads a volume from Zarr or TIFF, returning a Zarr array object or a NumPy array.
    For Zarr, 'key' specifies the internal dataset path.
    """
    if path.endswith(".zarr"):
        try:
            zarr_group = zarr.open(path, mode='r')
            if key:
                # If a key is provided, return the specific dataset
                return zarr_group[key]
            else:
                # If no key, assume a single array at the top level
                # This is less common; a key is usually needed.
                return zarr_group
        except KeyError as e:
            raise KeyError(f"Dataset key '{key}' not found in Zarr file '{path}'. Original Error: {e}")
    elif path.endswith((".tif", ".tiff")):
        return tff.imread(path)
    else:
        raise ValueError(f"Unsupported file format for {path}. Must be .zarr or .tif/.tiff")


# --- Helper for converting instance GT to semantic GT ---
def instance_to_semantic(instance_volume: np.ndarray) -> np.ndarray:
    """Converts an instance segmentation volume (e.g., uint64) to a binary semantic mask (0 or 1)."""
    return (instance_volume != 0).astype(np.uint8)


# --- Evaluation Function ---
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Ground Truth (GT) Instance Segmentation ---
    print(f"\nLoading Ground Truth from: {args.gt_instance_path}")
    gt_instance_volume_reader = load_volume(args.gt_instance_path, key=args.gt_label_key)
    gt_volume_shape = gt_instance_volume_reader.shape

    # --- Initialize Metrics ---
    semantic_iou_metric = MeanIoUBinary(n_classes=2, skip_channels=(0,))
    semantic_dice_metric = DiceCoefficient()
    instance_ap_metric = BlobsAveragePrecision(
        thresholds=args.instance_ap_thresholds,
        min_instance_size=args.min_instance_size,
        metric='ap'
    )
    # Accumulators for results from each chunk
    all_semantic_iou = []
    all_semantic_dice = []
    all_instance_ap = []

    # --- Determine Evaluation Modes ---
    run_semantic_eval = args.semantic_pred_path is not None
    run_instance_eval = args.instance_pred_path is not None

    if not run_semantic_eval and not run_instance_eval:
        raise ValueError("No prediction paths provided. Please specify --semantic_pred_path or --instance_pred_path.")

    print("\nStarting Evaluation...")

    # Iterate through the volume in chunks
    chunk_size = args.eval_chunk_size
    overlap = args.eval_overlap

    step_z = chunk_size[0] - overlap[0]
    step_y = chunk_size[1] - overlap[1]
    step_x = chunk_size[2] - overlap[2]

    z_starts = list(range(0, gt_volume_shape[0], step_z))
    y_starts = list(range(0, gt_volume_shape[1], step_y))
    x_starts = list(range(0, gt_volume_shape[2], step_x))

    if (gt_volume_shape[0] - chunk_size[0]) not in z_starts and gt_volume_shape[0] > chunk_size[0]:
        z_starts.append(gt_volume_shape[0] - chunk_size[0])
    if (gt_volume_shape[1] - chunk_size[1]) not in y_starts and gt_volume_shape[1] > chunk_size[1]:
        y_starts.append(gt_volume_shape[1] - chunk_size[1])
    if (gt_volume_shape[2] - chunk_size[2]) not in x_starts and gt_volume_shape[2] > chunk_size[2]:
        x_starts.append(gt_volume_shape[2] - chunk_size[2])

    z_starts = sorted(list(set(z_starts)))
    y_starts = sorted(list(set(y_starts)))
    x_starts = sorted(list(set(x_starts)))

    num_chunks = len(z_starts) * len(y_starts) * len(x_starts)

    for z_start in tqdm(z_starts, desc="Processing Chunks"):
        z_end = min(z_start + chunk_size[0], gt_volume_shape[0])
        for y_start in y_starts:
            y_end = min(y_start + chunk_size[1], gt_volume_shape[1])
            for x_start in x_starts:
                x_end = min(x_start + chunk_size[2], gt_volume_shape[2])

                chunk_slice_global = (slice(z_start, z_end), slice(y_start, y_end), slice(x_start, x_end))

                # Load GT chunk (instance labels)
                gt_instance_chunk = gt_instance_volume_reader[chunk_slice_global]

                # Convert GT instance chunk to semantic for semantic evaluation
                gt_semantic_chunk = instance_to_semantic(gt_instance_chunk)

                # Convert to PyTorch tensors
                gt_semantic_torch = torch.from_numpy(gt_semantic_chunk).unsqueeze(0).unsqueeze(0).to(device)
                gt_instance_chunk = gt_instance_chunk.astype(np.int64)
                gt_instance_torch = torch.from_numpy(gt_instance_chunk).unsqueeze(0).unsqueeze(0).to(device)

                # --- Semantic Evaluation ---
                if run_semantic_eval:
                    semantic_pred_volume_reader = load_volume(args.semantic_pred_path, key=args.semantic_pred_key)
                    semantic_pred_chunk = semantic_pred_volume_reader[chunk_slice_global]

                    if semantic_pred_chunk.dtype == np.uint8 and np.max(semantic_pred_chunk) > 1:
                        semantic_pred_chunk = semantic_pred_chunk.astype(np.float32) / 255.0
                    elif semantic_pred_chunk.dtype != np.float32 and semantic_pred_chunk.dtype != np.float64:
                        semantic_pred_chunk = semantic_pred_chunk.astype(np.float32)

                    semantic_pred_torch = torch.from_numpy(semantic_pred_chunk).unsqueeze(0).unsqueeze(0).to(device)

                    # Calculate and store semantic metrics for this chunk
                    # CRITICAL CHANGE: Call the metric object and append the result
                    # MeanIoU Calls for Single-Channel Binary Input ---

                    # 1. Get the IoU for the foreground (where value is 1)
                    # Binarize the prediction by thresholding at 0.5 and convert to long tensor
                    pred_foreground = (semantic_pred_torch > 0.5).long().squeeze(1)

                    # Get the ground truth foreground (where value is 1)
                    gt_foreground = (gt_semantic_torch == 1).long().squeeze(1)

                    foreground_iou = semantic_iou_metric._jaccard_index(pred_foreground, gt_foreground)

                    # 2. Get the IoU for the background (where value is 0)
                    # Get the prediction background (where value is 0)
                    pred_background = (semantic_pred_torch <= 0.5).long().squeeze(1)

                    # Get the ground truth background (where value is 0)
                    gt_background = (gt_semantic_torch == 0).long().squeeze(1)

                    background_iou = semantic_iou_metric._jaccard_index(pred_background, gt_background)

                    # 3. Average the two IoU scores for the mean IoU
                    mean_iou_score_chunk = (foreground_iou + background_iou) / 2.0

                    # Append the final score to the list
                    all_semantic_iou.append(mean_iou_score_chunk.item())

                    # Calc the DICE coefficient
                    all_semantic_dice.append(semantic_dice_metric(semantic_pred_torch, gt_semantic_torch).item())

                # --- Instance Evaluation ---
                if run_instance_eval:
                    instance_pred_volume_reader = load_volume(args.instance_pred_path, key=args.instance_pred_key)
                    instance_pred_chunk = instance_pred_volume_reader[chunk_slice_global]

                    if instance_pred_chunk.dtype == np.uint8 and np.max(instance_pred_chunk) > 1:
                        instance_pred_chunk = instance_pred_chunk.astype(np.float32) / 255.0
                    elif instance_pred_chunk.dtype != np.float32 and instance_pred_chunk.dtype != np.float64:
                        instance_pred_chunk = instance_pred_chunk.astype(np.float32)

                    instance_pred_np = instance_pred_chunk[np.newaxis, :, :, :]
                    gt_instance_np = gt_instance_chunk

                    # CRITICAL CHANGE: Call the metric object and append the result
                    all_instance_ap.append(instance_ap_metric(instance_pred_np, gt_instance_np).item())


    # --- Final Results ---
    print("\n--- Evaluation Results ---")

    if run_semantic_eval:
        print("\nSemantic Segmentation Metrics:")
        # CRITICAL CHANGE: Calculate the mean of the list of values
        print(f"  Mean IoU: {np.mean(all_semantic_iou):.4f}")
        print(f"  Dice Coefficient: {np.mean(all_semantic_dice):.4f}")

    if run_instance_eval:
        print("\nInstance Segmentation Metrics:")
        # CRITICAL CHANGE: Calculate the mean of the list of values
        print(f"  Blobs Average Precision (AP): {np.mean(all_instance_ap):.4f}")

    print("\nEvaluation complete.")


# --- Command-line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Segmentation Predictions")

    parser.add_argument('--gt_instance_path', type=str, required=True,
                        help='Path to the ground truth instance segmentation volume (.zarr or .tif).')
    parser.add_argument('--gt_label_key', type=str, default='volumes/labels/neuron_ids',
                        help='Internal Zarr key for the ground truth label dataset. E.g., "volumes/labels/mito_ids".')
    parser.add_argument('--semantic_pred_path', type=str, default=None,
                        help='Path to the semantic prediction volume (.zarr or .tif).')
    parser.add_argument('--semantic_pred_key', type=str, default=None,
                        help='Internal Zarr key for the ground truth label dataset. E.g., "seg".')
    parser.add_argument('--instance_pred_path', type=str, default=None,
                        help='Path to the instance prediction volume (.zarr or .tif). '
                             'NOTE: For BlobsAveragePrecision, this should be the semantic probability map '
                             'from which instances are derived.')
    parser.add_argument('--instance_pred_key', type=str, default='seg',
                        help='Internal Zarr key for the ground truth label dataset. E.g., "seg".')

    # Chunking parameters for evaluation (must fit memory for each chunk)
    parser.add_argument('--eval_chunk_size', type=int, nargs=3, default=[128, 128, 128],
                        help='Chunk size (Z Y X) for processing volumes during evaluation.')
    parser.add_argument('--eval_overlap', type=int, nargs=3, default=[16, 16, 16],
                        help='Overlap size (Z Y X) between chunks during evaluation.')

    # Parameters for Instance AP metric
    parser.add_argument('--instance_ap_thresholds', type=float, nargs='+', default=[0.5, 0.75, 0.9],
                        help='IoU thresholds for BlobsAveragePrecision calculation.')
    parser.add_argument('--min_instance_size', type=int, default=128,
                        help='Minimum instance size (pixels) for BlobsAveragePrecision.')

    args = parser.parse_args()
    evaluate(args)
