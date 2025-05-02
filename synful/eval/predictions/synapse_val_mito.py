"""
Objectives:
1. First assign all pre-post detected points to their neuron segmentation.

2. Assign all segmentation mitochondria to their corresponding neuron segmentation

3. Check for duplicate pre-post detections for the neuron pairs. For example, if we see that a synapse pair has been detected twice for the same neuron pair, we should agglomerate those into one detection.

4. Find the distances of the pre-syn sites from its nearest mitochondria.

5. Suppress all synapse detections that lie within the same neuron. Synapses must be between neurons and the positional detections should not be on the same neuron.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Set, Optional
import numpy.typing as npt
from scipy.ndimage import binary_dilation, distance_transform_edt
import os
import argparse
import json
import zarr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def assign_synapses_to_neurons(
        pre_positions: npt.NDArray,
        post_positions: npt.NDArray,
        neuron_segmentation: npt.NDArray,
        resolution: npt.NDArray
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Assign pre and post-synaptic sites to their corresponding neurons.
    
    Args:
        pre_positions: Nx3 array of pre-synaptic positions (in nm)
        post_positions: Nx3 array of post-synaptic positions (in nm)
        neuron_segmentation: 3D array of neuron labels
        resolution: 3-element array of voxel sizes in nm
    
    Returns:
        pre_assignments: Dict mapping synapse indices to neuron IDs for pre-synaptic sites
        post_assignments: Dict mapping synapse indices to neuron IDs for post-synaptic sites
    """
    pre_assignments = {}
    post_assignments = {}

    # Convert positions from nm to voxel coordinates
    # Assuming pre_positions are in XYZ order, we need to flip them to ZYX
    pre_positions_zyx = pre_positions.copy()
    post_positions_zyx = post_positions.copy()

    # if pre_positions.shape[1] == 3:
    #     # Flip XYZ to ZYX if needed
    #     pre_positions_zyx = pre_positions[:, ::-1]
    #     post_positions_zyx = post_positions[:, ::-1]
    #     print("Converted synapse positions from XYZ to ZYX order")
    
    pre_voxels = np.round(pre_positions_zyx / resolution).astype(int)
    post_voxels = np.round(post_positions_zyx / resolution).astype(int)

    # Ensure coordinates are within bounds
    for i, pos in enumerate(pre_voxels):
        if all(0 <= p < s for p, s in zip(pos, neuron_segmentation.shape)):
            neuron_id = neuron_segmentation[pos[0], pos[1], pos[2]]
            if neuron_id > 0:  # Ignore background (usually label 0)
                pre_assignments[i] = neuron_id

    for i, pos in enumerate(post_voxels):
        if all(0 <= p < s for p, s in zip(pos, neuron_segmentation.shape)):
            neuron_id = neuron_segmentation[pos[0], pos[1], pos[2]]
            if neuron_id > 0:
                post_assignments[i] = neuron_id

    return pre_assignments, post_assignments


def assign_mitochondria_to_neurons(
        mito_segmentation: npt.NDArray,
        neuron_segmentation: npt.NDArray
) -> Tuple[Dict[int, int], List[int], List[int]]:
    """
    Assign mitochondria to their corresponding neurons.
    
    Args:
        mito_segmentation: 3D array containing mitochondria labels
        neuron_segmentation: 3D array containing neuron labels
    
    Returns:
        mito_to_neuron: Dict mapping mitochondria IDs to neuron IDs
        mito_ids: List of identified mitochondria IDs
        neuron_ids: List of identified neuron IDs
    """
    # Convert Zarr arrays to numpy arrays for processing
    if hasattr(mito_segmentation, 'is_zarr_array') or str(type(mito_segmentation)).find('zarr') != -1:
        print("Converting mito segmentation from Zarr to NumPy array...")
        mito_segmentation = np.array(mito_segmentation[:])
    
    if hasattr(neuron_segmentation, 'is_zarr_array') or str(type(neuron_segmentation)).find('zarr') != -1:
        print("Converting neuron segmentation from Zarr to NumPy array...")
        neuron_segmentation = np.array(neuron_segmentation[:])
    
    # Get all unique mitochondria IDs (excluding background)
    mito_ids = np.unique(mito_segmentation)
    mito_ids = mito_ids[mito_ids > 0]
    
    # Get all unique neuron IDs (excluding background)
    neuron_ids = np.unique(neuron_segmentation)
    neuron_ids = neuron_ids[neuron_ids > 0]
    
    print(f"Found {len(mito_ids)} unique mitochondria IDs")
    print(f"Found {len(neuron_ids)} unique neuron IDs")
    
    # Now assign each mitochondrion to its containing neuron
    mito_to_neuron = {}
    
    for mito_id in tqdm(mito_ids, desc="Assigning mitochondria to neurons", total=len(mito_ids)):
        mito_mask = mito_segmentation == mito_id
        
        # Find overlapping neuron IDs
        overlapping_neurons = neuron_segmentation[mito_mask]
        
        # Count occurrences of each neuron ID
        neuron_counts = {}
        for nid in np.unique(overlapping_neurons):
            if nid > 0:  # Skip background
                count = np.sum(overlapping_neurons == nid)
                neuron_counts[nid] = count
        
        # Assign to neuron with maximum overlap
        if neuron_counts:
            best_neuron = max(neuron_counts.items(), key=lambda x: x[1])[0]
            mito_to_neuron[mito_id] = best_neuron
    
    return mito_to_neuron, mito_ids.tolist(), neuron_ids.tolist()


def find_duplicate_synapses(
        pre_positions: npt.NDArray,
        post_positions: npt.NDArray,
        pre_assignments: Dict[int, int],
        post_assignments: Dict[int, int],
        distance_threshold: float = 500.0  # nm
) -> List[Set[int]]:
    """
    Find groups of duplicate synapse detections between the same neuron pairs.
    
    Args:
        pre_positions: Nx3 array of pre-synaptic positions
        post_positions: Nx3 array of post-synaptic positions
        pre_assignments: Dict mapping synapse indices to neuron IDs for pre-synaptic sites
        post_assignments: Dict mapping synapse indices to neuron IDs for post-synaptic sites
        distance_threshold: Maximum distance (nm) to consider synapses as duplicates
    
    Returns:
        List of sets containing indices of duplicate synapses
    """
    duplicate_groups = []
    processed = set()

    # Create a KD-tree for efficient spatial searching
    positions = np.concatenate([pre_positions, post_positions], axis=1)  # Nx6
    tree = cKDTree(positions)

    for i in range(len(pre_positions)):
        if i in processed:
            continue

        if i not in pre_assignments or i not in post_assignments:
            continue

        # Find all points within distance threshold
        nearby_indices = tree.query_ball_point(positions[i], distance_threshold)

        # Check which nearby points are between the same neurons
        duplicates = {i}
        for j in nearby_indices:
            if (j != i and j not in processed and
                    j in pre_assignments and j in post_assignments and
                    pre_assignments[i] == pre_assignments[j] and
                    post_assignments[i] == post_assignments[j]):
                duplicates.add(j)

        if len(duplicates) > 1:
            duplicate_groups.append(duplicates)
            processed.update(duplicates)

    return duplicate_groups


def calculate_distances_to_mitochondria(
        pre_positions: npt.NDArray,
        mito_segmentation: npt.NDArray,
        neuron_segmentation: npt.NDArray,
        pre_assignments: Dict[int, int],
        resolution: npt.NDArray,
        max_search_radius: float = 1000.0  # nm
) -> Dict[int, float]:
    """
    Calculate distances from pre-synaptic sites to their nearest mitochondria within the same neuron.
    
    Args:
        pre_positions: Nx3 array of pre-synaptic positions (in nm)
        mito_segmentation: 3D array of mitochondria labels
        neuron_segmentation: 3D array of neuron labels
        pre_assignments: Dict mapping synapse indices to neuron IDs for pre-synaptic sites
        resolution: 3-element array of voxel sizes in nm
        max_search_radius: Maximum search radius in nm
    
    Returns:
        Dict mapping synapse indices to distances (nm) to nearest mitochondria within the same neuron
    """
    # Convert Zarr arrays to numpy arrays for processing
    if hasattr(mito_segmentation, 'is_zarr_array') or str(type(mito_segmentation)).find('zarr') != -1:
        print("Converting mito segmentation from Zarr to NumPy array...")
        mito_segmentation = np.array(mito_segmentation[:])

    if hasattr(neuron_segmentation, 'is_zarr_array') or str(type(neuron_segmentation)).find('zarr') != -1:
        print("Converting neuron segmentation from Zarr to NumPy array...")
        neuron_segmentation = np.array(neuron_segmentation[:])

    # Check unique neuron IDs
    unique_neuron_ids = np.unique(neuron_segmentation)
    print(f"Len of Unique neuron IDs in segmentation: {len(unique_neuron_ids)}")

    if len(unique_neuron_ids) == 0:
        raise Exception("No neurons found in segmentation.")

    # Check unique mitochondria IDs
    unique_mito_ids = np.unique(mito_segmentation)
    print(f"Len of Unique mitochondria IDs in segmentation: {len(unique_mito_ids)}")

    if len(unique_mito_ids) == 0:
        raise Exception("No mitochondria found in segmentation.")

    # First, assign mitochondria to neurons
    mito_to_neuron, mito_ids, _ = assign_mitochondria_to_neurons(mito_segmentation, neuron_segmentation)

    # Group mitochondria by neuron ID
    neuron_to_mitos = {}
    for mito_id, neuron_id in mito_to_neuron.items():
        if neuron_id not in neuron_to_mitos:
            neuron_to_mitos[neuron_id] = []
        neuron_to_mitos[neuron_id].append(mito_id)

    # Find coordinates of all mitochondria in voxel space
    mito_coords = {}
    for mito_id in tqdm(mito_ids, desc="Finding mitochondria coordinates", total=len(mito_ids)):
        # Find the center of mass of the mitochondrion
        mito_mask = mito_segmentation == mito_id
        if np.any(mito_mask):
            coords = np.where(mito_mask)
            # Note: coords are in ZYX order from np.where
            # Create center in ZYX order to match segmentation
            center_zyx = np.array([np.mean(coords[0]), np.mean(coords[1]), np.mean(coords[2])])
            # Convert from voxel coordinates to nm
            center_nm_zyx = center_zyx * resolution
            mito_coords[mito_id] = center_nm_zyx

    # Calculate distances for each pre-synaptic site
    distances = {}

    # Convert pre-positions to ZYX order if they're not already
    # Assuming pre_positions are in XYZ order, we need to flip them to ZYX
    pre_positions_zyx = pre_positions.copy()
    # if pre_positions.shape[1] == 3:
    #     # Flip XYZ to ZYX if needed
    #     pre_positions_zyx = pre_positions[:, ::-1]
    #     print("Converted pre-synaptic positions from XYZ to ZYX order")

    for i, pos in enumerate(pre_positions_zyx):
        # Skip if the pre-synaptic site is not assigned to a neuron
        if i not in pre_assignments:
            continue

        neuron_id = pre_assignments[i]

        # Skip if no mitochondria in this neuron
        if neuron_id not in neuron_to_mitos or not neuron_to_mitos[neuron_id]:
            continue

        # Find distances to all mitochondria in the same neuron
        mito_distances = []
        for mito_id in neuron_to_mitos[neuron_id]:
            if mito_id in mito_coords:
                mito_pos = mito_coords[mito_id]
                dist = np.linalg.norm(pos - mito_pos)
                if dist <= max_search_radius:
                    mito_distances.append(dist)

        # Record the minimum distance if any mitochondria were found
        if mito_distances:
            distances[i] = min(mito_distances)

    return distances

def save_results_to_json(
        out_dir: str,
        gt_pre_positions: npt.NDArray,
        gt_post_positions: npt.NDArray,
        gt_pre_assignments: Dict[int, int],
        gt_post_assignments: Dict[int, int],
        gt_mito_distances: Dict[int, float],
        gt_to_suppress: List[int],
        pred_pre_positions: Optional[npt.NDArray] = None,
        pred_post_positions: Optional[npt.NDArray] = None,
        pred_pre_assignments: Optional[Dict[int, int]] = None,
        pred_post_assignments: Optional[Dict[int, int]] = None,
        pred_mito_distances: Optional[Dict[int, float]] = None,
        pred_to_suppress: Optional[List[int]] = None,
        dataset_name: str = "unknown"
) -> str:
    """
    Save analysis results to a JSON file.
    
    Args:
        out_dir: Output directory
        gt_pre_positions: Ground truth pre-synaptic positions
        gt_post_positions: Ground truth post-synaptic positions
        gt_pre_assignments: Ground truth pre-synaptic neuron assignments
        gt_post_assignments: Ground truth post-synaptic neuron assignments
        gt_mito_distances: Ground truth distances to mitochondria
        gt_to_suppress: Ground truth synapses to suppress (same neuron)
        pred_pre_positions: Predicted pre-synaptic positions
        pred_post_positions: Predicted post-synaptic positions
        pred_pre_assignments: Predicted pre-synaptic neuron assignments
        pred_post_assignments: Predicted post-synaptic neuron assignments
        pred_mito_distances: Predicted distances to mitochondria
        pred_to_suppress: Predicted synapses to suppress (same neuron)
        dataset_name: Name of the dataset
    
    Returns:
        Path to the saved JSON file
    """
    results = {
        "dataset_name": dataset_name,
        "ground_truth": {
            "synapses": [],
            "stats": {
                "total_synapses": len(gt_pre_positions),
                "assigned_pre": len(gt_pre_assignments),
                "assigned_post": len(gt_post_assignments),
                "with_mito_distance": len(gt_mito_distances),
                "same_neuron_synapses": len(gt_to_suppress)
            }
        }
    }
    
    # Add ground truth synapse data
    for i in range(len(gt_pre_positions)):
        synapse = {
            "id": i,
            "pre_position": gt_pre_positions[i].tolist(),
            "post_position": gt_post_positions[i].tolist(),
            "pre_neuron_id": gt_pre_assignments.get(i),
            "post_neuron_id": gt_post_assignments.get(i),
            "mito_distance": gt_mito_distances.get(i),
            "same_neuron": i in gt_to_suppress
        }
        results["ground_truth"]["synapses"].append(synapse)
    
    # Add predicted data if available
    if pred_pre_positions is not None:
        results["prediction"] = {
            "synapses": [],
            "stats": {
                "total_synapses": len(pred_pre_positions),
                "assigned_pre": len(pred_pre_assignments) if pred_pre_assignments else 0,
                "assigned_post": len(pred_post_assignments) if pred_post_assignments else 0,
                "with_mito_distance": len(pred_mito_distances) if pred_mito_distances else 0,
                "same_neuron_synapses": len(pred_to_suppress) if pred_to_suppress else 0
            }
        }
        
        for i in range(len(pred_pre_positions)):
            synapse = {
                "id": i,
                "pre_position": pred_pre_positions[i].tolist(),
                "post_position": pred_post_positions[i].tolist(),
                "pre_neuron_id": pred_pre_assignments.get(i) if pred_pre_assignments else None,
                "post_neuron_id": pred_post_assignments.get(i) if pred_post_assignments else None,
                "mito_distance": pred_mito_distances.get(i) if pred_mito_distances else None,
                "same_neuron": i in pred_to_suppress if pred_to_suppress else False
            }
            results["prediction"]["synapses"].append(synapse)
    
    # Save to JSON file
    json_path = os.path.join(out_dir, f"synapse_analysis_{dataset_name}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results to {json_path}")
    return json_path

    
def visualize_synapses(
        pre_positions: npt.NDArray,
        post_positions: npt.NDArray,
        out_dir: str,
        prefix: str = "synapse",
        to_suppress: Optional[List[int]] = None,
        mito_distances: Optional[Dict[int, float]] = None,
        max_synapses: int = 20
):
    """
    Visualize synapses with pre and post-synaptic sites and arrows.
    
    Args:
        pre_positions: Pre-synaptic positions
        post_positions: Post-synaptic positions
        out_dir: Output directory
        prefix: Prefix for output files
        to_suppress: List of synapse indices to suppress (same neuron)
        mito_distances: Dict mapping synapse indices to distances to mitochondria
        max_synapses: Maximum number of synapses to visualize
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create output directory for visualizations
    vis_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Select a subset of synapses to visualize
    if to_suppress is None:
        to_suppress = []
    
    # Prioritize synapses with mitochondria distances if available
    if mito_distances:
        indices = list(mito_distances.keys())
        # Sort by distance (closest first)
        indices.sort(key=lambda i: mito_distances[i])
        # Take the first max_synapses/2 and last max_synapses/2
        indices = indices[:max_synapses//2] + indices[-max_synapses//2:]
    else:
        # Otherwise, just take the first max_synapses
        indices = list(range(min(max_synapses, len(pre_positions))))
    
    # Add some same-neuron synapses if available
    same_neuron_indices = [i for i in to_suppress if i < len(pre_positions)]
    if same_neuron_indices:
        # Add up to 5 same-neuron synapses
        indices.extend(same_neuron_indices[:5])
    
    # Make indices unique
    indices = list(set(indices))
    
    # Create a figure for each axis view
    views = [
        ('xy', 0, 1, 'XY Plane (Top View)'),
        ('xz', 0, 2, 'XZ Plane (Front View)'),
        ('yz', 1, 2, 'YZ Plane (Side View)'),
        ('3d', None, None, '3D View')
    ]
    
    for view_name, axis1, axis2, title in views:
        plt.figure(figsize=(12, 10))
        
        if view_name == '3d':
            ax = plt.subplot(111, projection='3d')
            
            # Plot each synapse
            for i in indices:
                pre = pre_positions[i]
                post = post_positions[i]
                
                # Determine color based on whether it's a same-neuron synapse
                color = 'red' if i in to_suppress else 'blue'
                
                # Plot pre-synaptic site (red)
                ax.scatter(pre[0], pre[1], pre[2], color='red', s=100, label='Pre-synaptic' if i == indices[0] else "")
                
                # Plot post-synaptic site (green)
                ax.scatter(post[0], post[1], post[2], color='green', s=100, label='Post-synaptic' if i == indices[0] else "")
                
                # Draw arrow from pre to post (orange)
                ax.quiver(pre[0], pre[1], pre[2], 
                         post[0]-pre[0], post[1]-pre[1], post[2]-pre[2],
                         color='orange', arrow_length_ratio=0.1, label='Connection' if i == indices[0] else "")
                
                # Add text label with synapse ID and mito distance if available
                label = f"ID: {i}"
                if mito_distances and i in mito_distances:
                    label += f"\nMito: {mito_distances[i]:.1f}nm"
                if i in to_suppress:
                    label += "\nSame neuron"
                
                ax.text(pre[0], pre[1], pre[2], label, fontsize=8)
            
            ax.set_xlabel('X (nm)')
            ax.set_ylabel('Y (nm)')
            ax.set_zlabel('Z (nm)')
            ax.set_title(f"{title} - {prefix}")
            
            # Add legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            
        else:
            ax = plt.subplot(111)
            
            # Plot each synapse
            for i in indices:
                pre = pre_positions[i]
                post = post_positions[i]
                
                # Determine color based on whether it's a same-neuron synapse
                color = 'red' if i in to_suppress else 'blue'
                
                # Plot pre-synaptic site (red)
                ax.scatter(pre[axis1], pre[axis2], color='red', s=100, label='Pre-synaptic' if i == indices[0] else "")
                
                # Plot post-synaptic site (green)
                ax.scatter(post[axis1], post[axis2], color='green', s=100, label='Post-synaptic' if i == indices[0] else "")
                
                # Draw arrow from pre to post (orange)
                ax.arrow(pre[axis1], pre[axis2], 
                        post[axis1]-pre[axis1], post[axis2]-pre[axis2],
                        color='orange', width=5, head_width=20, head_length=20, 
                        length_includes_head=True, label='Connection' if i == indices[0] else "")
                
                # Add text label with synapse ID and mito distance if available
                label = f"ID: {i}"
                if mito_distances and i in mito_distances:
                    label += f"\nMito: {mito_distances[i]:.1f}nm"
                if i in to_suppress:
                    label += "\nSame neuron"
                
                ax.text(pre[axis1], pre[axis2], label, fontsize=8)
            
            ax.set_xlabel(f"{'X' if axis1 == 0 else 'Y'} (nm)")
            ax.set_ylabel(f"{'Y' if axis2 == 1 else 'Z'} (nm)")
            ax.set_title(f"{title} - {prefix}")
            
            # Add legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"{prefix}_{view_name}.png"), dpi=300)
        plt.close()
    
    print(f"Saved visualizations to {vis_dir}")


def filter_same_neuron_synapses(
        pre_assignments: Dict[int, int],
        post_assignments: Dict[int, int]
) -> List[int]:
    """
    Find synapses where pre and post-synaptic sites are on the same neuron.
    
    Args:
        pre_assignments: Dict mapping synapse indices to neuron IDs for pre-synaptic sites
        post_assignments: Dict mapping synapse indices to neuron IDs for post-synaptic sites
    
    Returns:
        List of synapse indices to suppress
    """
    to_suppress = []

    for idx in pre_assignments:
        if idx in post_assignments:
            if pre_assignments[idx] == post_assignments[idx]:
                to_suppress.append(idx)

    return to_suppress


def get_coordinate_columns(df, prefix):
    """
    Identify coordinate columns in a dataframe.

    Args:
        df: Pandas DataFrame
        prefix: Prefix for column names (e.g., 'Pre' or 'Post')

    Returns:
        cols: List of coordinate column names
        id_col: Name of ID column if found, otherwise None
    """
    # Try standard naming conventions
    if prefix + '_X' in df.columns and prefix + '_Y' in df.columns and prefix + '_Z' in df.columns:
        cols = [prefix + '_X', prefix + '_Y', prefix + '_Z']
    # Try alternative naming (axis-based)
    elif 'axis-0' in df.columns and 'axis-1' in df.columns and 'axis-2' in df.columns:
        cols = ['axis-0', 'axis-1', 'axis-2']
    # Try x, y, z naming
    elif 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
        cols = ['x', 'y', 'z']
    else:
        # Look for any columns with X, Y, Z in their names
        x_cols = [col for col in df.columns if 'x' in col.lower()]
        y_cols = [col for col in df.columns if 'y' in col.lower()]
        z_cols = [col for col in df.columns if 'z' in col.lower()]

        if x_cols and y_cols and z_cols:
            cols = [x_cols[0], y_cols[0], z_cols[0]]
        else:
            raise ValueError(f"Cannot identify coordinate columns in dataframe with columns: {df.columns}")

    # Look for ID column
    id_col = None
    id_candidates = [prefix + '_ID', prefix + 'ID', prefix.lower() + '_id', prefix.lower() + 'id', 'id', 'ID']
    for candidate in id_candidates:
        if candidate in df.columns:
            id_col = candidate
            break

    return cols, id_col


def main():
    """Run GT-Only:
    python /Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Synapse_localisation/synapse_curation/synapse_val_mito.py \
     --gt-pre /Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/COMBINED_NEURIPS_SAME_PREID/combined_same_preid_nips_gt_test/hemi_synapses_x15035-15635_y28559-29159_z9602-10202_gt_pre_locations.csv \
    --gt-post /Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/COMBINED_NEURIPS_SAME_PREID/combined_same_preid_nips_gt_test/hemi_synapses_x15035-15635_y28559-29159_z9602-10202_gt_post_locations.csv \
    --pred-pre /Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/COMBINED_NEURIPS_SAME_PREID/dani_synapse_mapping_results/hemi/HEMIBRAIN_synapses_x15035-15635_y28559-29159_z9602-10202_pred_pre_locations.csv \
    --pred-post /Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/COMBINED_NEURIPS_SAME_PREID/dani_synapse_mapping_results/hemi/HEMIBRAIN_synapses_x15035-15635_y28559-29159_z9602-10202_pred_post_locations.csv \
    --pred-mapping /Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/COMBINED_NEURIPS_SAME_PREID/dani_synapse_mapping_results/hemi/HEMIBRAIN_synapses_x15035-15635_y28559-29159_z9602-10202_pre_post_mapping.csv \
    --gt-only --visualize

    """
    # read the zarr first
    zarr_path = "/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/COMBINED_NEURIPS_SAME_PREID/data_3d/mito_tests/mito_hemi_x15035-15635_y28559-29159_z9602-10202.zarr"

    # Print available datasets in the zarr file
    print("Available datasets in zarr file:")
    zarr_root = zarr.open(zarr_path, mode='r')
    for key in zarr_root:
        print(f"- {key}")
        if isinstance(zarr_root[key], zarr.hierarchy.Group):
            for subkey in zarr_root[key]:
                print(f"  - {key}/{subkey}")
                if isinstance(zarr_root[key][subkey], zarr.hierarchy.Group):
                    for subsubkey in zarr_root[key][subkey]:
                        print(f"    - {key}/{subkey}/{subsubkey}")

    # Load the mito data - adjust path if needed based on the output above
    try:
        mito_data = zarr_root["volumes/labels/mito_ids"]
        print(f"Loaded mito data with shape: {mito_data.shape}")
    except KeyError:
        print("Could not find mito_ids at volumes/labels/mito_ids, trying volumes/mito...")
        try:
            mito_data = zarr_root["volumes/mito"]
            print(f"Loaded mito data with shape: {mito_data.shape}")
        except KeyError:
            print("Error: Could not find mitochondria data in the zarr file.")
            print("Please check the zarr file structure and update the path.")
            return

    # Load the neuron segmentation
    try:
        neuron_segmentation = zarr_root["volumes/labels/neuron_ids"]
        print(f"Loaded neuron segmentation with shape: {neuron_segmentation.shape}")
    except KeyError:
        print("Could not find neuron_ids at volumes/labels/neuron_ids, trying volumes/labels/neuron...")
        try:
            neuron_segmentation = zarr_root["volumes/labels/neuron"]
            print(f"Loaded neuron segmentation with shape: {neuron_segmentation.shape}")
        except KeyError:
            print("Error: Could not find neuron segmentation in the zarr file.")
            print("Please check the zarr file structure and update the path.")
            return


    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Validate synapses with mitochondria data')
    parser.add_argument('--gt-pre', required=True,
                        help='CSV file containing ground truth pre-synaptic locations')
    parser.add_argument('--gt-post', required=True,
                        help='CSV file containing ground truth post-synaptic locations')
    parser.add_argument('--pred-pre', required=True,
                        help='CSV file containing predicted pre-synaptic locations')
    parser.add_argument('--pred-post', required=True,
                        help='CSV file containing predicted post-synaptic locations')
    parser.add_argument('--pred-mapping', required=True,
                        help='CSV file containing predicted mapping of synapses to neurons')
    parser.add_argument('--matching-threshold', type=float, default=550,
                        help='Distance threshold for matching synapses (default: 550nm)')
    parser.add_argument('--output-dir', default='./results_mito',
                        help='Directory to save results (default: results)')
    parser.add_argument('--mito-distance-threshold', type=float, default=3000,
                        help='Distance threshold for mitochondria in nm (default: 3000nm)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations of the results')
    parser.add_argument('--gt-only', action='store_true',
                        help='Only analyze ground truth data')
    args = parser.parse_args()

    # Create output directory
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Get resolution from zarr metadata if available
    try:
        resolution = np.array(mito_data.attrs.get('resolution', [8, 8, 8]))
        print(f"Using resolution from zarr: {resolution} nm")
    except:
        resolution = np.array([8, 8, 8])  # Default resolution in nm
        print(f"Using default resolution: {resolution} nm")

    # Load synapse data from CSV files
    gt_pre_df = pd.read_csv(args.gt_pre)
    gt_post_df = pd.read_csv(args.gt_post)

    print(f"Loaded ground truth synapse data:")
    print(f"GT Pre: {gt_pre_df.shape}, GT Post: {gt_post_df.shape}")

    # Extract coordinate columns for ground truth
    gt_pre_cols, gt_pre_id = get_coordinate_columns(gt_pre_df, 'Pre')
    gt_post_cols, gt_post_id = get_coordinate_columns(gt_post_df, 'Post')

    # Convert to numpy arrays
    gt_pre_positions = gt_pre_df[gt_pre_cols].values
    gt_post_positions = gt_post_df[gt_post_cols].values

    print(f"Neuron segmentation shape (ZYX): {neuron_segmentation.shape}")
    print(f"Mito segmentation shape (ZYX): {mito_data.shape}")
    print(f"First few pre-synaptic positions: {gt_pre_positions[:3]}")
    print(f"Resolution (ZYX): {resolution}")

    # Assign ground truth synapses to neurons
    gt_pre_assignments, gt_post_assignments = assign_synapses_to_neurons(
        gt_pre_positions, gt_post_positions, neuron_segmentation, resolution)

    print(f"Assigned {len(gt_pre_assignments)} GT pre-synaptic sites to neurons")
    print(f"Assigned {len(gt_post_assignments)} GT post-synaptic sites to neurons")

    # Find same-neuron synapses to suppress in ground truth
    gt_to_suppress = filter_same_neuron_synapses(gt_pre_assignments, gt_post_assignments)
    print(f"Found {len(gt_to_suppress)} GT synapses to suppress (same neuron)")

    # Calculate distances to mitochondria for ground truth
    gt_mito_distances = calculate_distances_to_mitochondria(
                        gt_pre_positions, mito_data, neuron_segmentation, gt_pre_assignments, resolution, args.mito_distance_threshold)
    print(f"Calculated distances to mitochondria for {len(gt_mito_distances)} GT pre-synaptic sites")

    # Analyze the distribution of distances
    if gt_mito_distances:
        distances = list(gt_mito_distances.values())
        print(f"GT Mito distance statistics:")
        print(f"  Min: {np.min(distances):.2f} nm")
        print(f"  Max: {np.max(distances):.2f} nm")
        print(f"  Mean: {np.mean(distances):.2f} nm")
        print(f"  Median: {np.median(distances):.2f} nm")

        # Count synapses within different distance thresholds
        thresholds = [500, 1000, 1500, 2000, 3000, 4000]
        for threshold in thresholds:
            count = sum(1 for d in distances if d <= threshold)
            percentage = (count / len(distances)) * 100
            print(f"  Synapses within {threshold} nm: {count} ({percentage:.1f}%)")

    # Only process predicted data if not in gt-only mode
    if not args.gt_only:
        pred_pre_df = pd.read_csv(args.pred_pre)
        pred_post_df = pd.read_csv(args.pred_post)

        print(f"Loaded predicted synapse data:")
        print(f"Pred Pre: {pred_pre_df.shape}, Pred Post: {pred_post_df.shape}")

        # Extract coordinate columns for predictions
        pred_pre_cols, pred_pre_id = get_coordinate_columns(pred_pre_df, 'Pre')
        pred_post_cols, pred_post_id = get_coordinate_columns(pred_post_df, 'Post')

        # Convert to numpy arrays
        pred_pre_positions = pred_pre_df[pred_pre_cols].values
        pred_post_positions = pred_post_df[pred_post_cols].values

        # Assign predicted synapses to neurons
        pred_pre_assignments, pred_post_assignments = assign_synapses_to_neurons(
            pred_pre_positions, pred_post_positions, neuron_segmentation, resolution)

        # Find same-neuron synapses to suppress in predictions
        pred_to_suppress = filter_same_neuron_synapses(pred_pre_assignments, pred_post_assignments)

        # Calculate distances to mitochondria for predictions
        pred_mito_distances = calculate_distances_to_mitochondria(
            pred_pre_positions, mito_data, neuron_segmentation, pred_pre_assignments, resolution,
            args.mito_distance_threshold)

        # Extract dataset name from the input file path
    dataset_name = os.path.basename(args.gt_pre).split('_')[0]
    if not dataset_name:
        dataset_name = "unknown"
    
    # Save results to JSON
    json_path = save_results_to_json(
        out_dir,
        gt_pre_positions, gt_post_positions,
        gt_pre_assignments, gt_post_assignments,
        gt_mito_distances, gt_to_suppress,
        pred_pre_positions if not args.gt_only else None,
        pred_post_positions if not args.gt_only else None,
        pred_pre_assignments if not args.gt_only else None,
        pred_post_assignments if not args.gt_only else None,
        pred_mito_distances if not args.gt_only else None,
        pred_to_suppress if not args.gt_only else None,
        dataset_name
    )   

    # Generate visualizations for synapses when they are on the same neuron
    if args.visualize:
        print("Generating visualizations...")
        
        # Visualize ground truth synapses
        visualize_synapses(
            gt_pre_positions, gt_post_positions,
            out_dir, "gt_synapse",
            gt_to_suppress, gt_mito_distances
        )
        
        # Visualize predicted synapses if available
        if not args.gt_only:
            visualize_synapses(
                pred_pre_positions, pred_post_positions,
                out_dir, "pred_synapse",
                pred_to_suppress, pred_mito_distances
            )

    # Generate visualizations if requested
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Create a directory for visualizations
            vis_dir = os.path.join(out_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

            # Visualize distance distribution for ground truth
            if gt_mito_distances:
                plt.figure(figsize=(10, 6))
                sns.histplot(list(gt_mito_distances.values()), bins=30, kde=True)
                plt.title('Distribution of Distances from GT Pre-synaptic Sites to Nearest Mitochondria')
                plt.xlabel('Distance (nm)')
                plt.ylabel('Count')
                plt.axvline(x=500, color='r', linestyle='--', label='500 nm')
                plt.axvline(x=1000, color='g', linestyle='--', label='1000 nm')
                plt.axvline(x=2000, color='y', linestyle='--', label='2000 nm')
                plt.axvline(x=3000, color='b', linestyle='--', label='3000 nm')
                plt.legend()
                plt.savefig(os.path.join(vis_dir, 'gt_mito_distance_distribution.png'))
                plt.close()

                # Create a cumulative distribution plot
                plt.figure(figsize=(10, 6))
                sns.ecdfplot(list(gt_mito_distances.values()))
                plt.title('Cumulative Distribution of Distances from GT Pre-synaptic Sites to Nearest Mitochondria')
                plt.xlabel('Distance (nm)')
                plt.ylabel('Proportion')
                plt.axvline(x=500, color='r', linestyle='--', label='500 nm')
                plt.axvline(x=1000, color='g', linestyle='--', label='1000 nm')
                plt.axvline(x=2000, color='y', linestyle='--', label='2000 nm')
                plt.axvline(x=3000, color='y', linestyle='--', label='2000 nm')
                plt.legend()
                plt.savefig(os.path.join(vis_dir, 'gt_mito_distance_cumulative.png'))
                plt.close()

            # Compare GT and predicted distances if both are available
            if not args.gt_only and gt_mito_distances and pred_mito_distances:
                plt.figure(figsize=(10, 6))
                sns.histplot(list(gt_mito_distances.values()), bins=30, alpha=0.5, label='Ground Truth', kde=True)
                sns.histplot(list(pred_mito_distances.values()), bins=30, alpha=0.5, label='Predicted', kde=True)
                plt.title('Comparison of Distances to Nearest Mitochondria')
                plt.xlabel('Distance (nm)')
                plt.ylabel('Count')
                plt.legend()
                plt.savefig(os.path.join(vis_dir, 'mito_distance_comparison.png'))
                plt.close()

                # Create a box plot comparison
                plt.figure(figsize=(8, 6))
                data = {
                    'Ground Truth': list(gt_mito_distances.values()),
                    'Predicted': list(pred_mito_distances.values())
                }
                sns.boxplot(data=data)
                plt.title('Comparison of Distances to Nearest Mitochondria')
                plt.ylabel('Distance (nm)')
                plt.savefig(os.path.join(vis_dir, 'mito_distance_boxplot.png'))
                plt.close()

            print(f"Visualizations saved to {vis_dir}")

        except ImportError:
            print("Visualization requires matplotlib and seaborn. Please install with:")
            print("pip install matplotlib seaborn")

    # Save results to file
    results = {
        'gt_synapses': len(gt_pre_positions),
        'gt_assigned': len(gt_pre_assignments),
        'gt_same_neuron': len(gt_to_suppress),
        'gt_with_mito_distance': len(gt_mito_distances),
    }

    if not args.gt_only:
        results.update({
            'pred_synapses': len(pred_pre_positions),
            'pred_assigned': len(pred_pre_assignments),
            'pred_same_neuron': len(pred_to_suppress),
            'pred_with_mito_distance': len(pred_mito_distances),
        })

    # Add distance statistics
    if gt_mito_distances:
        distances = list(gt_mito_distances.values())
        results['gt_mito_distance_stats'] = {
            'min': float(np.min(distances)),
            'max': float(np.max(distances)),
            'mean': float(np.mean(distances)),
            'median': float(np.median(distances)),
        }

        # Count synapses within different distance thresholds
        for threshold in [500, 1000, 2000, 3000]:
            count = sum(1 for d in distances if d <= threshold)
            percentage = (count / len(distances)) * 100
            results[f'gt_within_{threshold}nm'] = {
                'count': count,
                'percentage': float(percentage)
            }

    # Save results to file
    with open(os.path.join(out_dir, 'mito_analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Analysis complete. Results saved to {out_dir}")


if __name__ == "__main__":
    # Uncomment the following lines to run the script
    main()
