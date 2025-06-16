"""
Objectives:
0. First group all Pre-site and Post-sites in GT by position to make uniques. Get Coordinate Columns because they may differ across models (people developing them)

1. Assign all pre-post detected points to their neuron segmentation.

2. Assign all segmentation mitochondria to their corresponding neuron segmentation

3. Check for duplicate pre-post detections for the neuron pairs. For example, if we see that a synapse pair has been detected twice for the same neuron pair, we should agglomerate those into one detection.

4. Find the distances of the pre-syn sites from its nearest mitochondria.

5. Suppress all synapse detections that lie within the same neuron. Synapses must be between neurons and the positional detections should not be on the same neuron.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Set, Optional, Union
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
from scipy.stats import ks_2samp


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


def assign_synapses_to_neurons(
        pre_positions: npt.NDArray | dict,
        post_positions: npt.NDArray | dict,
        neuron_segmentation: npt.NDArray,
        resolution: npt.NDArray,
        out_path: str = "./gt_presyn-to-mito.csv"
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

    # print(type(pre_positions_zyx))
    if isinstance(pre_positions_zyx, set):
        pre_voxels = np.array(
            [np.round(x / resolution).astype(int) for x in pre_positions_zyx])  # needed to handle the unique gt pos
    else:
        pre_voxels = np.round(pre_positions_zyx / resolution).astype(int)  # when ndarray no conversion needed

    if isinstance(post_positions_zyx, set):
        post_voxels = np.array(
            [np.round(x / resolution).astype(int) for x in post_positions_zyx])  # needed to handle the unique gt pos
    else:
        post_voxels = np.round(post_positions_zyx / resolution).astype(int)  # when ndarray no conversion needed

    # save which pos could not be assigned
    pre_assignments_missing = {}
    post_assignments_missing = {}

    # save which pos could be assigned with id
    pre_assignments_pos = []

    # Ensure coordinates are within bounds
    for i, pos in enumerate(pre_voxels):
        if all(0 <= p < s for p, s in zip(pos, neuron_segmentation.shape)):
            neuron_id = neuron_segmentation[pos[0], pos[1], pos[2]]
            if neuron_id > 0:  # Ignore background (usually label 0)
                pre_assignments[i] = neuron_id
                pre_assignments_pos.append({
                    "id": i,
                    "loc_zyx": tuple(pos),
                    "neuron_id": neuron_id
                })
            else:
                pre_assignments_missing[tuple(pos)] = 'Neuron id  = 0'  # cannot be assigned
        else:
            pre_assignments_missing[tuple(pos)] = 'Not within bounds'  # cannot be assigned

    for i, pos in enumerate(post_voxels):
        if all(0 <= p < s for p, s in zip(pos, neuron_segmentation.shape)):
            neuron_id = neuron_segmentation[pos[0], pos[1], pos[2]]
            if neuron_id > 0:
                post_assignments[i] = neuron_id
            else:
                post_assignments_missing[tuple(pos)] = 'Neuron id  = 0'  # cannot be assigned
        else:
            post_assignments_missing[tuple(pos)] = 'Not within bounds'  # cannot be assigned

    print("Missing assignments pre-post")
    print(f"pre \n {pre_assignments_missing}")
    print(f"post \n {post_assignments_missing}")

    print(f"Saving presyn-to-neuron assignments: {out_path}")
    df = pd.DataFrame(pre_assignments_pos)
    df[['z', 'y', 'x']] = pd.DataFrame(df['loc_zyx'].tolist(), index=df.index)
    #   drop the original 'loc' column if you like:
    df = df.drop(columns='loc_zyx')

    # 3) Save to CSV (or Excel, pickle, etc.)
    df.to_csv(f'{out_path}', index=False)

    return pre_assignments, post_assignments


def assign_mitochondria_to_neurons(
        mito_segmentation,
        neuron_segmentation,
        out_path: str = "./mito_to_neuron_mapping.csv") -> Tuple[Dict[int, int], List[int], List[int]]:
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
    # # Convert Zarr arrays to numpy arrays for processing
    # if hasattr(mito_segmentation, 'is_zarr_array') or str(type(mito_segmentation)).find('zarr') != -1:
    #     print("Converting mito segmentation from Zarr to NumPy array...")
    #     mito_segmentation = np.array(mito_segmentation[:])
    #
    # if hasattr(neuron_segmentation, 'is_zarr_array') or str(type(neuron_segmentation)).find('zarr') != -1:
    #     print("Converting neuron segmentation from Zarr to NumPy array...")
    #     neuron_segmentation = np.array(neuron_segmentation[:])

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
    mito_to_neuron_mapping = []

    for mito_id in tqdm(mito_ids, desc="Assigning mitochondria to neurons", total=len(mito_ids)):
        mito_mask = mito_segmentation == mito_id

        # Find overlapping neuron IDs
        dilated_mito_mask = binary_dilation(mito_mask, iterations=1)
        overlapping_neurons = neuron_segmentation[dilated_mito_mask]

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

            # now append **all** mappings, flagging the best one
            for neuron_id, count in neuron_counts.items():
                mito_to_neuron_mapping.append({
                    "mito_id": mito_id,
                    "neuron_id": neuron_id,
                    "overlap": count,
                    "is_best": (neuron_id == best_neuron)
                })

    # convert to DataFrame
    df = pd.DataFrame(mito_to_neuron_mapping)

    # if you like, sort so that best ones appear first
    df = df.sort_values(["mito_id", "is_best"], ascending=[True, False])

    # and save it
    df.to_csv(f"{out_path}", index=False)

    return mito_to_neuron, mito_ids.tolist(), neuron_ids.tolist()


def check_duplicate_assignments_in_pred_n_flag():
    pass


def calculate_distances_to_mitochondria(
        pre_positions: npt.NDArray,
        mito_segmentation: npt.NDArray,
        neuron_segmentation: npt.NDArray,
        mito_to_neuron,
        mito_ids,
        pre_assignments: Dict[int, int],
        resolution: npt.NDArray,
        max_search_radius: float = 1000.0,  # nm
        out_path: str = "./pre_to_mito_mapping.csv"
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
    # if hasattr(mito_segmentation, 'is_zarr_array') or str(type(mito_segmentation)).find('zarr') != -1:
    #     print("Converting mito segmentation from Zarr to NumPy array...")
    #     mito_segmentation = np.array(mito_segmentation[:])
    #
    # if hasattr(neuron_segmentation, 'is_zarr_array') or str(type(neuron_segmentation)).find('zarr') != -1:
    #     print("Converting neuron segmentation from Zarr to NumPy array...")
    #     neuron_segmentation = np.array(neuron_segmentation[:])

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

    # # First, assign mitochondria to neurons
    # mito_to_neuron, mito_ids, _ = assign_mitochondria_to_neurons(mito_segmentation, neuron_segmentation)

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
    mappings: List[Dict[str, float]] = []  # find syn to mito mappings

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
        for mito_id in tqdm(neuron_to_mitos[neuron_id], total=len(neuron_to_mitos[neuron_id]),
                            desc="Num of mitos in neuron"):
            min_dist = np.inf
            if mito_id in mito_coords:
                mito_pos = mito_coords[mito_id]
                dist = np.linalg.norm(pos - mito_pos)
                # if dist <= max_search_radius: Do we need a search radius?
                mito_distances.append(dist)
                mappings.append({
                    "pre_index": i,
                    "pre_locs_zyx": pos,
                    "neuron_id": neuron_id,
                    "mito_id": mito_id,
                    "distance_nm": dist
                })

        # Record the minimum distance if any mitochondria were found
        if mito_distances:
            distances[i] = min(mito_distances)

            for item in mappings:
                if item["pre_index"] == i:
                    if item["distance_nm"] == min(mito_distances):
                        item.update({"is_min_dist": True})
                    else:
                        item.update({"is_min_dist": False})

    # Build DataFrame and save
    df = pd.DataFrame(mappings)
    df[['z', 'y', 'x']] = pd.DataFrame(df['pre_locs_zyx'].tolist(), index=df.index)
    #   drop the original 'loc' column if you like:
    df = df.drop(columns='pre_locs_zyx')
    df.to_csv(f"{out_path}", index=False)

    return distances, mito_coords


def save_synapses_to_csv(
        out_dir: str,
        pre_positions: npt.NDArray,
        post_positions: npt.NDArray,
        pre_assignments: Dict[int, int],
        post_assignments: Dict[int, int],
        mito_distances: Dict[int, float],
        to_suppress: List[int],
        prefix: str = "synapses"
) -> str:
    """
    Save synapse information to a CSV file.

    Args:
        out_dir: Output directory
        pre_positions: Pre-synaptic positions
        post_positions: Post-synaptic positions
        pre_assignments: Dict mapping synapse indices to neuron IDs for pre-synaptic sites
        post_assignments: Dict mapping synapse indices to neuron IDs for post-synaptic sites
        mito_distances: Dict mapping synapse indices to distances to mitochondria
        to_suppress: List of synapse indices to suppress (same neuron)
        prefix: Prefix for output file

    Returns:
        Path to the saved CSV file
    """
    # Create a DataFrame to store the synapse information
    data = []

    for i in range(len(pre_positions)):
        # Get pre and post positions
        pre_pos = pre_positions[i]
        post_pos = post_positions[i]

        # Get neuron assignments if available
        pre_neuron_id = pre_assignments.get(i)
        post_neuron_id = post_assignments.get(i)

        # Get mitochondria distance if available
        mito_distance = mito_distances.get(i)

        # Check if this synapse should be suppressed
        is_same_neuron = i in to_suppress

        # Create a row for this synapse
        row = {
            'synapse_id': i,
            'pre_x': pre_pos[0],
            'pre_y': pre_pos[1],
            'pre_z': pre_pos[2],
            'post_x': post_pos[0],
            'post_y': post_pos[1],
            'post_z': post_pos[2],
            'pre_neuron_id': pre_neuron_id if pre_neuron_id is not None else -1,
            'post_neuron_id': post_neuron_id if post_neuron_id is not None else -1,
            'mito_distance': mito_distance if mito_distance is not None else -1,
            'same_neuron': is_same_neuron,
            'valid': not is_same_neuron and pre_neuron_id is not None and post_neuron_id is not None
        }

        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    df.to_csv(csv_path, index=False)

    print(f"Saved {len(data)} synapses to {csv_path}")
    return csv_path


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

    # Create a custom JSON encoder to handle NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    results = {
        "dataset_name": dataset_name,
        "ground_truth": {
            "synapses": [],
            "stats": {
                "total_synapses": int(len(gt_pre_positions)),
                "assigned_pre": int(len(gt_pre_assignments)),
                "assigned_post": int(len(gt_post_assignments)),
                "with_mito_distance": int(len(gt_mito_distances)),
                "same_neuron_synapses": int(len(gt_to_suppress))
            }
        }
    }

    # Add ground truth synapse data
    for i in range(len(gt_pre_positions)):
        # Convert neuron IDs to standard Python int
        pre_neuron_id = gt_pre_assignments.get(i)
        if pre_neuron_id is not None:
            pre_neuron_id = int(pre_neuron_id)

        post_neuron_id = gt_post_assignments.get(i)
        if post_neuron_id is not None:
            post_neuron_id = int(post_neuron_id)

        # Convert mito distance to standard Python float
        mito_distance = gt_mito_distances.get(i)
        if mito_distance is not None:
            mito_distance = float(mito_distance)

        synapse = {
            "id": int(i),
            "pre_position": gt_pre_positions[i].tolist(),
            "post_position": gt_post_positions[i].tolist(),
            "pre_neuron_id": pre_neuron_id,
            "post_neuron_id": post_neuron_id,
            "mito_distance": mito_distance,
            "same_neuron": i in gt_to_suppress
        }
        results["ground_truth"]["synapses"].append(synapse)

    # Add predicted data if available
    if pred_pre_positions is not None:
        results["prediction"] = {
            "synapses": [],
            "stats": {
                "total_synapses": int(len(pred_pre_positions)),
                "assigned_pre": int(len(pred_pre_assignments) if pred_pre_assignments else 0),
                "assigned_post": int(len(pred_post_assignments) if pred_post_assignments else 0),
                "with_mito_distance": int(len(pred_mito_distances) if pred_mito_distances else 0),
                "same_neuron_synapses": int(len(pred_to_suppress) if pred_to_suppress else 0)
            }
        }

        for i in range(len(pred_pre_positions)):
            # Convert neuron IDs to standard Python int
            pre_neuron_id = pred_pre_assignments.get(i) if pred_pre_assignments else None
            if pre_neuron_id is not None:
                pre_neuron_id = int(pre_neuron_id)

            post_neuron_id = pred_post_assignments.get(i) if pred_post_assignments else None
            if post_neuron_id is not None:
                post_neuron_id = int(post_neuron_id)

            # Convert mito distance to standard Python float
            mito_distance = pred_mito_distances.get(i) if pred_mito_distances else None
            if mito_distance is not None:
                mito_distance = float(mito_distance)

            synapse = {
                "id": int(i),
                "pre_position": pred_pre_positions[i].tolist(),
                "post_position": pred_post_positions[i].tolist(),
                "pre_neuron_id": pre_neuron_id,
                "post_neuron_id": post_neuron_id,
                "mito_distance": mito_distance,
                "same_neuron": i in pred_to_suppress if pred_to_suppress else False
            }
            results["prediction"]["synapses"].append(synapse)

    # Save to JSON file
    json_path = os.path.join(out_dir, f"synapse_analysis_{dataset_name}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"Saved results to {json_path}")
    return json_path


def visualize_synapses(
        pre_positions: npt.NDArray,
        post_positions: npt.NDArray,
        out_dir: str,
        prefix: str = "synapse",
        to_suppress: Optional[List[int]] = None,
        mito_distances: Optional[Dict[int, float]] = None,
        max_synapses: int = 20,
        em_data: Optional[npt.NDArray] = None,
        resolution: Optional[npt.NDArray] = None
):
    """
    Visualize synapses with pre and post-synaptic sites and arrows, optionally with EM data.

    Args:
        pre_positions: Pre-synaptic positions (in nm)
        post_positions: Post-synaptic positions (in nm)
        out_dir: Output directory
        prefix: Prefix for output files
        to_suppress: List of synapse indices to suppress (same neuron)
        mito_distances: Dict mapping synapse indices to distances to mitochondria
        max_synapses: Maximum number of synapses to visualize
        em_data: Optional EM data array for background visualization (in pixel coordinates)
        resolution: Resolution of the EM data in nm per pixel (3-element array)
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
        indices = indices[:max_synapses // 2] + indices[-max_synapses // 2:]
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
                ax.scatter(post[0], post[1], post[2], color='green', s=100,
                           label='Post-synaptic' if i == indices[0] else "")

                # Draw arrow from pre to post (orange)
                ax.quiver(pre[0], pre[1], pre[2],
                          post[0] - pre[0], post[1] - pre[1], post[2] - pre[2],
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

            # If EM data is provided, show it as background
            if em_data is not None and resolution is not None:
                # Calculate average position of synapses in nm
                avg_pos_nm = np.mean([pre_positions[i] for i in indices], axis=0)

                # Convert average position from nm to pixel coordinates
                # Note: resolution is in nm/pixel, so divide nm by resolution to get pixels
                avg_pos_pixel = np.round(avg_pos_nm / resolution).astype(int)

                # Get the appropriate slice based on the view
                if view_name == 'xy':
                    # Make sure slice index is within bounds
                    slice_idx = min(max(0, avg_pos_pixel[2]), em_data.shape[2] - 1)
                    em_slice = em_data[:, :, slice_idx]

                    # Calculate the extent in nm coordinates for proper alignment
                    # extent = [left, right, bottom, top] in data coordinates
                    extent = [0, em_slice.shape[1] * resolution[1],
                              em_slice.shape[0] * resolution[0], 0]  # Flip Y-axis for image coordinates

                    # Display the EM slice with correct scaling
                    ax.imshow(em_slice, cmap='gray', alpha=0.7, extent=extent)

                elif view_name == 'xz':
                    slice_idx = min(max(0, avg_pos_pixel[1]), em_data.shape[1] - 1)
                    em_slice = em_data[:, slice_idx, :]

                    extent = [0, em_slice.shape[1] * resolution[2],
                              em_slice.shape[0] * resolution[0], 0]  # Flip Y-axis

                    ax.imshow(em_slice, cmap='gray', alpha=0.7, extent=extent)

                elif view_name == 'yz':
                    slice_idx = min(max(0, avg_pos_pixel[0]), em_data.shape[0] - 1)
                    em_slice = em_data[slice_idx, :, :]

                    extent = [0, em_slice.shape[1] * resolution[2],
                              em_slice.shape[0] * resolution[1], 0]  # Flip Y-axis

                    ax.imshow(em_slice, cmap='gray', alpha=0.7, extent=extent)

            # Plot each synapse
            for i in indices:
                pre = pre_positions[i]
                post = post_positions[i]

                # Determine color based on whether it's a same-neuron synapse
                color = 'red' if i in to_suppress else 'blue'

                # Plot pre-synaptic site (red)
                ax.scatter(pre[axis1], pre[axis2], color='red', s=100, label='Pre-synaptic' if i == indices[0] else "")

                # Plot post-synaptic site (green)
                ax.scatter(post[axis1], post[axis2], color='green', s=100,
                           label='Post-synaptic' if i == indices[0] else "")

                # Draw arrow from pre to post (orange)
                ax.arrow(pre[axis1], pre[axis2],
                         post[axis1] - pre[axis1], post[axis2] - pre[axis2],
                         color='orange', width=5, head_width=20, head_length=20,
                         length_includes_head=True, label='Connection' if i == indices[0] else "")

                # Add text label with synapse ID and mito distance if available
                label = f"ID: {i}"
                if mito_distances and i in mito_distances:
                    label += f"\nMito: {mito_distances[i]:.1f}nm"
                if i in to_suppress:
                    label += "\nSame neuron"

                ax.text(pre[axis1], pre[axis2], label, fontsize=8)

            ax.set_xlabel(f"{'X' if axis1 == 0 else 'Y' if axis1 == 1 else 'Z'} (nm)")
            ax.set_ylabel(f"{'Y' if axis2 == 1 else 'Z'} (nm)")
            ax.set_title(f"{title} - {prefix}")

            # Add legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        # Save the figure
        plt.tight_layout()
        sns.despine()
        plt.savefig(os.path.join(vis_dir, f"{prefix}_{view_name}.png"), dpi=300)
        plt.savefig(os.path.join(vis_dir, f"{prefix}_{view_name}.svg"), dpi=300)
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


def get_nearest_mito_for_pre(
        pre_positions: npt.NDArray,
        pre_assignments: Dict[int, int],
        mito_coords: Dict[int, npt.NDArray],
        neuron_to_mitos: Dict[int, List[int]]
) -> Dict[int, Tuple[int, float]]:
    """
    For each pre-synapse index i (in pre_positions), find its nearest mitochondrion:
      - returns a dict: i -> (mito_id, distance_nm)
    """
    nearest = {}
    for i, pos in enumerate(pre_positions):
        if i not in pre_assignments:
            continue
        neuron_id = pre_assignments[i]
        candidates = neuron_to_mitos.get(neuron_id, [])
        best = None
        best_d = np.inf
        for m in candidates:
            if m not in mito_coords:
                continue
            d = np.linalg.norm(pos - mito_coords[m])
            if d < best_d:
                best_d, best = d, m
        if best is not None:
            nearest[i] = (best, best_d)
    return nearest


def visualize_pre_synapse_with_mito(
        pre_idx: int,
        raw_em: npt.NDArray,
        mito_seg: npt.NDArray,
        pre_positions: Union[npt.NDArray, Dict[int, npt.NDArray]],
        nearest_map: Dict[int, Tuple[int, float]],
        resolution: npt.NDArray,
        window: int = 50
):
    """
    Show a single Z-slice of raw EM, overlaying the pre-syn point
    and its assigned mito mask.
    Accepts pre_positions either as:
      • ndarray of shape [N,3], or
      • dict mapping pre_idx -> 3-element array.
    """

    # ---- sanity check ----
    if isinstance(pre_positions, set):
        raise TypeError(
            "visualize_pre_synapse_with_mito: pre_positions is a set; "
            "please pass an np.ndarray (shape [N,3]) or a dict."
        )

    # ---- fetch the 3D nm-coordinate ----
    if isinstance(pre_positions, dict):
        pos_nm = pre_positions[pre_idx]
    else:
        pos_nm = pre_positions[pre_idx]

    if pre_idx not in nearest_map:
        raise ValueError(f"pre_idx {pre_idx} has no mito assignment")

    mito_id, dist = nearest_map[pre_idx]

    # ---- convert nm → voxels and crop ----
    voxel = np.round(pos_nm / resolution).astype(int)
    z, y, x = voxel

    y0, y1 = max(0, y - window), min(raw_em.shape[1], y + window)
    x0, x1 = max(0, x - window), min(raw_em.shape[2], x + window)

    em_slice = raw_em[z, y0:y1, x0:x1]
    mito_mask = (mito_seg[z, y0:y1, x0:x1] == mito_id)

    # ---- plot ----
    plt.figure(figsize=(6, 6))
    plt.imshow(em_slice, cmap='gray')
    plt.contour(mito_mask, colors='orange', linewidths=1,
                extent=(0, x1 - x0, y1 - y0, 0))
    py, px = y - y0, x - x0
    plt.scatter([px], [py], c='red', s=50, label=f'Pre #{pre_idx}')
    plt.title(f"Pre {pre_idx} ↔ Mito {mito_id} ({dist:.1f} nm)")
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.show()


def get_mitos_within_radius(
        pre_positions: npt.NDArray,  # [N_pre,3] in nm
        pre_assignments: Dict[int, int],  # pre_idx -> neuron_id
        mito_coords: Dict[int, npt.NDArray],  # mito_id -> [3] in nm
        neuron_to_mitos: Dict[int, List[int]],  # neuron_id -> [mito_ids]
        radius_nm: float
) -> Dict[int, List[Tuple[int, float]]]:
    """
    For each pre-synapse i, find all mito IDs *in the same neuron* within `radius_nm`.
    Returns: pre_idx -> [(mito_id, distance_nm), ...], sorted by distance.
    """
    multi_map: Dict[int, List[Tuple[int, float]]] = {}
    for i, pos in enumerate(pre_positions):
        neuron_id = pre_assignments.get(i)
        if neuron_id is None:
            continue
        hits: List[Tuple[int, float]] = []
        for m_id in neuron_to_mitos.get(neuron_id, []):
            m_pos = mito_coords.get(m_id)
            if m_pos is None:
                continue
            d = float(np.linalg.norm(pos - m_pos))
            if d <= radius_nm:
                hits.append((m_id, d))
        if hits:
            multi_map[i] = sorted(hits, key=lambda x: x[1])
    return multi_map


def analyze_and_plot_mito_to_presynapse_distances(
        mito_ids: List[int],
        mito_coords: Dict[int, npt.NDArray],
        mito_to_neuron: Dict[int, int],
        pre_positions: Union[npt.NDArray, set],
        neuron_to_presynapses: Dict[int, List[int]],
        out_dir: str
) -> List[float]:
    """
    For every mitochondrion, finds the distance to the nearest pre-synaptic site
    in the same neuron and plots the distribution of these distances.

    Args:
        mito_ids (List[int]): A list of all mitochondrion IDs to analyze.
        mito_coords (Dict[int, npt.NDArray]): Dict mapping mito ID to its coordinates.
        mito_to_neuron (Dict[int, int]): Dict mapping mito ID to its host neuron ID.
        pre_positions (Union[npt.NDArray, set]): Collection of all pre-synaptic positions.
        neuron_to_presynapses (Dict[int, List[int]]): Dict mapping neuron ID to a list of its pre-synapse indices.
        out_dir (str): Directory to save the output plots.

    Returns:
        A list of the minimum distances found for each mitochondrion.
    """
    print("\n--- Analyzing distances from each mitochondrion to its nearest pre-synapse ---")

    # Ensure pre_positions is an indexable numpy array
    if isinstance(pre_positions, set):
        pre_positions = np.array(list(pre_positions))

    mito_to_nearest_presyn_dist = {}

    # Iterate through all mitochondria to find the nearest pre-synaptic site for each
    for mito_id in tqdm(mito_ids, desc="Calculating mito-to-presynapse distances"):
        neuron_id = mito_to_neuron.get(mito_id)
        if neuron_id is None:
            continue

        mito_pos = mito_coords.get(mito_id)
        if mito_pos is None:
            continue

        presynapse_indices_in_neuron = neuron_to_presynapses.get(neuron_id, [])
        if not presynapse_indices_in_neuron:
            continue

        presynapse_positions_in_neuron = pre_positions[presynapse_indices_in_neuron]

        if presynapse_positions_in_neuron.size == 0:
            continue

        # Calculate distances from the current mitochondrion to all pre-synapses in the same neuron
        distances = cdist(mito_pos.reshape(1, -1), presynapse_positions_in_neuron)[0]
        min_dist = np.min(distances)
        mito_to_nearest_presyn_dist[mito_id] = min_dist

    if not mito_to_nearest_presyn_dist:
        print("Could not calculate any mito-to-presynapse distances. Skipping plot generation.")
        return []

    # Now, plot the aggregated distribution of these minimum distances
    all_distances = list(mito_to_nearest_presyn_dist.values())
    print(f"\nCalculated nearest pre-synapse distance for {len(all_distances)} mitochondria.")
    print(
        f"Distance stats (nm): Min={np.min(all_distances):.2f}, Max={np.max(all_distances):.2f}, Mean={np.mean(all_distances):.2f}")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        vis_dir = os.path.join(out_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # Plot 1: Histogram of distances
        plt.figure(figsize=(10, 6))
        sns.histplot(all_distances, bins=30, kde=True)
        plt.title('Distribution of Distances from Mitochondria to Nearest Pre-Synaptic Site')
        plt.xlabel('Distance to Nearest Pre-Synapse (nm)')
        plt.ylabel('Mitochondrion Count')
        sns.despine()
        plt.savefig(os.path.join(vis_dir, 'mito_to_presynapse_distance_distribution.png'), dpi=300)
        plt.close()

        # Plot 2: Cumulative Distribution Plot
        plt.figure(figsize=(10, 6))
        sns.ecdfplot(all_distances, stat="count")
        plt.title('Cumulative Count of Mitochondria by Distance to Nearest Pre-Synapse')
        plt.xlabel('Distance to Nearest Pre-Synapse (nm)')
        plt.ylabel('Cumulative Number of Mitochondria')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        sns.despine()
        plt.savefig(os.path.join(vis_dir, 'mito_to_presynapse_distance_cumulative.png'), dpi=300)
        plt.close()

        print(f"Saved mito-to-presynapse distance plots to {vis_dir}")

    except ImportError:
        print("Visualization requires matplotlib and seaborn. Please install them to generate plots.")

    return all_distances


def perform_statistical_comparison(distances1: List[float], distances2: List[float], label1: str, label2: str,
                                   out_dir: str = "./"):
    """
    Performs a two-sample Kolmogorov-Smirnov (K-S) test to compare two distance distributions.

    Args:
        distances1 (List[float]): First list of distances.
        distances2 (List[float]): Second list of distances.
        label1 (str): Label for the first distribution.
        label2 (str): Label for the second distribution.
    """
    print("\n--- Statistical Comparison of Distance Distributions ---")
    print(f"Comparing '{label1}' (N={len(distances1)}) vs. '{label2}' (N={len(distances2)})")

    if not distances1 or not distances2:
        print("One or both distance lists are empty. Cannot perform K-S test.")
        return

    # Perform the two-sample K-S test
    ks_statistic, p_value = ks_2samp(distances1, distances2)

    print("\nKolmogorov-Smirnov Test Results:")
    print(f"  K-S Statistic: {ks_statistic:.4f}")
    print(f"  P-value: {p_value:.4g}")

    content_to_write = f"\nKolmogorov-Smirnov Test Results:\n  K-S Statistic: {ks_statistic:.4f} \n  P-value: {p_value:.4g}"

    alpha = 0.05
    if p_value < alpha:
        print(f"\nConclusion: The p-value is less than {alpha}, so we reject the null hypothesis.")
        print("The two distance distributions are statistically different.")
        content_to_write += f"\nConclusion: The p-value is less than {alpha}, so we reject the null hypothesis. " \
                            f"\n The two distance distributions are statistically different."
    else:
        print(
            f"\nConclusion: The p-value is greater than or equal to {alpha}, so we fail to reject the null hypothesis.")
        print("We cannot conclude that the two distributions are different.")
        content_to_write += f"\nConclusion: The p-value is greater than or equal to {alpha}, so we fail to reject the null hypothesis." \
                            f"We cannot conclude that the two distributions are different."

    with open(f"{out_dir}/KS_stat_results.txt", 'w', encoding='utf-8') as file:
        file.write(content_to_write)


def load_raw_seg(zarr_path):
    """ One single zarr must contain the raw EM, neuron seg and mito seg.
    Offsets are not considered"""
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

    # Load the raw EM
    try:
        raw_em = zarr_root["volumes/raw"]
        print(f"Loaded raw EM data with shape: {raw_em.shape}")
    except KeyError:
        print("Could not find raw EM data at volumes/raw, trying volumes/raw/s0...")
        try:
            raw_em = zarr_root["volumes/raw/s0"]
            print(f"Loaded raw EM with shape: {raw_em.shape}")
        except KeyError:
            print("Error: Could not find raw EM in the zarr file.")
            print("Please check the zarr file structure and update the path.")
            return

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

    # Get resolution from zarr metadata if available
    try:
        resolution = np.array(mito_data.attrs.get('resolution', [8, 8, 8]))
        print(f"Using resolution from zarr: {resolution} nm")
    except Exception as e:
        resolution = np.array([8, 8, 8])  # Default resolution in nm
        raise Warning(f"Using default resolution: {resolution} nm")

    # Convert Zarr arrays to numpy arrays for processing
    if hasattr(mito_data, 'is_zarr_array') or str(type(mito_data)).find('zarr') != -1:
        print("Converting mito segmentation from Zarr to NumPy array...")
        mito_segmentation = np.array(mito_data[:])

    if hasattr(neuron_segmentation, 'is_zarr_array') or str(type(neuron_segmentation)).find('zarr') != -1:
        print("Converting neuron segmentation from Zarr to NumPy array...")
        neuron_segmentation = np.array(neuron_segmentation[:])

    return raw_em, neuron_segmentation, mito_segmentation, resolution


def load_synapses(gt_pre, gt_post):
    # Load synapse data from CSV files
    gt_pre_df = pd.read_csv(gt_pre)
    gt_post_df = pd.read_csv(gt_post)

    print(f"Loaded ground truth synapse data:")
    print(f"GT Pre: {gt_pre_df.shape}, GT Post: {gt_post_df.shape}")

    # Extract coordinate columns for ground truth
    gt_pre_cols, gt_pre_id = get_coordinate_columns(gt_pre_df, 'Pre')
    gt_post_cols, gt_post_id = get_coordinate_columns(gt_post_df, 'Post')

    # Convert to numpy arrays
    gt_pre_positions = gt_pre_df[gt_pre_cols].values
    gt_post_positions = gt_post_df[gt_post_cols].values

    print(f"First few pre-synaptic positions: {gt_pre_positions[:3]}")

    ## Group by pre-post points
    gt_pre_grouped = gt_pre_df.groupby(gt_pre_cols)
    gt_post_grouped = gt_post_df.groupby(gt_post_cols)

    print("Show gt_pre_grouped and post_grouped dfs...")
    print(f"gt_pre_grouped \n: {gt_pre_grouped.apply(lambda a: a.drop(gt_pre_cols, axis=1)[:])}")
    # print(f"gt_post_grouped \n: {gt_post_grouped.apply(print)}")

    # Find uniques directly based on the positions?
    positions_gt_pre_tuple = [tuple(pos) for pos in gt_pre_positions]
    gt_pre_unique_positions = set(positions_gt_pre_tuple)

    positions_gt_positions_tuple = [tuple(pos) for pos in gt_post_positions]
    gt_post_unique_positions = set(positions_gt_positions_tuple)

    # print(gt_pre_unique_positions)
    # print(len(gt_pre_unique_positions))
    #
    # print(gt_post_unique_positions)
    # print(len(gt_post_unique_positions))

    return gt_pre_grouped, gt_post_grouped, gt_pre_unique_positions, gt_post_unique_positions


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

    parser = argparse.ArgumentParser(description='Validate synapses with mitochondria data')
    parser.add_argument('--zarr-path', required=True,
                        help='Path containing the neuron and mito segmentations and raw EM data')
    # Parse command line arguments

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
    parser.add_argument('--mito-distance-threshold', type=float, default=6000,
                        help='Distance threshold for mitochondria in nm (default: 3000nm)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations of the results')
    parser.add_argument('--gt-only', action='store_true',
                        help='Only analyze ground truth data')
    args = parser.parse_args()

    # Extract dataset name from the input file path
    dataset_name = os.path.basename(args.gt_pre).split('_')[0]
    if not dataset_name:
        dataset_name = "unknown"

    # Create output directory
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    zarr_path = args.zarr_path

    raw_em, neuron_segmentation, mito_segmentation, resolution = load_raw_seg(zarr_path)

    gt_pre_grouped, gt_post_grouped, gt_pre_unique_positions, gt_post_unique_positions = load_synapses(
        gt_pre=args.gt_pre, gt_post=args.gt_post)

    # Assign ground truth synapses to neurons
    gt_pre_assignments, gt_post_assignments = assign_synapses_to_neurons(
        gt_pre_unique_positions, gt_post_unique_positions, neuron_segmentation, resolution,
        out_path=f"{out_dir}/gt_presyn-neuron-assignments.csv")

    # Assign mitochondria to neurons. Calculating this once is enough because this mapping will not change for syn preds
    mito_to_neuron, mito_ids, neuron_ids = assign_mitochondria_to_neurons(mito_segmentation, neuron_segmentation,
                                                                          out_path=f"{out_dir}/gt_mito_to_neuron_mapping.csv")

    # Calculate distances to mitochondria for ground truth
    gt_mito_distances, mito_coords = calculate_distances_to_mitochondria(
        pre_positions=gt_pre_unique_positions,
        mito_segmentation=mito_segmentation,
        neuron_segmentation=neuron_segmentation,
        mito_to_neuron=mito_to_neuron,
        mito_ids=mito_ids,
        pre_assignments=gt_pre_assignments,
        resolution=resolution,
        max_search_radius=args.mito_distance_threshold,
        out_path=f"{out_dir}/gt_pre_to_mito_mapping_at_distances.csv"
    )
    print(f"Calculated distances to mitochondria for {len(gt_mito_distances)} GT pre-synaptic sites")

    # --- New Analysis: For every mito, find nearest pre-synapse and plot distributions ---
    mito_to_pre_distances = []
    if mito_ids and gt_pre_assignments and mito_coords:
        # Create a reverse map from neuron_id to its pre-synapse indices for efficient lookup
        neuron_to_gt_presynapses = {}
        for pre_idx, neuron_id in gt_pre_assignments.items():
            if neuron_id not in neuron_to_gt_presynapses:
                neuron_to_gt_presynapses[neuron_id] = []
            neuron_to_gt_presynapses[neuron_id].append(pre_idx)

        mito_to_pre_distances = analyze_and_plot_mito_to_presynapse_distances(
            mito_ids=mito_ids,
            mito_coords=mito_coords,
            mito_to_neuron=mito_to_neuron,
            pre_positions=gt_pre_unique_positions,
            neuron_to_presynapses=neuron_to_gt_presynapses,
            out_dir=out_dir
        )

    # --- Perform Statistical Comparison ---
    syn_to_mito_distances = list(gt_mito_distances.values())
    if syn_to_mito_distances and mito_to_pre_distances:
        perform_statistical_comparison(
            syn_to_mito_distances,
            mito_to_pre_distances,
            label1="Pre-Synapse to nearest Mito",
            label2="Mito to nearest Pre-Synapse",
            out_dir=out_dir
        )

    if mito_coords:

        mito_coords = {m: cm for m, cm in mito_coords.items()}  # from calculate_distances_to_mitochondria
        neuron_to_mitos = {}
        for m, n in mito_to_neuron.items():
            neuron_to_mitos.setdefault(n, []).append(m)
        # Build nearest-mito map:
        gt_nearest = get_nearest_mito_for_pre(
            pre_positions=gt_pre_unique_positions,
            pre_assignments=gt_pre_assignments,
            mito_coords=mito_coords,
            neuron_to_mitos=neuron_to_mitos
        )
        for pre_i, (mito_id, d) in gt_nearest.items():
            print(f"Pre-syn {pre_i} → Mito {mito_id} @ {d:.1f} nm")

        # define your search radius
        radius = 3000.0

        multi_mito_map = get_mitos_within_radius(
            pre_positions=gt_pre_unique_positions,
            pre_assignments=gt_pre_assignments,
            mito_coords=mito_coords,
            neuron_to_mitos=neuron_to_mitos,
            radius_nm=radius
        )

        # save to disk
        import json
        with open(f'{out_dir}/multi_mito_map.json', 'w') as f:
            # {pre_idx: [[mito_id, dist_nm], ...]}
            json.dump({str(k): [[int(m), d] for m, d in hits]
                       for k, hits in multi_mito_map.items()}, f)

        # if args.visualize: # this is worthless
        #     visualize_pre_synapse_with_mito(
        #         pre_idx=5, # take the last pre_i from above
        #         raw_em=raw_em,
        #         mito_seg=mito_segmentation,
        #         pre_positions=np.array(list(gt_pre_unique_positions)),
        #         nearest_map=gt_nearest,
        #         resolution=resolution
        #     )

    # Analyze the distribution of distances
    if gt_mito_distances:
        distances = list(gt_mito_distances.values())
        print(f"GT Mito distance statistics:")
        print(f"  Min: {np.min(distances):.2f} nm")
        print(f"  Max: {np.max(distances):.2f} nm")
        print(f"  Mean: {np.mean(distances):.2f} nm")
        print(f"  Median: {np.median(distances):.2f} nm")

        # Count synapses within different distance thresholds
        thresholds = [500, 1000, 1500, 2000, 3000, 4000, 6000]
        for threshold in thresholds:
            count = sum(1 for d in distances if d <= threshold)
            percentage = (count / len(gt_mito_distances)) * 100
            print(f"  Synapses within {threshold} nm: {count} ({percentage:.1f}%)")

    # convert the pre_positions to ndarray
    gt_pre_unique_positions = np.array(list(gt_pre_unique_positions))
    gt_post_unique_positions = np.array(list(gt_post_unique_positions))
    # Save ground truth synapses to CSV
    gt_to_suppress = []
    gt_csv_path = save_synapses_to_csv(
        out_dir,
        gt_pre_unique_positions, gt_post_unique_positions,
        gt_pre_assignments, gt_post_assignments,
        gt_mito_distances, gt_to_suppress,
        prefix="gt_synapses"
    )

    print(f"GT Synapses saved: {gt_csv_path}")

    ## Predicted:
    # Assign predicted synapses to neurons and calculate distances to mitochondria
    if not args.gt_only:
        pred_pre_df = pd.read_csv(args.pred_pre)
        pred_post_df = pd.read_csv(args.pred_post)

        print(f"Loaded predicted synapse data:")
        print(f"Pred Pre: {pred_pre_df.shape}, Pred Post: {pred_post_df.shape}")

        # Extract coordinate columns for predictions
        pred_pre_cols, pred_pre_id = get_coordinate_columns(pred_pre_df, 'Pre')
        pred_post_cols, pred_post_id = get_coordinate_columns(pred_post_df, 'Post')

        # Convert to numpy arrays. Ungrouped because we want to get all assignments without losing the duplicate FPs
        pred_pre_positions = pred_pre_df[pred_pre_cols].values
        pred_post_positions = pred_post_df[pred_post_cols].values

        # Assign predicted synapses to neurons
        pred_pre_assignments, pred_post_assignments = assign_synapses_to_neurons(
            pred_pre_positions, pred_post_positions, neuron_segmentation, resolution,
            out_path=f"{out_dir}/pred_presyn-neuron-assignments.csv")

        # Find same-neuron synapses to suppress in predictions
        pred_to_suppress = filter_same_neuron_synapses(pred_pre_assignments, pred_post_assignments)

        # Calculate distances to mitochondria for predictions
        pred_mito_distances, pred_mito_coords = calculate_distances_to_mitochondria(
            pre_positions=pred_pre_positions,
            mito_segmentation=mito_segmentation,
            neuron_segmentation=neuron_segmentation,
            mito_to_neuron=mito_to_neuron,
            mito_ids=mito_ids,
            pre_assignments=pred_pre_assignments,
            resolution=resolution,
            max_search_radius=args.mito_distance_threshold,
            out_path=f"{out_dir}/pred_pre_to_mito_mapping_at_distances.csv")

        # Save results to JSON
        json_path = save_results_to_json(
            out_dir,
            gt_pre_unique_positions, gt_post_unique_positions,
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

        pred_csv_path = save_synapses_to_csv(
            out_dir,
            pred_pre_positions, pred_post_positions,
            pred_pre_assignments, pred_post_assignments,
            pred_mito_distances, pred_to_suppress,
            prefix="pred_synapses"
        )

        print(f"p"
              f"Predicted Synapses saved: {pred_csv_path} with saved json {json_path}")

        # Generate visualizations for synapses when they are on the same neuron
        if args.visualize:
            print("Generating visualizations...")
            visualize_synapses(
                pred_pre_positions, pred_post_positions,
                out_dir, "pred_synapse",
                pred_to_suppress, pred_mito_distances,
                em_data=raw_em,  # Load EM data from zarr
                resolution=resolution

            )

    ## Visualizations:
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
                sns.despine()
                plt.savefig(os.path.join(vis_dir, 'gt_mito_distance_distribution.png'), dpi=300)
                plt.savefig(os.path.join(vis_dir, 'gt_mito_distance_distribution.svg'), dpi=300)
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
                plt.axvline(x=3000, color='y', linestyle='--', label='3000 nm')
                plt.axvline(x=4000, color='gray', linestyle='--', label='4000 nm')
                plt.axvline(x=5000, color='gray', linestyle='--', label='5000 nm')
                plt.axvline(x=6000, color='gray', linestyle='--', label='6000 nm')
                plt.legend()
                sns.despine()
                plt.savefig(os.path.join(vis_dir, 'gt_mito_distance_cumulative.png'), dpi=300)
                plt.savefig(os.path.join(vis_dir, 'gt_mito_distance_cumulative.svg'), dpi=300)
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
                # This handles lists of different lengths by creating a DataFrame with NaN padding
                sns.boxplot(data=pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()])))
                plt.title('Comparison of Distances to Nearest Mitochondria')
                plt.ylabel('Distance (nm)')
                sns.despine()
                plt.savefig(os.path.join(vis_dir, 'mito_distance_boxplot.png'), dpi=300)
                plt.savefig(os.path.join(vis_dir, 'mito_distance_boxplot.svg'), dpi=300)
                plt.close()

            print(f"Visualizations saved to {vis_dir}")

        except ImportError:
            print("Visualization requires matplotlib and seaborn. Please install with:")
            print("pip install matplotlib seaborn")

    print()


if __name__ == '__main__':
    main()
