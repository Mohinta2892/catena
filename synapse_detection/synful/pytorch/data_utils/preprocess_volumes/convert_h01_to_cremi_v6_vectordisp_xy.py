import os
import glob
import numpy as np
import h5py
import tifffile
from scipy import ndimage
from scipy.spatial import distance
from tqdm import tqdm


def load_sorted_stack(directory_path):
    """
    Loads a stack of tiffs from a directory, ensuring correct z-order sorting.
    """
    search_path = os.path.join(directory_path, "*.tif*")
    files = sorted(glob.glob(search_path))

    if not files:
        raise FileNotFoundError(f"No tiff files found in {directory_path}")

    # Read first file to determine shape
    first_slice = tifffile.imread(files[0])
    height, width = first_slice.shape
    depth = len(files)

    print(f"Loading {depth} slices from {os.path.basename(directory_path)}...")

    # Initialize volume
    volume = np.zeros((depth, height, width), dtype=first_slice.dtype)

    for i, f in enumerate(tqdm(files, desc="Reading TIFFs")):
        volume[i] = tifffile.imread(f)

    return volume


def extract_centroids_and_bboxes(mask_volume, resolution):
    """
    Finds connected components and calculates the standard Center of Mass.
    """
    print("Labeling connected components...")
    labeled_vol, num_features = ndimage.label(mask_volume > 0)

    if num_features == 0:
        return labeled_vol, {}, []

    print(f"Found {num_features} structures. Calculating centroids...")

    object_slices = ndimage.find_objects(labeled_vol)

    label_data = {}  # label_id -> {'centroid': (z,y,x), 'slice': slice_tuple}
    res_z, res_y, res_x = resolution

    for idx, sl in enumerate(tqdm(object_slices, desc="Calculating Centroids")):
        label_id = idx + 1
        if sl is None: continue

        # Extract small sub-volume
        sub_vol = labeled_vol[sl]

        # Calculate center of mass (Standard approach)
        local_center = ndimage.center_of_mass(sub_vol == label_id)

        if np.any(np.isnan(local_center)):
            continue

        # Convert local to global coordinates
        z_global = (sl[0].start + local_center[0]) * res_z
        y_global = (sl[1].start + local_center[1]) * res_y
        x_global = (sl[2].start + local_center[2]) * res_x

        label_data[label_id] = {
            'centroid': (z_global, y_global, x_global),
            'slice': sl
        }

    return labeled_vol, label_data


def get_local_centroid(mask_slice, label_id, local_cleft_idx, radius_px=30):
    """
    Calculates the Center of Mass for a specific label ONLY within a small radius
    around the cleft. This gives a local direction vector orthogonal to the surface.
    """
    cz, cy, cx = int(local_cleft_idx[0]), int(local_cleft_idx[1]), int(local_cleft_idx[2])

    z_start = max(0, cz - radius_px)
    z_end = min(mask_slice.shape[0], cz + radius_px)
    y_start = max(0, cy - radius_px)
    y_end = min(mask_slice.shape[1], cy + radius_px)
    x_start = max(0, cx - radius_px)
    x_end = min(mask_slice.shape[2], cx + radius_px)

    # Crop around the cleft
    crop = (mask_slice[z_start:z_end, y_start:y_end, x_start:x_end] == label_id)

    if not np.any(crop):
        return None

    local_cm = ndimage.center_of_mass(crop)

    # Adjust back to coordinate system of 'mask_slice'
    return (
        z_start + local_cm[0],
        y_start + local_cm[1],
        x_start + local_cm[2]
    )


def get_refined_location_geometric(cleft_nm, target_nm, mask_vol, label_id, resolution, raw_shape, push_dist_nm=40):
    """
    Refines location using a target direction vector (Local Centroid).

    CRITICAL CHANGE: Displacement vector is zeroed out in Z (0 index) to keep points
    on the original Z plane defined by the cleft center.
    """
    cleft_vec = np.array(cleft_nm)
    target_vec = np.array(target_nm)

    direction = target_vec - cleft_vec

    # --- 1. ENFORCE XY-PLANE DISPLACEMENT ---
    # Set Z direction component to zero, so displacement only happens in Y/X
    direction[0] = 0.0

    norm = np.linalg.norm(direction)

    res_z, res_y, res_x = resolution
    max_z, max_y, max_x = raw_shape

    # --- STRATEGY A: Vector Push ---
    if norm > 0:
        unit_vector = direction / norm
        candidate_nm = cleft_vec + (unit_vector * push_dist_nm)

        # Keep Z from cleft, only Y/X move
        candidate_nm[0] = cleft_vec[0]

        # FIX: Use floor() for robust boundary checking
        z_idx = int(np.floor(candidate_nm[0] / res_z))
        y_idx = int(np.floor(candidate_nm[1] / res_y))
        x_idx = int(np.floor(candidate_nm[2] / res_x))

        # Check Bounds and Mask
        if (0 <= z_idx < mask_vol.shape[0] and
                0 <= y_idx < mask_vol.shape[1] and
                0 <= x_idx < mask_vol.shape[2]):

            if mask_vol[z_idx, y_idx, x_idx] == label_id:
                # Success: Pushed point is valid. Now clamp to final ROI boundary.
                final_nm = np.array(candidate_nm)

                # --- BOUNDARY CLAMPING ---
                final_nm[0] = np.clip(final_nm[0], 0, max_z * res_z)
                final_nm[1] = np.clip(final_nm[1], 0, max_y * res_y)
                final_nm[2] = np.clip(final_nm[2], 0, max_x * res_x)

                return tuple(final_nm)

    # --- STRATEGY B: Fallback to Pole of Inaccessibility (XY CLAMPED) ---
    # If vector push fails, use deepest point in a local crop around the target.
    # The Pole of Inaccessibility naturally pushes away from the boundary.

    tz, ty, tx = int(target_nm[0] / res_z), int(target_nm[1] / res_y), int(target_nm[2] / res_x)
    pad = 60
    z_s = max(0, tz - pad);
    z_e = min(mask_vol.shape[0], tz + pad)
    y_s = max(0, ty - pad);
    y_e = min(mask_vol.shape[1], ty + pad)
    x_s = max(0, tx - pad);
    x_e = min(mask_vol.shape[2], tx + pad)

    sub_mask = (mask_vol[z_s:z_e, y_s:y_e, x_s:x_e] == label_id)

    if not np.any(sub_mask):
        return target_nm  # Absolute fallback

    # Distance Transform
    dt = ndimage.distance_transform_edt(sub_mask, sampling=resolution)
    max_idx_flat = np.argmax(dt)
    max_idx = np.unravel_index(max_idx_flat, dt.shape)

    deepest_global = np.array([
        (z_s + max_idx[0]) * res_z,
        (y_s + max_idx[1]) * res_y,
        (x_s + max_idx[2]) * res_x
    ])

    # --- BOUNDARY CLAMPING (Fallback) ---
    deepest_global[0] = np.clip(deepest_global[0], 0, max_z * res_z)
    deepest_global[1] = np.clip(deepest_global[1], 0, max_y * res_y)
    deepest_global[2] = np.clip(deepest_global[2], 0, max_x * res_x)

    return tuple(deepest_global)


def find_partners_and_refine_coordinates(raw_vol, pre_vol, post_vol, pre_data, post_data, resolution):
    """
    1. Finds 1-to-1 partners.
    2. Refines coordinates using LOCAL orientation vectors for orthogonal connections.
    """
    candidates = []
    raw_shape = raw_vol.shape  # (Z, Y, X) pixel dimensions

    print(f"Scanning overlaps for {len(pre_data)} pre-sites...")

    # --- Phase 1: Find Candidates ---
    for pre_label, p_data in tqdm(pre_data.items(), desc="Finding Candidates"):
        sl = p_data['slice']

        pad = 6
        z_start = max(0, sl[0].start - pad)
        z_end = min(pre_vol.shape[0], sl[0].stop + pad)
        y_start = max(0, sl[1].start - pad)
        y_end = min(pre_vol.shape[1], sl[1].stop + pad)
        x_start = max(0, sl[2].start - pad)
        x_end = min(pre_vol.shape[2], sl[2].stop + pad)

        expanded_slice = (slice(z_start, z_end), slice(y_start, y_end), slice(x_start, x_end))

        sub_pre = pre_vol[expanded_slice]
        sub_post = post_vol[expanded_slice]

        this_pre_mask = (sub_pre == pre_label)
        if not np.any(this_pre_mask): continue

        this_dilated = ndimage.binary_dilation(this_pre_mask, iterations=6)

        overlap = sub_post * this_dilated
        touched_posts = np.unique(overlap)
        touched_posts = touched_posts[touched_posts > 0]

        pre_centroid = np.array(p_data['centroid'])

        for post_label in touched_posts:
            if post_label in post_data:
                post_centroid = np.array(post_data[post_label]['centroid'])
                dist = distance.euclidean(pre_centroid, post_centroid)

                candidates.append({
                    'pre': pre_label,
                    'post': post_label,
                    'dist': dist,
                    'slice': expanded_slice
                })

    # --- Phase 2: Greedy 1-to-1 Matching ---
    candidates.sort(key=lambda x: x['dist'])

    final_links = []
    used_pre = set()
    used_post = set()

    print("Refining coordinates (Local Orthogonal Push, XY-Only)...")

    res_z, res_y, res_x = resolution

    for cand in tqdm(candidates, desc="Optimizing Points"):
        p_id = cand['pre']
        o_id = cand['post']

        if p_id not in used_pre and o_id not in used_post:
            final_links.append((p_id, o_id))
            used_pre.add(p_id)
            used_post.add(o_id)

            # --- Phase 3: Coordinate Refinement ---
            sl = cand['slice']
            sub_pre = pre_vol[sl]
            sub_post = post_vol[sl]

            mask_p = (sub_pre == p_id)
            mask_o = (sub_post == o_id)

            dilated_p = ndimage.binary_dilation(mask_p, iterations=6)
            intersection = dilated_p & mask_o

            if np.any(intersection):
                # 1. Find Cleft (Local coordinates relative to 'sl')
                local_cleft = ndimage.center_of_mass(intersection)

                # 2. Find LOCAL Centroids (near the cleft) to establish orthogonal direction
                loc_pre_rel = get_local_centroid(sub_pre, p_id, local_cleft, radius_px=30)
                loc_post_rel = get_local_centroid(sub_post, o_id, local_cleft, radius_px=30)

                # Convert everything to global NM for the refinement function

                cleft_global = (
                    (sl[0].start + local_cleft[0]) * res_z,
                    (sl[1].start + local_cleft[1]) * res_y,
                    (sl[2].start + local_cleft[2]) * res_x
                )

                # Determine Target for Pre Vector
                if loc_pre_rel:
                    target_pre_nm = (
                        (sl[0].start + loc_pre_rel[0]) * res_z,
                        (sl[1].start + loc_pre_rel[1]) * res_y,
                        (sl[2].start + loc_pre_rel[2]) * res_x
                    )
                else:
                    target_pre_nm = pre_data[p_id]['centroid']  # Fallback to global

                # Determine Target for Post Vector
                if loc_post_rel:
                    target_post_nm = (
                        (sl[0].start + loc_post_rel[0]) * res_z,
                        (sl[1].start + loc_post_rel[1]) * res_y,
                        (sl[2].start + loc_post_rel[2]) * res_x
                    )
                else:
                    target_post_nm = post_data[o_id]['centroid']  # Fallback to global

                # --- REFINEMENT LOGIC (40nm push, clamped, XY-Only) ---
                new_pre_cent = get_refined_location_geometric(
                    cleft_global, target_pre_nm, pre_vol, p_id, resolution, raw_shape, push_dist_nm=20
                )

                new_post_cent = get_refined_location_geometric(
                    cleft_global, target_post_nm, post_vol, o_id, resolution, raw_shape, push_dist_nm=20
                )

                pre_data[p_id]['centroid'] = new_pre_cent
                post_data[o_id]['centroid'] = new_post_cent

    print(f"Resolved and refined {len(final_links)} pairs.")
    return final_links


def convert_to_cremi(root_dir, output_file, resolution=(40, 4, 4)):
    raw_dir = os.path.join(root_dir, 'raw')
    pre_dir = os.path.join(root_dir, 'pre-syns')
    post_dir = os.path.join(root_dir, 'post-syns')

    print("\n--- Loading Data ---")
    raw_vol = load_sorted_stack(raw_dir)
    pre_vol_raw = load_sorted_stack(pre_dir)
    post_vol_raw = load_sorted_stack(post_dir)

    print("\n--- Pre-Synaptic Analysis ---")
    pre_labeled, pre_data = extract_centroids_and_bboxes(pre_vol_raw, resolution)

    print("\n--- Post-Synaptic Analysis ---")
    post_labeled, post_data = extract_centroids_and_bboxes(post_vol_raw, resolution)

    print("\n--- Partner Matching & Refinement ---")
    local_links = find_partners_and_refine_coordinates(
        raw_vol, pre_labeled, post_labeled, pre_data, post_data, resolution
    )

    # Prepare HDF5 Lists
    annot_ids = []
    annot_locs = []
    annot_types = []
    partner_links = []

    pre_global_map = {}
    post_global_map = {}
    current_id = 1

    # Add Pre Points (Refined)
    for local_id, data in pre_data.items():
        annot_ids.append(current_id)
        annot_locs.append(data['centroid'])
        annot_types.append(b'presynaptic_site')
        pre_global_map[local_id] = current_id
        current_id += 1

    # Add Post Points (Refined)
    for local_id, data in post_data.items():
        annot_ids.append(current_id)
        annot_locs.append(data['centroid'])
        annot_types.append(b'postsynaptic_site')
        post_global_map[local_id] = current_id
        current_id += 1

    # Links
    for l_pre, l_post in local_links:
        if l_pre in pre_global_map and l_post in post_global_map:
            g_pre = pre_global_map[l_pre]
            g_post = post_global_map[l_post]
            partner_links.append((g_pre, g_post))

    print(f"Writing to {output_file}...")
    with h5py.File(output_file, 'w') as f:
        dset_raw = f.create_dataset('volumes/raw', data=raw_vol, compression="gzip")
        dset_raw.attrs['resolution'] = resolution

        f.create_dataset('volumes/labels/presynaptic_segments', data=pre_labeled, compression="gzip")
        f.create_dataset('volumes/labels/postsynaptic_segments', data=post_labeled, compression="gzip")

        f.create_dataset('annotations/ids', data=np.array(annot_ids, dtype=np.uint64))
        f.create_dataset('annotations/locations', data=np.array(annot_locs, dtype=np.float64))
        f.create_dataset('annotations/type', data=np.array(annot_types))
        f.create_dataset('annotations/presynaptic_site/partners', data=np.array(partner_links, dtype=np.uint64))
        f.create_dataset('annotations/comments/target_ids', data=np.array([], dtype=np.uint64))
        f.create_dataset('annotations/comments/comments', data=np.array([], dtype='S1'))

        # Add required attributes
        dset_raw.attrs['offset'] = (0, 0, 0)  # Offset is 0, coords are now global
        f["annotations"].attrs["resolution"] = resolution
        f["volumes/labels"].attrs["resolution"] = resolution
        f["annotations"].attrs["offset"] = (0, 0, 0)  # Offset is 0, coords are now global
        f["volumes/labels"].attrs["offset"] = (0, 0, 0)  # Offset is 0, coords are now global

    print("Conversion Complete.")


if __name__ == "__main__":
    # Example usage:
    root_directory = "/Users/sam/Physics_Cambridge Dropbox/Samia Mohinta/Cardona-Zlatic labs shared/Cardona Lab/Samia/Automatic-SynapseDetection/synapse_predictions_paper_w_dani/TEM/HUMAN-H01/ORIGINAL-VAST/Synapse GT Region 6"
    output_filename = "/Users/sam/Physics_Cambridge Dropbox/Samia Mohinta/Cardona-Zlatic labs shared/Cardona Lab/Samia/Automatic-SynapseDetection/synapse_predictions_paper_w_dani/TEM/HUMAN-H01/ORIGINAL-VAST/Synapse GT Region 6/h01_region_6.hdf"
    resolution_nm = (33, 4, 4)  # (z, y, x) - IMPORTANT: Update this to your specific voxel size

    convert_to_cremi(root_directory, output_filename, resolution_nm)
