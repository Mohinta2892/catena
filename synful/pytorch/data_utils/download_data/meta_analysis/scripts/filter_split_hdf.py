"""
The script performs the following operations:
- Checks and filters out autapses.
- Check and filters out synapses that are duplicates, meaning it's same pre and the post
repeating multiple times in a region.
- Splits the hdf files into train and test sets, but keeps the original dims intact.
For example, for a 600px^3 with 700 synapses,
an 80-20 split would make train hdf with 560 synapses with dims 600px^3 and
the rest would go to the test set with dims 600px^3.
- Calculates the post point distances from the boundary pixels given a neuron segmentation.
Distances are in pixels and nms.
- For filtering and boundary pixel distance calculations,
csv files with corresponding filtered points and distances are saved to disk.

Author: Samia Mohinta
Affiliation: University of Cambridge, UK
"""

import os
import glob
import argparse
import h5py
import numpy as np
import pandas as pd
import itertools

# Define the spatial resolution in nanometers per pixel.
RESOLUTION_NM = 8.0


def nm_to_px(locations_nm, offset_nm, resolution):
    """Converts coordinates from nanometers to pixels."""
    if locations_nm is None or offset_nm is None:
        return None
    locations_px = (locations_nm - offset_nm) / resolution
    return np.round(locations_px).astype(int)


def get_neuron_id_at_loc(location_px, segmentation_vol):
    """Safely retrieves the neuron ID from the segmentation volume."""
    dims = segmentation_vol.shape
    if not (0 <= location_px[0] < dims[0] and
            0 <= location_px[1] < dims[1] and
            0 <= location_px[2] < dims[2]):
        return 0
    return segmentation_vol[location_px[0], location_px[1], location_px[2]]


def filter_autapses(partners, loc_map, segmentation_vol, offset_nm, resolution):
    """
    Identifies and filters synapses where pre and post sites are on the same neuron.
    """
    valid_synapses = []
    autapses_report = []

    print("Filtering autapses...")
    for pre_id, post_id in partners:
        pre_loc_nm = loc_map.get(pre_id)
        post_loc_nm = loc_map.get(post_id)

        if pre_loc_nm is None or post_loc_nm is None:
            continue

        pre_loc_px = nm_to_px(pre_loc_nm, offset_nm, resolution)
        post_loc_px = nm_to_px(post_loc_nm, offset_nm, resolution)

        pre_neuron_id = get_neuron_id_at_loc(pre_loc_px, segmentation_vol)
        post_neuron_id = get_neuron_id_at_loc(post_loc_px, segmentation_vol)

        if pre_neuron_id == 0 or post_neuron_id == 0:
            continue

        synapse_info = {
            'pre_id': pre_id,
            'post_id': post_id,
            'pre_loc_nm': pre_loc_nm,
            'post_loc_nm': post_loc_nm,
            'pre_loc_px': pre_loc_px,
            'post_loc_px': post_loc_px,
            'pre_neuron_id': pre_neuron_id,
            'post_neuron_id': post_neuron_id,
        }

        if pre_neuron_id == post_neuron_id:
            autapses_report.append({
                'pre_id': pre_id,
                'post_id': post_id,
                'pre_location_nm': pre_loc_nm,
                'post_location_nm': post_loc_nm,
                'neuron_id': pre_neuron_id
            })
        else:
            valid_synapses.append(synapse_info)

    print(f"Found {len(valid_synapses)} valid synapses and {len(autapses_report)} autapses.")
    return valid_synapses, autapses_report


def filter_multi_contact_synapses(synapses):
    """
    Identifies and filters cases where one presynaptic site connects to multiple
    postsynaptic sites on the same target neuron.
    """
    if not synapses:
        return [], []

    print("Filtering multi-contact synapses (one pre-site to one post-neuron)...")

    synapse_df = pd.DataFrame(synapses)
    connection_counts = synapse_df.groupby(['pre_id', 'post_neuron_id']).size().reset_index(name='count')
    multi_contact_groups = connection_counts[connection_counts['count'] > 1]

    if multi_contact_groups.empty:
        print("No multi-contact synapses found.")
        return synapses, []

    multi_contact_set = set(map(tuple, multi_contact_groups[['pre_id', 'post_neuron_id']].to_numpy()))

    single_contact_synapses = []
    multi_contact_report = []

    for synapse in synapses:
        if (synapse['pre_id'], synapse['post_neuron_id']) in multi_contact_set:
            multi_contact_report.append(synapse)
        else:
            single_contact_synapses.append(synapse)

    print(f"Found {len(single_contact_synapses)} single-contact synapses and "
          f"filtered out {len(multi_contact_report)} synapses belonging to multi-contact groups.")

    return single_contact_synapses, multi_contact_report


def find_close_postsynaptic_pairs(synapses, distance_threshold_nm):
    """
    Finds pairs of postsynaptic sites on the same neuron that are within a
    given distance of each other.
    """
    if not synapses:
        return []

    print(f"Finding close postsynaptic pairs within {distance_threshold_nm} nm...")

    post_sites = []
    seen_post_ids = set()
    for s in synapses:
        if s['post_id'] not in seen_post_ids:
            post_sites.append({
                'post_id': s['post_id'],
                'post_loc_nm': s['post_loc_nm'],
                'post_neuron_id': s['post_neuron_id'],
                'pre_id': s['pre_id']
            })
            seen_post_ids.add(s['post_id'])

    if not post_sites:
        return []

    post_df = pd.DataFrame(post_sites)
    close_pairs_report = []
    grouped_by_neuron = post_df.groupby('post_neuron_id')

    for neuron_id, group in grouped_by_neuron:
        if len(group) < 2:
            continue

        for site1, site2 in itertools.combinations(group.to_dict('records'), 2):
            distance = np.linalg.norm(site1['post_loc_nm'] - site2['post_loc_nm'])

            if distance <= distance_threshold_nm:
                close_pairs_report.append({
                    'neuron_id': neuron_id,
                    'post_id_1': site1['post_id'],
                    'post_id_2': site2['post_id'],
                    'pre_id_1': site1['pre_id'],
                    'pre_id_2': site2['pre_id'],
                    'post_loc_nm_1': site1['post_loc_nm'],
                    'post_loc_nm_2': site2['post_loc_nm'],
                    'distance_nm': distance
                })

    print(f"Found {len(close_pairs_report)} pairs of close postsynaptic sites.")
    return close_pairs_report


def calculate_distance_to_boundary(post_loc_px, pre_loc_px, neuron_id, segmentation_vol):
    """
    Calculates the point on the neuron's boundary in the direction of the presynaptic site.
    This now operates on the full segmentation volume and global coordinates.
    """
    direction_vec = pre_loc_px - post_loc_px
    dist_to_pre_px = np.linalg.norm(direction_vec)

    if dist_to_pre_px == 0:
        return post_loc_px

    norm_direction = direction_vec / dist_to_pre_px

    for step in range(1, int(dist_to_pre_px) + 2):
        current_pos_px = post_loc_px + norm_direction * step
        current_loc_px = np.round(current_pos_px).astype(int)
        current_neuron_id = get_neuron_id_at_loc(current_loc_px, segmentation_vol)
        if current_neuron_id != neuron_id:
            return current_loc_px

    return pre_loc_px


def save_new_hdf5(filepath, raw_vol, labels_vol, synapses, offset_nm, id_map, type_map):
    """
    Saves the data into a new HDF5 file. The raw and labels volumes are saved
    in their entirety, while the synapse annotations are filtered for the specific split.
    """
    with h5py.File(filepath, 'w') as f:
        # Save the full, unsplit volumes
        f.create_dataset('volumes/raw', data=raw_vol, compression='gzip')
        f.create_dataset('volumes/labels/neuron_ids', data=labels_vol, compression='gzip')

        # Save the filtered list of synapses for this split
        if synapses:
            # Get all unique synapse point IDs present in this split
            synapse_ids_in_split = set()
            for s in synapses:
                synapse_ids_in_split.add(s['pre_id'])
                synapse_ids_in_split.add(s['post_id'])

            unique_ids = sorted(list(synapse_ids_in_split))

            locations = [id_map[uid] for uid in unique_ids]
            types = [type_map[uid] for uid in unique_ids]
            partners = [(s['pre_id'], s['post_id']) for s in synapses]

            anno_grp = f.create_group('annotations')
            anno_grp.attrs['offset'] = offset_nm
            anno_grp.create_dataset('ids', data=unique_ids)
            anno_grp.create_dataset('locations', data=locations, compression='gzip')
            anno_grp.create_dataset('types', data=[s.encode('utf8') for s in types])
            anno_grp.create_dataset('presynaptic_site/partners', data=partners)

        print(f"Saved new HDF5 file: {filepath}")


def process_hdf_file(file_path, output_dir, train_split_ratio, close_post_threshold):
    """
    Main processing function for a single HDF5 file.
    """
    print(f"\nProcessing file: {file_path}")
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    try:
        with h5py.File(file_path, 'r') as f:
            print("Loading data from HDF5 file...")
            neuron_segmentation = f['volumes/labels/neuron_ids'][:]
            raw_volume = f['volumes/raw'][:]
            ids = f['annotations/ids'][:]
            locations = f['annotations/locations'][:]
            types = [t.decode('utf8') for t in f['annotations/types'][:]]
            partners = f['annotations/presynaptic_site/partners'][:]
            offset = f['annotations'].attrs.get('offset', np.array([0, 0, 0]))

            loc_map = {id_val: loc for id_val, loc in zip(ids, locations)}
            type_map = {id_val: type_val for id_val, type_val in zip(ids, types)}

            valid_synapses, autapses_report = filter_autapses(
                partners, loc_map, neuron_segmentation, offset, RESOLUTION_NM
            )

            if autapses_report:
                autapses_df = pd.DataFrame(autapses_report)
                autapses_csv_path = os.path.join(output_dir, f"{base_name}_filtered_autapses.csv")
                autapses_df.to_csv(autapses_csv_path, index=False)
                print(f"Saved autapse report to {autapses_csv_path}")

            close_pairs_report = find_close_postsynaptic_pairs(valid_synapses, close_post_threshold)
            if close_pairs_report:
                report_df = pd.DataFrame(close_pairs_report)
                for col in ['post_loc_nm_1', 'post_loc_nm_2']:
                    report_df[[f'{col}_z', f'{col}_y', f'{col}_x']] = pd.DataFrame(report_df[col].tolist(), index=report_df.index)
                    report_df = report_df.drop(columns=col)
                close_pairs_csv_path = os.path.join(output_dir, f"{base_name}_close_postsynaptic_pairs.csv")
                report_df.to_csv(close_pairs_csv_path, index=False)
                print(f"Saved close postsynaptic pairs report to {close_pairs_csv_path}")

            final_synapses, multi_contact_report = filter_multi_contact_synapses(valid_synapses)

            if multi_contact_report:
                report_df = pd.DataFrame(multi_contact_report)
                for col in ['pre_loc_nm', 'post_loc_nm', 'pre_loc_px', 'post_loc_px']:
                    report_df[[f'{col}_z', f'{col}_y', f'{col}_x']] = pd.DataFrame(report_df[col].tolist(), index=report_df.index)
                    report_df = report_df.drop(columns=col)
                multi_contact_csv_path = os.path.join(output_dir, f"{base_name}_filtered_multi_contact.csv")
                report_df.to_csv(multi_contact_csv_path, index=False)
                print(f"Saved multi-contact synapse report to {multi_contact_csv_path}")

            print(f"Splitting synapse list with a {train_split_ratio * 100:.0f}/{100 - train_split_ratio * 100:.0f} ratio...")
            split_axis_idx = 2 # The split is still conceptually along the x-axis
            split_point_px = int(neuron_segmentation.shape[split_axis_idx] * train_split_ratio)

            train_synapses = []
            test_synapses = []

            for synapse in final_synapses:
                # Use the global pixel coordinates for the split decision
                pre_x_px = synapse['pre_loc_px'][split_axis_idx]
                post_x_px = synapse['post_loc_px'][split_axis_idx]
                if pre_x_px < split_point_px and post_x_px < split_point_px:
                    train_synapses.append(synapse)
                elif pre_x_px >= split_point_px and post_x_px >= split_point_px:
                    test_synapses.append(synapse)

            print(f"Train set has {len(train_synapses)} synapses. Test set has {len(test_synapses)} synapses.")

            # --- Boundary Distance Calculation (Simplified) ---
            # Now operates on the full volume for both sets
            for name, synapses in [('train', train_synapses), ('test', test_synapses)]:
                if not synapses:
                    print(f"No synapses in {name} set. Skipping distance calculation.")
                    continue

                print(f"Calculating boundary distances for {name} set...")
                boundary_distances = []
                for synapse in synapses:
                    # Use global pixel coordinates directly
                    boundary_px = calculate_distance_to_boundary(
                        synapse['post_loc_px'],
                        synapse['pre_loc_px'],
                        synapse['post_neuron_id'],
                        neuron_segmentation # Pass the full segmentation volume
                    )

                    boundary_nm = (boundary_px * RESOLUTION_NM) + offset
                    dist_vec_nm = boundary_nm - synapse['post_loc_nm']
                    dz, dy, dx = dist_vec_nm

                    boundary_distances.append({
                        'post_id': synapse['post_id'],
                        'pre_id': synapse['pre_id'],
                        'post_neuron_id': synapse['post_neuron_id'],
                        'post_location_nm_z_y_x': synapse['post_loc_nm'],
                        'pre_location_nm_z_y_x': synapse['pre_loc_nm'],
                        'boundary_location_nm_z_y_x': boundary_nm,
                        'total_distance_to_boundary_nm': np.linalg.norm(dist_vec_nm),
                        'distance_in_xy_plane_nm': np.sqrt(dx**2 + dy**2),
                        'distance_in_yz_plane_nm': np.sqrt(dy**2 + dz**2),
                        'distance_in_zx_plane_nm': np.sqrt(dz**2 + dx**2),
                    })

                if boundary_distances:
                    distances_df = pd.DataFrame(boundary_distances)
                    distances_csv_path = os.path.join(output_dir, f"{base_name}_{name}_boundary_distances.csv")
                    distances_df.to_csv(distances_csv_path, index=False)
                    print(f"Saved {name} set boundary distances to {distances_csv_path}")

            # --- Save Final HDF5 Files ---
            # Save the train split
            save_new_hdf5(
                os.path.join(output_dir, f"{base_name}_train.hdf"),
                raw_volume, neuron_segmentation, train_synapses, offset, loc_map, type_map
            )

            # Save the test split
            save_new_hdf5(
                os.path.join(output_dir, f"{base_name}_test.hdf"),
                raw_volume, neuron_segmentation, test_synapses, offset, loc_map, type_map
            )

    except Exception as e:
        print(f"Could not process file {file_path}. Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Process synapse annotations in HDF5 files. This script filters autapses, "
                    "splits data into train/test sets, and calculates boundary distances.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing the input HDF5 files."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory where the output files will be saved."
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.8,
        help="The ratio for the train/test split (e.g., 0.8 for an 80/20 split). Default is 0.8."
    )
    parser.add_argument(
        "--close_post_threshold",
        type=float,
        default=80.0,
        help="Distance in nm to consider two postsynaptic sites 'close' for reporting. Default is 1000.0."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at '{args.input_dir}'")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    hdf_files = glob.glob(os.path.join(args.input_dir, '*.h5')) + \
                glob.glob(os.path.join(args.input_dir, '*.hdf')) + \
                glob.glob(os.path.join(args.input_dir, '*.hdf5'))

    if not hdf_files:
        print(f"No HDF5 files found in '{args.input_dir}'")
        return

    for hdf_file in hdf_files:
        process_hdf_file(hdf_file, args.output_dir, args.split_ratio, args.close_post_threshold)

    print("\nProcessing complete.")


if __name__ == '__main__':
    main()
