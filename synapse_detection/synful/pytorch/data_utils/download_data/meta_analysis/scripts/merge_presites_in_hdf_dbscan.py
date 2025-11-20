"""
This script reads an existing CREMI-formatted HDF5 file, merges
pre-synaptic sites that are within a given distance threshold,
and writes a new, cleaned-up HDF5 file.

This version is "slice-aware" and uses density-based clustering
(DBSCAN) to intelligently merge pre-sites. It only forms clusters
from 2 or more points, leaving isolated sites as-is.

Requires: h5py, numpy, scipy, scikit-learn, matplotlib

Example Run:
pip install scikit-learn matplotlib
python merge_presites_in_hdf.py input_file.hdf output_file_merged.hdf --distance_threshold 50 --plot_clusters

----ant ----
python merge_presites_in_hdf.py \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_FB_1.hdf \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_FB_1_m.hdf \
--distance_threshold 300

python merge_presites_in_hdf.py \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_FB_2.hdf \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_FB_2_m.hdf \
--distance_threshold 450

python merge_presites_in_hdf.py \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_FB_3.hdf \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_FB_3_m.hdf \
--distance_threshold 300

--- too little synapses---
python merge_presites_in_hdf.py \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_FB_4.hdf \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_FB_4_m.hdf \
--distance_threshold 300

python merge_presites_in_hdf.py \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_FB_5.hdf \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_FB_5_m.hdf \
--distance_threshold 550

python merge_presites_in_hdf.py \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_NO_1.hdf \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_NO_1_m.hdf \
--distance_threshold 350

python merge_presites_in_hdf.py \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_NO_3.hdf \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_NO_3_m.hdf \
--distance_threshold 300

python merge_presites_in_hdf.py \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_NO_4.hdf \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_NO_4_m.hdf \
--distance_threshold 300

python merge_presites_in_hdf.py \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_NO_5.hdf \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_NO_5_m.hdf \
--distance_threshold 450

python merge_presites_in_hdf.py \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_PB_1.hdf \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_PB_1_m.hdf \
--distance_threshold 450

python merge_presites_in_hdf.py \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_PB_2.hdf \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_PB_2_m.hdf \
--distance_threshold 450

python merge_presites_in_hdf.py \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_PB_4.hdf \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_PB_4_m.hdf \
--distance_threshold 550

python merge_presites_in_hdf.py \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_PB_5.hdf \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_PB_5_m.hdf \
--distance_threshold 550


"""

import argparse
import h5py
import numpy as np
import sys
import os
import itertools

# Scipy is required for clustering
try:
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, fcluster
    # DBSCAN is the new, smarter clustering algorithm
    from sklearn.cluster import DBSCAN
    # Matplotlib is for plotting the clusters
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: This script requires 'scipy', 'scikit-learn', and 'matplotlib'.")
    print("Please install them:")
    print("  pip install scipy scikit-learn matplotlib")
    sys.exit(1)


def merge_pre_sites_slice_aware(input_hdf_path, output_hdf_path, distance_threshold, plot_clusters=False):
    """
    Reads an HDF5 file, merges pre-sites based on 2D (XY) distance
    within the same Z-slice using DBSCAN, and writes a new HDF5 file.
    """

    plot_dir = "cluster_plots_output"
    if plot_clusters:
        os.makedirs(plot_dir, exist_ok=True)
        print(f"Saving cluster verification plots to: {plot_dir}")

    print(f"Opening input file: {input_hdf_path}")
    with h5py.File(input_hdf_path, 'r') as f_in:
        # Read all necessary data from the source file
        in_ids = f_in['annotations/ids'][:]
        in_locations = f_in['annotations/locations'][:]
        in_types = f_in['annotations/types'][:]
        in_partners = f_in['annotations/presynaptic_site/partners'][:]

        # Get attributes to copy
        resolution = f_in['annotations'].attrs['resolution']
        offset = f_in['annotations'].attrs['offset']

        z_res, y_res, x_res = resolution

        # --- 1. Build maps of all original annotations ---
        id_to_loc = {id: in_locations[i] for i, id in enumerate(in_ids)}

        # Find all unique pre-synaptic IDs from the partner list
        unique_pre_ids = sorted(list(set(p[0] for p in in_partners)))
        if not unique_pre_ids:
            print("No pre-synaptic sites found. Aborting.")
            return

        # Get the locations for these unique pre-sites
        unique_pre_locs = {id: id_to_loc[id] for id in unique_pre_ids}

    print(f"Found {len(unique_pre_ids)} unique pre-sites to process.")

    # --- 2. Group pre-sites by Z-slice ---
    # Convert Z-nanometer coord to Z-pixel index for grouping
    sites_by_z_slice = {}
    for pre_id, loc_zyx in unique_pre_locs.items():
        z_nm = loc_zyx[0]
        # Round to nearest pixel index
        z_pixel = int(np.round(z_nm / z_res))

        if z_pixel not in sites_by_z_slice:
            sites_by_z_slice[z_pixel] = []

        sites_by_z_slice[z_pixel].append(pre_id)

    print(f"Grouped pre-sites into {len(sites_by_z_slice)} unique Z-slices.")

    # --- 3. Perform 2D (XY) DBSCAN clustering within each Z-slice ---
    old_pre_id_to_cluster_id = {}
    next_global_cluster_id = 1 # Start from 1

    for z_pixel, pre_ids_in_slice in sites_by_z_slice.items():

        if len(pre_ids_in_slice) == 1:
            # Only one site on this slice, it's its own cluster
            cluster_id = next_global_cluster_id
            next_global_cluster_id += 1

            old_pre_id = pre_ids_in_slice[0]
            old_pre_id_to_cluster_id[old_pre_id] = cluster_id
            continue

        # Get 2D (Y, X) locations for clustering
        slice_locs_2d = np.array([id_to_loc[id][1:] for id in pre_ids_in_slice]) # Get (Y, X)

        # Perform 2D DBSCAN clustering
        # eps = The distance threshold
        # min_samples = 2 means a "cluster" must have at least 2 points.
        # This prevents merging isolated points.
        db = DBSCAN(eps=distance_threshold, min_samples=2).fit(slice_locs_2d)
        cluster_labels = db.labels_ # e.g., [-1, 0, 0, 1, -1, 1]

        # --- PLOTTING ---
        if plot_clusters:
            plt.figure(figsize=(10, 10))
            # Plot X (index 1) vs Y (index 0)
            scatter = plt.scatter(
                slice_locs_2d[:, 1],
                slice_locs_2d[:, 0],
                c=cluster_labels,
                cmap='Paired',
                s=50
            )

            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            plt.title(f"Z-Slice: {z_pixel} (Found {n_clusters} clusters, 'black' = isolated noise)")
            plt.xlabel("X Coordinate (nm)")
            plt.ylabel("Y Coordinate (nm)")
            # Set aspect ratio to equal, essential for distance visualization
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid(True)

            # Add a legend
            handles, labels = scatter.legend_elements()
            legend_labels = []
            for label in np.unique(cluster_labels):
                if label == -1:
                    legend_labels.append("Isolated (Noise)")
                else:
                    legend_labels.append(f"Cluster {label}")
            plt.legend(handles, legend_labels, loc="best")

            plot_filename = os.path.join(plot_dir, f"z_slice_{z_pixel}.png")
            plt.savefig(plot_filename)
            plt.close()
        # --- END PLOTTING ---

        # Map local slice cluster labels to globally unique cluster IDs
        slice_label_to_global_id = {}

        for i, old_pre_id in enumerate(pre_ids_in_slice):
            label = cluster_labels[i]

            if label == -1:
                # Point is "noise" (isolated), assign it a new unique global ID
                global_cluster_id = next_global_cluster_id
                next_global_cluster_id += 1
            else:
                # Point is part of a dense cluster
                if label not in slice_label_to_global_id:
                    # This is the first time we see this cluster label on this slice
                    slice_label_to_global_id[label] = next_global_cluster_id
                    next_global_cluster_id += 1

                global_cluster_id = slice_label_to_global_id[label]

            old_pre_id_to_cluster_id[old_pre_id] = global_cluster_id

    # --- 4. Calculate centroids for all new clusters ---
    # Invert the map to group old IDs by their new cluster ID
    clusters_to_old_ids = {}
    for old_id, cluster_id in old_pre_id_to_cluster_id.items():
        if cluster_id not in clusters_to_old_ids:
            clusters_to_old_ids[cluster_id] = []
        clusters_to_old_ids[cluster_id].append(old_id)

    # Calculate centroid for each cluster
    cluster_id_to_centroid = {}
    for cluster_id, old_ids in clusters_to_old_ids.items():
        cluster_points = np.array([id_to_loc[id] for id in old_ids])
        centroid = np.mean(cluster_points, axis=0)
        cluster_id_to_centroid[cluster_id] = centroid

    num_clusters = len(cluster_id_to_centroid)
    print(f"Clustered {len(unique_pre_ids)} sites into {num_clusters} new merged pre-sites.")

    # --- 5. Re-build the entire annotation dataset from scratch ---
    final_ids = []
    final_locations = []
    final_types = []
    final_partners = []

    # Maps to track new, consolidated IDs
    cluster_id_to_final_id = {}
    old_post_id_to_final_id = {}

    # Reset final ID counter to 1
    next_final_id = 1 # CREMI IDs start from 1

    # --- 5a. Add all new, merged pre-sites ---
    for cluster_id, location in cluster_id_to_centroid.items():
        final_id = next_final_id
        next_final_id += 1

        cluster_id_to_final_id[cluster_id] = final_id

        final_ids.append(final_id)
        final_locations.append(location)
        final_types.append(b'presynaptic_site')

    # --- 5b. Iterate original partners to add post-sites and new partnerships ---
    for old_pre_id, old_post_id in in_partners:
        # Get the new pre-site ID (from its cluster)
        cluster_id = old_pre_id_to_cluster_id[old_pre_id]
        new_pre_id = cluster_id_to_final_id[cluster_id]

        # Get the new post-site ID (or create it if not seen yet)
        if old_post_id not in old_post_id_to_final_id:
            new_post_id = next_final_id
            next_final_id += 1

            old_post_id_to_final_id[old_post_id] = new_post_id

            # Add this new post-site to our lists
            final_ids.append(new_post_id)
            final_locations.append(id_to_loc[old_post_id]) # Get loc from original map
            final_types.append(b'postsynaptic_site')

        new_post_id = old_post_id_to_final_id[old_post_id]

        # Add the new partnership
        final_partners.append((new_pre_id, new_post_id))

    # --- 6. Write the new HDF5 file ---
    print(f"Writing new merged data to: {output_hdf_path}")
    with h5py.File(output_hdf_path, 'w') as f_out:
        # First, copy the entire 'volumes' group
        with h5py.File(input_hdf_path, 'r') as f_in:
            if 'volumes' in f_in:
                f_in.copy('volumes', f_out)
            else:
                print("Warning: 'volumes' group not found in input, not copied.")

        # Create the new annotations group
        ann_group = f_out.create_group('annotations')
        ann_group.attrs['resolution'] = resolution
        ann_group.attrs['offset'] = offset

        # Convert final lists to numpy arrays
        ids_arr = np.array(final_ids, dtype=np.uint64)
        locations_arr = np.array(final_locations, dtype=np.float32)
        types_arr = np.array(final_types, dtype=h5py.special_dtype(vlen=str))
        partners_arr = np.array(final_partners, dtype=np.uint64)

        # Write the new, merged datasets
        ann_group.create_dataset('ids', data=ids_arr)
        ann_group.create_dataset('locations', data=locations_arr)
        ann_group.create_dataset('types', data=types_arr)

        pre_group = ann_group.create_group('presynaptic_site')
        pre_group.create_dataset('partners', data=partners_arr)

        # --- Comments section (empty) ---
        com_group = ann_group.create_group('comments')
        com_group.create_dataset('target_ids', shape=(0,), dtype=np.uint64)
        com_group.create_dataset('comments', shape=(0,), dtype=h5py.special_dtype(vlen=str))

    print("Merge complete.")
    print(f"Original pre-sites: {len(unique_pre_ids)}")
    print(f"Merged pre-sites:   {len(cluster_id_to_centroid)}")
    print(f"Total post-sites: {len(old_post_id_to_final_id)}")
    print(f"Total partnerships: {len(final_partners)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Merge nearby pre-synaptic sites in a CREMI HDF5 file using DBSCAN."
    )
    parser.add_argument(
        'input_hdf_path',
        type=str,
        help="Path to the original HDF5 file to process."
    )
    parser.add_argument(
        'output_hdf_path',
        type=str,
        help="Path to write the new, merged HDF5 file. (Will not overwrite)"
    )
    parser.add_argument(
        '--distance_threshold',
        type=float,
        required=True,
        help="The maximum 2D (XY) distance (in nanometers) to merge pre-sites *within the same Z-slice*."
    )
    parser.add_argument(
        '--plot_clusters',
        action='store_true',
        help="Generate and save scatter plots of the 2D clusters for each Z-slice."
    )

    args = parser.parse_args()

    if os.path.exists(args.output_hdf_path):
        print(f"Error: Output file already exists: {args.output_hdf_path}")
        print("Please specify a new file path. This script will not overwrite.")
        sys.exit(1)

    merge_pre_sites_slice_aware(
        args.input_hdf_path,
        args.output_hdf_path,
        args.distance_threshold,
        args.plot_clusters
    )

