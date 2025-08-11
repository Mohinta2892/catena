# Run with (save mapping) python your_script_name.py --save_mapping id_map.csv 
#Run with (load saved mapping) python your_script_name.py --load_mapping id_map.csv

import sys
import json  # Added for saving/loading the mapping
import h5py
import skimage
import zarr
import numpy as np
import napari
import argparse
import csv

def load_zarr(inputfilename, dataset):
    f = zarr.open(inputfilename, 'r')
    offset = (0, 0, 0)
    if dataset in f:
        data = f[dataset][:]
        if 'offset' in f[dataset].attrs.keys():
            offset = f[dataset].attrs['offset']
        print(dataset, data.shape, data.dtype)
    else:
        data = None
        print(dataset, 'does not exist')
    # f.close()
    return data, offset


def load_hdf5(inputfilename, dataset):
    f = h5py.File(inputfilename, 'r')
    offset = (0, 0, 0)
    if dataset in f:
        data = f[dataset][:]
        if 'offset' in f[dataset].attrs.keys():
            offset = f[dataset].attrs['offset']
        print(dataset, data.shape, data.dtype)
    else:
        data = None
        print(dataset, 'does not exist')
    f.close()
    return data, offset


def parse_nargs(in_str):
    # Custom function to parse a resolution string into a tuple of integers

    try:
        if isinstance(in_str, list) and len(in_str) == 1:
            in_str = in_str[0]
        parts = in_str.split(',')
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("Resolution must contain exactly three integers separated by commas.")
        return tuple(int(part.strip()) for part in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("Resolution must contain only integers separated by commas.")


def plot_syn(filename, res=(40, 4, 4), save_mapping=None, load_mapping=None):
    """
    Visualizes synaptic data and handles saving/loading of point ID mappings.
    """
    # --- New Feature: Load Mapping ---
    # if load_mapping:
    #     try:
    #         with open(load_mapping, 'r') as f:
    #             loaded_maps = json.load(f)
    #         print(f"✅ Successfully loaded mapping from {load_mapping}")
    #         # You can now use loaded_maps['presynaptic_mapping'] etc. for other purposes if needed.
    #     except FileNotFoundError:
    #         print(f"⚠️ Warning: Mapping file not found at {load_mapping}. Will proceed without loading.")
    #     except (json.JSONDecodeError, KeyError):
    #         print(f"⚠️ Warning: Could not decode or parse JSON from {load_mapping}. Proceeding without loading.")

    # --- Updated Feature: Load Mapping from CSV ---
    if load_mapping:
        loaded_pre_mapping = {}
        loaded_post_mapping = {}
        try:
            with open(load_mapping, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip the header row
                for row in reader:
                    # Unpack row and convert types
                    napari_index, point_type, original_id = int(row[0]), row[1], int(row[2])

                    if point_type == 'presynaptic':
                        loaded_pre_mapping[napari_index] = original_id
                    elif point_type == 'postsynaptic':
                        loaded_post_mapping[napari_index] = original_id

            # You can now use the loaded dictionaries if needed
            print(f"✅ Successfully loaded mapping from CSV file: {load_mapping}")

        except FileNotFoundError:
            print(f"⚠️ Warning: Mapping file not found at {load_mapping}. Will proceed without loading.")
        except (IOError, ValueError, IndexError) as e:
            print(f"⚠️ Warning: Could not parse CSV from {load_mapping}. Reason: {e}. Proceeding without loading.")



    raw, _ = load_hdf5(filename, "volumes/raw")

    cleft, cleft_offset = load_hdf5(filename, "volumes/labels/clefts")

    locs, offsets = load_hdf5(filename, 'annotations/locations')

    locs = [loc + np.array(offsets, dtype=np.float32) for loc in locs]
    # locs = [loc + offset for loc in locs]
    # print(locs)

    partners, offset = load_hdf5(filename, 'annotations/presynaptic_site/partners')
    labels, offset = load_hdf5(filename, 'volumes/labels/neuron_ids')

    annotation_ids, offset = load_hdf5(filename, 'annotations/ids')

    print("\n\n")
    print("--------------------------------------------------------")
    print("Annotation ids:")
    x = np.argwhere(partners[:, 0] == 153)
    y = np.argwhere(partners[:, 1] == 153)
    print("pre", x, partners[x])
    print("post", y, partners[y])

    # Comment out the sys.exit() to continue with visualization
    # sys.exit()

    # Lists to store coordinates and their corresponding original IDs
    pre_sites = []
    post_sites = []
    connectors = []

    # Store the original IDs for each point
    pre_ids = []
    post_ids = []

    for (pre, post) in partners:

        # Get indices of rows where pre matches the first column of annotation_ids
        pre_indices = np.where(annotation_ids == pre)[0]
        post_indices = np.where(annotation_ids == post)[0]

        if len(pre_indices) == 0 or len(post_indices) == 0:
            print(f"Skipping pair (pre: {pre}, post: {post}) - not found in annotation_ids")
            continue

        pre_index = int(pre_indices[0])
        post_index = int(post_indices[0])

        pre_site = locs[pre_index]
        post_site = locs[post_index]

        # convert presyn point annotations from nm to pixels
        pre_temp = []
        for p, r in zip(pre_site, res):
            p = p / r
            pre_temp.append(p)
        # get rid of the transpose when data is already in zyx
        pre_sites.append(np.array([pre_temp[2], pre_temp[1], pre_temp[0]]))
        pre_ids.append(pre)  # Store original pre ID
        print('pre_sites', pre_sites[-1], 'ID:', pre)

        # convert postsyn point annotations from nm to pixels
        post_temp = []
        for p, r in zip(post_site, res):
            p = p / r
            post_temp.append(p)
        # get rid of the transpose when data is already in zyx
        post_sites.append(np.array([post_temp[2], post_temp[1], post_temp[0]]))
        post_ids.append(post)  # Store original post ID
        print('post_sites', post_sites[-1], 'ID:', post)

        # get rid of the transpose when data is already in zyx
        connectors.append([[pre_temp[2], pre_temp[1], pre_temp[0]], [post_temp[2], post_temp[1], post_temp[0]]])
        print("connectors", connectors[-1])

    # --- New Feature: Create and Save Mapping ---
    # Create a mapping from the napari point index (0, 1, 2...) to the original annotation ID
    # We convert numpy integer types to standard Python ints for JSON compatibility.
    pre_mapping = {i: int(pre_id) for i, pre_id in enumerate(pre_ids)}
    post_mapping = {i: int(post_id) for i, post_id in enumerate(post_ids)}

    # if save_mapping:
    #     # Combine mappings into a single dictionary for saving
    #     full_mapping = {
    #         "presynaptic_mapping": pre_mapping,
    #         "postsynaptic_mapping": post_mapping
    #     }
    #     try:
    #         with open(save_mapping, 'w') as f:
    #             json.dump(full_mapping, f, indent=4)
    #         print(f"\n✅ Successfully saved ID mapping to {save_mapping}")
    #     except IOError as e:
    #         print(f"\n❌ Error: Could not save mapping file to {save_mapping}. Reason: {e}")

    if save_mapping:
        try:
            # Use 'w' mode and newline='' for CSV writing
            with open(save_mapping, 'w', newline='') as f:
                writer = csv.writer(f)

                # Write the header row
                writer.writerow(['napari_index', 'point_type', 'original_id'])

                # Write the presynaptic mappings
                for index, original_id in pre_mapping.items():
                    writer.writerow([index, 'presynaptic', original_id])

                # Write the postsynaptic mappings
                for index, original_id in post_mapping.items():
                    writer.writerow([index, 'postsynaptic', original_id])

            print(f"\n✅ Successfully saved ID mapping to {save_mapping} as a CSV file.")
        except IOError as e:
            print(f"\n❌ Error: Could not save mapping CSV file to {save_mapping}. Reason: {e}")

    # --- End of New Feature ---

    v = napari.Viewer()

    # get rid of the transpose when data is already in zyx
    v.add_image(np.transpose(raw, (2, 1, 0)), name="raw")

    if cleft is not None:
        v.add_labels(cleft, name="cleft", opacity=0.7, blending="additive")

    # Create properties dictionaries with original IDs
    pre_properties = {
        'original_id': pre_ids,
        'point_type': ['presynaptic'] * len(pre_ids)
    }

    post_properties = {
        'original_id': post_ids,
        'point_type': ['postsynaptic'] * len(post_ids)
    }

    # Convert coordinates to numpy arrays with original IDs as indices
    pre_sites_array = np.array(pre_sites)
    post_sites_array = np.array(post_sites)

    # Add points using original annotation IDs directly as napari point IDs
    pre_layer = v.add_points(
        pre_sites_array,
        name="pre_syn",
        symbol='triangle_down',
        face_color="red",
        size=10,
        properties=pre_properties,
    )

    post_layer = v.add_points(
        post_sites_array,
        name="post_syn",
        symbol='star',
        face_color="blue",
        size=10,
        properties=post_properties,
    )

    # Create properties for connectors with both pre and post IDs
    connector_properties = {
        'pre_id': pre_ids,
        'post_id': post_ids,
        'connection_type': ['synapse'] * len(connectors)
    }

    connector_layer = v.add_shapes(
        connectors,
        shape_type='line',
        edge_color='cyan',
        face_color='cyan',
        edge_width=5,
        properties=connector_properties
    )

    if labels is not None:
        relabeled_labels, _, _ = skimage.segmentation.relabel_sequential(labels)
        # get rid of the transpose when data is already in zyx
        v.add_labels(np.transpose(relabeled_labels, (2, 1, 0)), opacity=0.7, blending="additive")

    # Print summary information
    print(f"\nVisualization Summary:")
    print(f"Number of presynaptic sites: {len(pre_sites)}")
    print(f"Number of postsynaptic sites: {len(post_sites)}")
    print(f"Number of synaptic connections: {len(connectors)}")
    print(f"Presynaptic IDs range: {min(pre_ids) if pre_ids else 'N/A'} - {max(pre_ids) if pre_ids else 'N/A'}")
    print(f"Postsynaptic IDs range: {min(post_ids) if post_ids else 'N/A'} - {max(post_ids) if post_ids else 'N/A'}")

    # Instructions for user
    print("\nNapari Instructions:")
    print("- NO text overlay - clean visualization!")
    print("- Original annotation IDs are stored in point properties")
    print("- When you SELECT/HIGHLIGHT a point, its original ID will be printed in the console")
    print("- Pre-synaptic sites: red triangles")
    print("- Post-synaptic sites: blue stars")
    print("- Cyan lines connect pre- and post-synaptic partners")
    print("- Click on points to see their original annotation IDs in the console")
    print("- You can also check the 'properties' in napari's point layer controls")

    napari.run()


def main():
    parser = argparse.ArgumentParser(description="Visualize synaptic data with napari and manage ID mappings.")
    parser.add_argument('-gt',
                        default='/media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/train/octo_cube3_calyx_5603_6267_y3254_3890_z7464_8163.hdf',
                        help='Pass the Ground truth h5py file')
    parser.add_argument('-res', nargs='+',
                        default='8,8,8',
                        help='Pass the imaging resolution of this volume, e.g., "8,8,8"')

    # --- New Arguments for Mapping ---
    parser.add_argument('--save_mapping', type=str,
                        help='(Optional) File path to save the napari index to original ID mapping (e.g., mapping.json).')
    parser.add_argument('--load_mapping', type=str,
                        help='(Optional) File path to load a previously saved ID mapping.')

    args = parser.parse_args()

    plot_syn(args.gt,
             res=parse_nargs(args.res),
             save_mapping=args.save_mapping,
             load_mapping=args.load_mapping)


if __name__ == '__main__':
    main()
