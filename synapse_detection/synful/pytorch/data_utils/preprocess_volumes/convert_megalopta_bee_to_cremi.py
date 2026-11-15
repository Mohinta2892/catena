"""
Run with:

python convert_ant_to_cremi.py --resolution 50 8 8 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_FB_1.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_FB_1.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_FB_1.hdf

python convert_ant_to_cremi.py --resolution 50 8 8 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_FB_2.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_FB_2.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_FB_2.hdf

python convert_ant_to_cremi.py --resolution 50 8 8 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_FB_3.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_FB_3.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_FB_3.hdf

python convert_ant_to_cremi.py --resolution 50 8 8 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_FB_4.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_FB_4.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_FB_4.hdf

python convert_ant_to_cremi.py --resolution 50 8 8 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_FB_5.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_FB_5.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_FB_5.hdf

python convert_ant_to_cremi.py --resolution 50 8 8 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_NO_1.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_NO_1.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_NO_1.hdf

python convert_ant_to_cremi.py --resolution 50 8 8 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_NO_2.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_NO_2.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_NO_2.hdf

python convert_ant_to_cremi.py --resolution 50 8 8 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_NO_3.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_NO_3.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_NO_3.hdf

python convert_ant_to_cremi.py --resolution 50 8 8 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_NO_4.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_NO_4.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_NO_4.hdf

python convert_ant_to_cremi.py --resolution 50 8 8 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_NO_5.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_NO_5.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_NO_5.hdf

python convert_ant_to_cremi.py --resolution 50 8 8 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_PB_1.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_PB_1.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_PB_1.hdf

python convert_ant_to_cremi.py --resolution 50 8 8 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_PB_2.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_PB_2.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_PB_2.hdf

python convert_ant_to_cremi.py --resolution 50 8 8 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_PB_3.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_PB_3.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_PB_3.hdf

python convert_ant_to_cremi.py --resolution 50 8 8 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_PB_4.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_PB_4.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_PB_4.hdf

python convert_ant_to_cremi.py --resolution 50 8 8 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_PB_5.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_PB_5.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_PB_5.hdf

Bee:

python convert_ant_to_cremi.py --resolution 50 10 10 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/bee_collaborators_HeinzeLab/megalopta_fb_cube_1.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/bee_collaborators_HeinzeLab/megalopta_fb_cube_1.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/converted_hdfs/megalopta_fb_cube_1.hdf


python convert_ant_to_cremi.py --resolution 50 10 10 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/bee_collaborators_HeinzeLab/megalopta_fb_cube_2.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/bee_collaborators_HeinzeLab/megalopta_fb_cube_2.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/converted_hdfs/megalopta_fb_cube_2.hdf

python convert_ant_to_cremi.py --resolution 50 10 10 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/bee_collaborators_HeinzeLab/megalopta_fb_cube_3.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/bee_collaborators_HeinzeLab/megalopta_fb_cube_3.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/converted_hdfs/megalopta_fb_cube_3.hdf

python convert_ant_to_cremi.py --resolution 50 10 10 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/bee_collaborators_HeinzeLab/megalopta_no_cube_2.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/bee_collaborators_HeinzeLab/megalopta_no_cube_2.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/converted_hdfs/megalopta_no_cube_2.hdf

python convert_ant_to_cremi.py --resolution 50 10 10 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/bee_collaborators_HeinzeLab/megalopta_no_cube_3.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/bee_collaborators_HeinzeLab/megalopta_no_cube_3.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/converted_hdfs/megalopta_no_cube_3.hdf

python convert_ant_to_cremi.py --resolution 50 10 10 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/bee_collaborators_HeinzeLab/megalopta_pb_cube_1.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/bee_collaborators_HeinzeLab/megalopta_pb_cube_1.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/converted_hdfs/megalopta_pb_cube_1.hdf

python convert_ant_to_cremi.py --resolution 50 10 10 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/bee_collaborators_HeinzeLab/megalopta_pb_cube_2.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/bee_collaborators_HeinzeLab/megalopta_pb_cube_2.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/converted_hdfs/megalopta_pb_cube_2.hdf

python convert_ant_to_cremi.py --resolution 50 10 10 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/bee_collaborators_HeinzeLab/megalopta_pb_cube_3.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/bee_collaborators_HeinzeLab/megalopta_pb_cube_3.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/bee/converted_hdfs/megalopta_pb_cube_3.hdf


"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import zarr


def process_annotations(annotations_data, resolution_nm):
    """
    Processes synapse annotations from the input JSON format to CREMI format.

    This function implements logic to assign the *same ID* to pre-synaptic
    sites that share the exact same location.

    Args:
        annotations_data (list): A list of annotation dictionaries from the JSON file.
        resolution_nm (np.ndarray): The voxel resolution in nm as a (Z, Y, X) array.

    Returns:
        tuple: A tuple containing NumPy arrays for IDs, locations, types,
               and partner links, ready for HDF5 storage.
    """

    # Dictionary to store pre_location (nm) -> pre_id mapping
    pre_location_to_id = {}
    next_id = 1  # Start CREMI IDs from 1

    ids = []
    locations = []
    partners = []
    types = []

    resolution_zyx = np.array(resolution_nm)

    for synapse in annotations_data:
        if synapse.get("type") != "line":
            continue

        # --- 1. Process Pre-synaptic site ---
        point_a_pixels_xyz = np.array(synapse['pointA'])
        point_a_pixels_zyx = point_a_pixels_xyz[[2, 1, 0]]
        point_a_nm_zyx = point_a_pixels_zyx * resolution_zyx

        # --- MODIFICATION: Round coordinates ---
        # Round the float coordinates
        pre_loc_rounded_float = np.round(point_a_nm_zyx)
        # Cast to int FOR THE DICTIONARY KEY ONLY
        pre_loc_rounded_int = pre_loc_rounded_float.astype(int)
        # Convert to tuple to be used as a dictionary key
        pre_loc_tuple = tuple(pre_loc_rounded_int)
        # --- END MODIFICATION ---

        # --- 2. Process Post-synaptic site ---
        point_b_pixels_xyz = np.array(synapse['pointB'])
        point_b_pixels_zyx = point_b_pixels_xyz[[2, 1, 0]]
        point_b_nm_zyx = point_b_pixels_zyx * resolution_zyx

        # --- NEW: Round the post-synaptic site as well ---
        post_loc_rounded_float = np.round(point_b_nm_zyx)

        # --- 3. Get or assign ID for pre-synaptic location ---
        if pre_loc_tuple in pre_location_to_id:
            pre_id = pre_location_to_id[pre_loc_tuple]
        else:
            pre_id = next_id
            pre_location_to_id[pre_loc_tuple] = pre_id
            next_id += 1

        # --- 4. Assign new ID for post-synaptic location (always unique) ---
        post_id = next_id
        next_id += 1

        # --- 5. Add all data to lists ---
        # --- MODIFICATION: Save the ROUNDED float coordinates ---
        types.extend(['presynaptic_site', 'postsynaptic_site'])
        ids.extend([pre_id, post_id])
        locations.extend([pre_loc_rounded_float, post_loc_rounded_float])
        partners.append((pre_id, post_id))


    # Convert lists to NumPy arrays for efficient writing to HDF5
    ids_arr = np.array(ids, dtype=np.uint64)
    locations_arr = np.array(locations, dtype=np.uint64)
    types_arr = np.array(types, dtype=h5py.special_dtype(vlen=str))
    partners_arr = np.array(partners, dtype=np.uint64)

    print(f"Processed {len(partners)} synapses.")
    print(f"Total unique pre-sites: {len(pre_location_to_id)}")
    print(f"Total annotation points (pre + post): {len(ids)}")

    return ids_arr, locations_arr, types_arr, partners_arr


def convert_to_cremi(zarr_path, json_path, output_hdf_path, resolution_nm):
    """
    Converts a dataset from Zarr/JSON format to the CREMI HDF5 format.

    Args:
        zarr_path (str): Path to the input Zarr file containing raw EM data.
        json_path (str): Path to the input JSON file with synapse annotations.
        output_hdf_path (str): Path for the output HDF5 file.
        resolution_nm (tuple): A tuple of three floats for voxel resolution
                               (Z, Y, X) in nanometers.
    """
    # Validate input paths
    if not os.path.exists(zarr_path):
        print(f"Error: Input Zarr file not found at {zarr_path}")
        sys.exit(1)
    if not os.path.exists(json_path):
        print(f"Error: Input JSON file not found at {json_path}")
        sys.exit(1)

    print(f"Reading raw data from: {zarr_path}")
    zarr_file = zarr.open(zarr_path, mode='r')
    if 'raw' in zarr_file:
        raw_data = zarr_file['raw'][:]
    else:
        raw_data = zarr_file['RAW'][:]
    print(f"Raw data shape: {raw_data.shape}")

    print(f"Reading annotations from: {json_path}")
    with open(json_path, 'r') as f:
        json_data = json.load(f)

        annotations = json_data.get('ann', [])
        if not annotations:
            annotations = json_data.get('annotations', [])
    print(f"Found {len(annotations)} total annotations.")

    if not len(annotations):
        print(f"No `{len(annotations)}` annotations found. Exiting!")
        sys.exit()

    # Process annotations into the required CREMI format
    print("Processing annotations...")
    ids, locations, types, partners = process_annotations(annotations, resolution_nm)

    print(f"Writing CREMI formatted data to: {output_hdf_path}")
    with h5py.File(output_hdf_path, 'w') as f:
        # --- Volumes section ---
        vol_group = f.create_group('volumes')
        raw_ds = vol_group.create_dataset('raw', data=raw_data, compression='gzip')
        # Add required attributes
        raw_ds.attrs['resolution'] = resolution_nm
        raw_ds.attrs['offset'] = (0, 0, 0)  # Assuming (0,0,0) offset

        # --- Annotations section ---
        ann_group = f.create_group('annotations')
        # Add required attributes
        ann_group.attrs['resolution'] = resolution_nm
        ann_group.attrs['offset'] = (0, 0, 0)  # Assuming (0,0,0) offset

        ann_group.create_dataset('ids', data=ids)
        ann_group.create_dataset('locations', data=locations)
        ann_group.create_dataset('types', data=types)

        # As per the specification, create presynaptic_site/partners
        pre_group = ann_group.create_group('presynaptic_site')
        pre_group.create_dataset('partners', data=partners)

        # --- Comments section (empty as per input) ---
        com_group = f.create_group('comments')
        com_group.create_dataset('target_ids', shape=(0,), dtype=np.uint64)
        com_group.create_dataset('comments', shape=(0,), dtype=h5py.special_dtype(vlen=str))

    print("Conversion complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert Zarr/JSON synapse data to CREMI HDF5 format."
    )
    parser.add_argument(
        'zarr_path',
        type=str,
        help="Path to the input Zarr file (e.g., eciton_fb_cube_1.zarr)."
    )
    parser.add_argument(
        'json_path',
        type=str,
        help="Path to the corresponding JSON annotations file (e.g., eciton_fb_cube_1.json)."
    )
    parser.add_argument(
        'output_path',
        type=str,
        help="Path for the output HDF5 file (e.g., eciton_fb_cube_1.hdf)."
    )
    parser.add_argument(
        '--resolution',
        nargs=3,
        type=float,
        metavar=('Z', 'Y', 'X'),
        required=True,
        help="Voxel resolution in nanometers (e.g., --resolution 40 4 4)."
    )

    args = parser.parse_args()

    # The resolution argument is parsed as a list of strings, convert to tuple of floats
    resolution_tuple = tuple(args.resolution)

    convert_to_cremi(args.zarr_path, args.json_path, args.output_path, resolution_tuple)

