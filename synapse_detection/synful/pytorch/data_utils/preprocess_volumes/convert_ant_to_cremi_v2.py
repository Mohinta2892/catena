#!/usr/bin/env python

"""
Converts Zarr/JSON synapse data to CREMI HDF5 format,
correctly applying the 'roi_offset' found in the JSON file.

This version fixes two critical issues from the original script:
1.  It reads the `roi_offset` (e.g., [Z, Y, X] pixel offset) from the
    JSON file and adds it to the annotation coordinates. This ensures
    the annotations are correctly placed within the full Zarr volume.
2.  It saves the final nanometer locations as `np.float64` instead of
    `np.uint64` to preserve precision, which is essential for
    correct visualization.

Example usage:

python convert_ant_to_cremi_v2.py --resolution 50 8 8 --offset 25 150 150  \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_FB_1.zarr  \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_FB_1.json  \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_FB_1.hdf

python convert_ant_to_cremi_v2.py --resolution 50 8 8  --offset 25 150 150 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_FB_2.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_FB_2.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_FB_2.hdf

python convert_ant_to_cremi_v2.py --resolution 50 8 8 --offset 25 150 150 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_FB_3.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_FB_3.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_FB_3.hdf

python convert_ant_to_cremi_v2.py --resolution 50 8 8 --offset 25 150 150 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_FB_4.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_FB_4.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_FB_4.hdf

python convert_ant_to_cremi_v2.py --resolution 50 8 8  --offset 25 150 150 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_FB_5.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_FB_5.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_FB_5.hdf

python convert_ant_to_cremi_v2.py --resolution 50 8 8 --offset 25 150 150 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_NO_1.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_NO_1.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_NO_1.hdf

-- no synapses, negative space ---
python convert_ant_to_cremi_v2.py --resolution 50 8 8 --offset 25 150 150 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_NO_2.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_NO_2.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_NO_2.hdf

python convert_ant_to_cremi_v2.py --resolution 50 8 8 --offset 25 150 150 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_NO_3.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_NO_3.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_NO_3.hdf

python convert_ant_to_cremi_v2.py --resolution 50 8 8 --offset 25 150 150 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_NO_4.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_NO_4.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_NO_4.hdf

python convert_ant_to_cremi_v2.py --resolution 50 8 8 --offset 25 150 150 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_NO_5.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_NO_5.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_NO_5.hdf

python convert_ant_to_cremi_v2.py --resolution 50 8 8  --offset 25 150 150 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_PB_1.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_PB_1.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_PB_1.hdf

python convert_ant_to_cremi_v2.py --resolution 50 8 8 --offset 25 150 150 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_PB_2.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_PB_2.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_PB_2.hdf

-- no synapses, negative space ---
python convert_ant_to_cremi_v2.py --resolution 50 8 8 --offset 25 150 150 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_PB_3.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_PB_3.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_PB_3.hdf

python convert_ant_to_cremi_v2.py --resolution 50 8 8 --offset 25 150 150 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_PB_4.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_PB_4.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_PB_4.hdf

python convert_ant_to_cremi_v2.py --resolution 50 8 8 --offset 25 150 150 \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/zarrs/eciton_PB_5.zarr \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/sam_files_new/jsons/eciton_PB_5.json \
/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/ant/converted_hdfs/eciton_PB_5.hdf



"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import zarr


def process_annotations(annotations_data, resolution_nm, roi_offset_zyx):
    """
    Processes synapse annotations from the input JSON format to CREMI format.

    This function implements the new logic:
    1.  Adds the `roi_offset_zyx` (in pixels) to the local annotation
        coordinates (`pointA`, `pointB`).
    2.  Converts these *global* pixel coordinates to nanometers.
    3.  Implements logic to assign the *same ID* to pre-synaptic
        sites that share the exact same rounded (nm) location.

    Args:
        annotations_data (list): A list of annotation dictionaries.
        resolution_nm (np.ndarray): The voxel resolution in nm as a (Z, Y, X) array.
        roi_offset_zyx (np.ndarray): The (Z, Y, X) pixel offset of the
                                     annotated ROI within the Zarr volume.

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
    # roi_offset_zyx is already a np.array

    for synapse in annotations_data:
        if synapse.get("type") != "line":
            continue

        # --- 1. Process Pre-synaptic site ---
        # point_a_pixels_xyz = np.array(synapse['pointA'])  # [X, Y, Z]
        # point_a_pixels_local_zyx = point_a_pixels_xyz[[2, 1, 0]]  # [Z, Y, X]
        point_a_pixels_local_zyx = np.array(synapse['pointA'])  # Assumed [Z, Y, X]

        # --- FIX: Add the ROI offset to get global pixel coordinates ---
        point_a_pixels_global_zyx = point_a_pixels_local_zyx + roi_offset_zyx

        # Convert global pixel coordinates to nanometers
        point_a_nm_zyx = point_a_pixels_global_zyx * resolution_zyx

        # Round the float coordinates
        pre_loc_rounded_float = np.round(point_a_nm_zyx)
        # Cast to int FOR THE DICTIONARY KEY ONLY
        pre_loc_rounded_int = pre_loc_rounded_float.astype(int)
        # Convert to tuple to be used as a dictionary key
        pre_loc_tuple = tuple(pre_loc_rounded_int)

        # --- 2. Process Post-synaptic site ---
        # point_b_pixels_xyz = np.array(synapse['pointB'])  # [X, Y, Z]
        # point_b_pixels_local_zyx = point_b_pixels_xyz[[2, 1, 0]]  # [Z, Y, X]
        point_b_pixels_local_zyx = np.array(synapse['pointB'])  # Assumed [Z, Y, X]

        # --- FIX: Add the ROI offset to get global pixel coordinates ---
        point_b_pixels_global_zyx = point_b_pixels_local_zyx + roi_offset_zyx

        # Convert global pixel coordinates to nanometers
        point_b_nm_zyx = point_b_pixels_global_zyx * resolution_zyx

        # Round the post-synaptic site as well
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
        # Save the ROUNDED float coordinates
        types.extend(['presynaptic_site', 'postsynaptic_site'])
        ids.extend([pre_id, post_id])
        locations.extend([pre_loc_rounded_float, post_loc_rounded_float])
        partners.append((pre_id, post_id))

    # Convert lists to NumPy arrays for efficient writing to HDF5
    ids_arr = np.array(ids, dtype=np.uint64)

    # --- CRITICAL FIX: Save locations as float64 to preserve nm precision ---
    # locations_arr = np.array(locations, dtype=np.float64)
    locations_arr = np.array(locations, dtype=np.uint64)  # precision not needed

    types_arr = np.array(types, dtype=h5py.special_dtype(vlen=str))
    partners_arr = np.array(partners, dtype=np.uint64)

    print(f"Processed {len(partners)} synapses.")
    print(f"Total unique pre-sites: {len(pre_location_to_id)}")
    print(f"Total annotation points (pre + post): {len(ids)}")

    return ids_arr, locations_arr, types_arr, partners_arr


def convert_to_cremi(
        zarr_path, json_path, output_hdf_path, resolution_nm, offset_zyx
):
    """
    Converts a dataset from Zarr/JSON format to the CREMI HDF5 format.
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
    if 'RAW' in zarr_file:
        raw_data = zarr_file['RAW']
    elif 'raw' in zarr_file:
        raw_data = zarr_file['raw']
    else:
        print(f"Error: Could not find 'RAW' or 'raw' dataset in {zarr_path}")
        sys.exit(1)
    print(f"Raw data shape: {raw_data.shape}")

    print(f"Reading annotations from: {json_path}")
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # --- RE-IMPLEMENTED LOGIC: Get annotations list, no offset lookup ---
    annotations_list = None
    if 'annotations' in json_data and isinstance(json_data['annotations'], list):
        # Format: {"annotations": [...]}
        annotations_list = json_data['annotations']
    elif 'ann' in json_data and isinstance(json_data['ann'], list):
        # Original format: {"ann": [...]}
        annotations_list = json_data['ann']
    elif isinstance(json_data, list):
        # Fallback for JSON being just a list
        annotations_list = json_data
    # --- END RE-IMPLEMENTED LOGIC ---

    if not annotations_list:
        print(f"Error: Could not find synapse list in {json_path}.")
        print("Looked for keys 'annotations', 'ann', or a root-level list.")
        sys.exit(1)

    # Convert the user-provided offset tuple to a numpy array
    roi_offset_zyx_np = np.array(offset_zyx)
    print(f"Found {len(annotations_list)} total annotations.")
    print(f"Using externally provided 'offset' (Z,Y,X): {roi_offset_zyx_np}")

    # Process annotations into the required CREMI format
    print("Processing annotations...")
    ids, locations, types, partners = process_annotations(
        annotations_list,
        resolution_nm,
        roi_offset_zyx_np
    )

    print(f"Writing CREMI formatted data to: {output_hdf_path}")
    with h5py.File(output_hdf_path, 'w') as f:
        # --- Volumes section ---
        vol_group = f.create_group('volumes')
        # Read/write raw data chunk by chunk
        raw_ds = vol_group.create_dataset(
            'raw',
            shape=raw_data.shape,
            dtype=raw_data.dtype,
            chunks=raw_data.chunks,
            compression='gzip'
        )
        raw_ds[:] = raw_data[:]

        # Add required attributes
        raw_ds.attrs['resolution'] = resolution_nm
        raw_ds.attrs['offset'] = (0, 0, 0)  # Offset is 0, coords are now global

        # --- Annotations section ---
        ann_group = f.create_group('annotations')
        # Add required attributes
        ann_group.attrs['resolution'] = resolution_nm
        ann_group.attrs['offset'] = (0, 0, 0)  # Offset is 0, coords are now global

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
        description="Convert Zarr/JSON synapse data to CREMI HDF5 format "
                    "with a user-provided '--offset'."
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
        help="Voxel resolution in nanometers (e.g., --resolution 50 8 8)."
    )

    # --- RE-IMPLEMENTED LOGIC: Added --offset argument ---
    parser.add_argument(
        '--offset',
        nargs=3,
        type=float,
        metavar=('Z', 'Y', 'X'),
        required=True,
        help="Pixel offset of the annotated ROI (e.g., --offset 25 150 150)."
    )
    # --- END RE-IMPLEMENTED LOGIC ---

    args = parser.parse_args()

    # Convert arguments to tuples
    resolution_tuple = tuple(args.resolution)
    offset_tuple = tuple(args.offset)

    convert_to_cremi(
        args.zarr_path,
        args.json_path,
        args.output_path,
        resolution_tuple,
        offset_tuple  # Pass the new offset
    )
