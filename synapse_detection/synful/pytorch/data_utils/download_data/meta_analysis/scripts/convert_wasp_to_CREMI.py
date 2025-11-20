# Convert the wasp-synapse pairs to the CREMI format
"""
Run env (mac os): conda activate /Users/sam/opt/anaconda3/envs/syndaisy/bin/python
python convert_wasp_to_cremi.py 
Hard-coded paths for input now. Output will be saved in the same directory as the input.
"""
import h5py
import numpy as np
import logging
import os
from synful import synapse


def pad_data_and_adjust_locations(raw_em, locations, output_shape, voxel_size_nm=(8, 8, 8)):
    """
    Pad the raw EM data and adjust synapse locations accordingly.
    
    Args:
        raw_em: The raw EM data to be padded
        locations: Synapse locations in nm
        output_shape: Desired output shape after padding
    
    Returns:
        padded_em: Padded EM data
        adjusted_locations: Adjusted locations in nm
    """

    # Get input shape
    input_shape = raw_em.shape

    # Calculate the padding required for each dimension
    pad_sizes = []
    for dim_in, dim_out in zip(input_shape, output_shape):
        pad_total = dim_out - dim_in
        pad_sizes.append((pad_total // 2, pad_total - pad_total // 2))

    # Pad the raw EM data
    padded_em = np.pad(raw_em, pad_sizes, mode='constant', constant_values=0)

    # Adjust locations based on padding
    # Assuming locations are in nm and need to be adjusted by the padding amount
    # The adjustment depends on the resolution/voxel size of your data
    adjusted_locations = locations.copy()

    for i in range(len(pad_sizes)):
        # Add the padding offset (in nm) to each coordinate
        pad_offset_nm = pad_sizes[i][0] * voxel_size_nm[i]
        adjusted_locations[:, i] += pad_offset_nm

    return padded_em, adjusted_locations


def write_padded_cremi_file(input_file, output_shape, offset=None):
    """Read a CREMI format file, pad the data, and write to a new file.
    
    Args:
        input_file: Path to the input HDF file in CREMI format
        output_shape: Desired output shape after padding
    
    Returns:
        output_file: Path to the output padded HDF file
    """
    import h5py
    import os
    import numpy as np

    # Create padded directory
    padded_dir = os.path.join(os.path.dirname(input_file), 'padded')
    os.makedirs(padded_dir, exist_ok=True)

    # Create output filename for padded data
    output_file = os.path.join(padded_dir, f"{os.path.basename(input_file)}")

    # Read the input file
    with h5py.File(input_file, 'r') as h5_file:
        # Read raw EM data
        raw_em = h5_file['volumes/raw'][:]

        # Read annotations
        locations = h5_file['annotations/locations'][:]
        ids = h5_file['annotations/ids'][:] if 'annotations/ids' in h5_file else None
        partners = h5_file['annotations/presynaptic_site/partners'][
                   :] if 'annotations/presynaptic_site/partners' in h5_file else None
        types = h5_file['annotations/types'][:] if 'annotations/types' in h5_file else None

        # Read offset if it exists
        offset = h5_file['annotations'].attrs.get('offset', None)

    # Pad the data and adjust locations
    padded_em, adjusted_locations = pad_data_and_adjust_locations(raw_em, locations, output_shape)

    # Write to output file
    with h5py.File(output_file, 'w') as h5_file:
        # Create datasets
        h5_file.create_dataset('volumes/raw', data=padded_em, compression='gzip')
        h5_file.create_dataset('annotations/locations', data=adjusted_locations, compression='gzip')

        if ids is not None:
            h5_file.create_dataset('annotations/ids', data=ids, compression='gzip')

        if partners is not None:
            h5_file.create_dataset('annotations/presynaptic_site/partners', data=partners, compression='gzip')

        if types is not None:
            h5_file.create_dataset('annotations/types', data=types, compression='gzip')

        # Set offset if it exists
        if offset is not None:
            h5_file['annotations'].attrs['offset'] = offset
        else:
            h5_file['annotations'].attrs['offset'] = (0, 0, 0)

        # set these important to run synful
        h5_file['volumes/raw'].attrs['offset'] = (0, 0, 0)
        h5_file['volumes/raw'].attrs['resolution'] = (8, 8, 8)  # Resolution in nm

    print(f"Padded data saved to: {output_file}")
    return output_file


def write_synapses_into_cremiformat_same_preid(synapses, filename, offset=None, overwrite=False):
    logging.warning(
        "All orientations must be same, that is if coordinates are saved as XYZ, the EM vol should also be in XYZ")

    # Dictionary to store pre_location -> id mapping
    pre_location_to_id = {}
    next_id = 0

    id_nr, ids, locations, partners, types = 0, [], [], [], []
    distances = []

    for syn in synapses:
        # Convert pre_location to tuple for dictionary key
        pre_loc_tuple = tuple(syn.location_pre)

        # Get or assign ID for pre-synaptic location
        if pre_loc_tuple in pre_location_to_id:
            pre_id = pre_location_to_id[pre_loc_tuple]
        else:
            pre_id = next_id
            pre_location_to_id[pre_loc_tuple] = pre_id
            next_id += 1

        # Assign new ID for post-synaptic location
        post_id = next_id
        next_id += 1

        types.extend(['presynaptic_site', 'postsynaptic_site'])
        ids.extend([pre_id, post_id])
        partners.extend([np.array((pre_id, post_id))])

        assert syn.location_pre is not None and syn.location_post is not None
        locations.extend([np.array(syn.location_pre), np.array(syn.location_post)])

        dist = np.linalg.norm(
            np.array(list(syn.location_pre), dtype=np.float32) - np.array(list(syn.location_post), dtype=np.float32))
        distances.append(dist)

    print('number of synapses in file {}'.format(len(synapses)))
    print('Distances: median {}, mean {}, max {}, min {}'.format(np.median(distances), np.mean(distances),
                                                                 np.max(distances),
                                                                 np.min(distances)))

    if filename.endswith(('.h5', '.hdf', '.hdf5')):
        if overwrite:
            h5_file = h5py.File(filename, 'w')
        else:
            h5_file = h5py.File(filename, 'a')

        dset = h5_file.create_dataset('annotations/ids', data=ids, compression='gzip')
        dset = h5_file.create_dataset('annotations/locations', data=np.stack(locations, axis=0).astype(np.float32),
                                      compression='gzip')
        dset = h5_file.create_dataset('annotations/presynaptic_site/partners',
                                      data=np.stack(partners, axis=0).astype(np.uint32), compression='gzip')
        dset = h5_file.create_dataset('annotations/types', data=np.array(types, dtype='S'), compression='gzip')

        if offset is not None:
            h5_file['annotations'].attrs['offset'] = offset
        else:
            h5_file['annotations'].attrs['offset'] = (0, 0, 0)
        h5_file.close()
    elif filename.endswith('.zarr'):
        if overwrite:
            h5_file = zarr.open(filename, 'w')
        else:
            h5_file = zarr.open(filename, 'a')

        dset = h5_file.create_dataset('annotations/ids', data=ids, compression='gzip')
        dset = h5_file.create_dataset('annotations/locations', data=np.stack(locations, axis=0).astype(np.float32),
                                      compression='gzip')
        dset = h5_file.create_dataset('annotations/presynaptic_site/partners',
                                      data=np.stack(partners, axis=0).astype(np.uint32), compression='gzip')
        dset = h5_file.create_dataset('annotations/types', data=np.array(types, dtype='S'), compression='gzip')

        if offset is not None:
            h5_file['annotations'].attrs['offset'] = offset

    print('File written to {}'.format(filename))
    return distances


# Function to write synapses into HDF5 format
def write_synapses_into_cremiformat(synapses, filename, offset=None, overwrite=False, image_shape=None):
    logging.warning(
        "All orientations must be same, that is if coordinates are saved as XYZ, the EM vol should also be in XYZ")
    id_nr, ids, locations, partners, types = 0, [], [], [], []
    distances = []
    for syn in synapses:
        types.extend(['presynaptic_site', 'postsynaptic_site'])
        ids.extend([id_nr, id_nr + 1])  # generate ids
        partners.extend([np.array((id_nr, id_nr + 1))])  # generate ids

        assert syn.location_pre is not None and syn.location_post is not None
        locations.extend([np.array(syn.location_pre), np.array(syn.location_post)])
        id_nr += 2
        dist = np.linalg.norm(
            np.array(list(syn.location_pre), dtype=np.float32) - np.array(list(syn.location_post), dtype=np.float32))
        distances.append(dist)

    print('number of synapses in file {}'.format(len(synapses)))
    print('Distances: median {}, mean {}, max {}, min {}'.format(np.median(distances), np.mean(distances),
                                                                 np.max(distances),
                                                                 np.min(distances)))

    # Save statistics to JSON file
    import json
    stats = {
        "num_synapses": len(synapses),
        "distances": {
            "median": float(np.median(distances)),
            "mean": float(np.mean(distances)),
            "max": float(np.max(distances)),
            "min": float(np.min(distances))
        }
    }

    if image_shape is not None:
        stats["raw_shape"] = {
            "z": int(image_shape[0]),
            "y": int(image_shape[1]),
            "x": int(image_shape[2])
        }

    json_filename = filename.replace('.hdf', '.json').replace('.h5', '.json')
    with open(json_filename, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Statistics saved to {json_filename}")

    if filename.endswith(('.h5', '.hdf', '.hdf5')):
        if overwrite:
            h5_file = h5py.File(filename, 'w')
        else:
            h5_file = h5py.File(filename, 'a')

        dset = h5_file.create_dataset('annotations/ids', data=ids, compression='gzip')
        dset = h5_file.create_dataset('annotations/locations', data=np.stack(locations, axis=0).astype(np.float32),
                                      compression='gzip')
        dset = h5_file.create_dataset('annotations/presynaptic_site/partners',
                                      data=np.stack(partners, axis=0).astype(np.uint32), compression='gzip')
        dset = h5_file.create_dataset('annotations/types', data=np.array(types, dtype='S'), compression='gzip')

        if offset is not None:
            h5_file['annotations'].attrs['offset'] = (0, 0, 0)
        h5_file.close()

    print('File written to {}'.format(filename))
    return distances


def convert_wasp_to_synapses(pre_data, post_data, image_shape=None):
    """
    Convert WASP pre and post synaptic data to Synapse objects
    
    Args:
        pre_data: Array of pre-synaptic coordinates (z, y, x)
        post_data: Array of post-synaptic data with format [pre_id, z, y, x]
        image_shape: Shape of the raw image to check bounds (z, y, x)
        
    Returns:
        List of Synapse objects
    """
    synapses = []

    # Create a mapping from pre_id to pre-synaptic coordinates
    pre_id_to_coords = {}
    for i, coords in enumerate(pre_data):
        pre_id_to_coords[i] = tuple(coords)

    # Create synapse objects by matching pre and post synaptic locations
    valid_count = 0
    out_of_bounds_count = 0

    # Generate unique IDs for post-synaptic sites
    # Start with a high number to avoid overlap with pre IDs
    post_id_counter = len(pre_data) + 1000

    for post_entry in post_data:
        pre_id = int(post_entry[0])
        post_coords = tuple(post_entry[1:])

        if pre_id in pre_id_to_coords:
            pre_coords = pre_id_to_coords[pre_id]

            # Check if coordinates are within bounds of the image and not zero
            if image_shape is not None:
                # Check bounds (must be within image and not on the edge)
                pre_in_bounds = all(0 < c < s - 1 for c, s in zip(pre_coords, image_shape))
                post_in_bounds = all(0 < c < s - 1 for c, s in zip(post_coords, image_shape))

                # Check that no coordinate is zero in either pixel or nm space
                pre_has_zeros = any(c == 0 for c in pre_coords) or any(c * 8 == 0 for c in pre_coords)
                post_has_zeros = any(c == 0 for c in post_coords) or any(c * 8 == 0 for c in post_coords)

                if not (pre_in_bounds and post_in_bounds) or pre_has_zeros or post_has_zeros:
                    out_of_bounds_count += 1
                    continue  # Skip this synapse

            # Convert to nm resolution (multiply by 8,8,8)
            pre_coords_nm = tuple(c * 8 for c in pre_coords)
            post_coords_nm = tuple(c * 8 for c in post_coords)

            # Create a synapse object using the synful.synapse module with unique post ID
            synapses.append(
                synapse.Synapse(
                    location_pre=pre_coords_nm,
                    location_post=post_coords_nm,
                    id_segm_pre=pre_id,
                    id_segm_post=post_id_counter  # Unique ID for post-synaptic site
                )
            )
            post_id_counter += 1
            valid_count += 1

    print(f"Total valid synapses: {valid_count}")
    print(f"Skipped out-of-bounds synapses: {out_of_bounds_count}")

    return synapses


# Main execution
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # You may need to change the following paths according to your file structure.
    path_image = '/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/wasp/WASPSYN_Dataset/training_set/train_sample3_vol4/img_zyx_1920-2336_4832-5248_6528-6944.h5'
    path_label = '/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Phd_Data/synapse_detection/wasp/WASPSYN_Dataset/training_set/train_sample3_vol4/syns_zyx_1920-2336_4832-5248_6528-6944.h5'

    # Extract offset from filename
    offset_zyx = path_label.split('/')[-1].split('_')
    offset_z = int(offset_zyx[2].split('-')[0])
    offset_y = int(offset_zyx[3].split('-')[0])
    offset_x = int(offset_zyx[4].split('-')[0])
    offset = (offset_z, offset_y, offset_x)

    # Read the data
    f_image = h5py.File(path_image, 'r')
    print(f_image.keys())
    image = f_image['main'][:]
    print(f"Image shape: {image.shape}")
    f_image.close()

    f_label = h5py.File(path_label, 'r')
    print(f_label.keys())
    pre = f_label['pre'][:]
    post = f_label['post'][:]
    f_label.close()

    # Print some information about the data
    print(f"Pre-synaptic data shape: {pre.shape}")
    print(f"Sample pre-synaptic data:\n{pre[:10, :]}")

    print(f"Post-synaptic data shape: {post.shape}")
    print(f"Sample post-synaptic data:\n{post[:10, :]}")

    # Adjust coordinates to be relative to the subvolume
    pre_adjusted = pre - [offset_z, offset_y, offset_x]
    post_adjusted = post.copy()
    post_adjusted[:, 1:] = post[:, 1:] - [offset_z, offset_y, offset_x]

    # Convert to synapse objects, checking bounds against image shape
    synapses = convert_wasp_to_synapses(pre_adjusted, post_adjusted, image_shape=image.shape)
    print(f"Created {len(synapses)} synapse objects")
    print(f"Sample synapse object:\n{synapses[0:10]}")

    # Extract the volume number from the path
    vol_num = path_label.split('vol')[1][0]

    # Write to CREMI format
    output_path = os.path.join(os.path.dirname(path_label),
                               f"train_vol{vol_num}_{os.path.basename(path_label).replace('.h5', '_cremi.hdf')}")

    # Add raw image data to the output file
    with h5py.File(output_path, 'w') as h5_file:
        # Create volumes group and raw dataset
        h5_file.create_dataset('volumes/raw', data=image, compression='gzip')
        # Add offset attribute if needed - convert to nm resolution
        h5_file['volumes/raw'].attrs['offset'] = (0, 0, 0)
        h5_file['volumes/raw'].attrs['resolution'] = (8, 8, 8)  # Resolution in nm

    # Now add the synapse annotations
    distances = write_synapses_into_cremiformat(synapses, output_path, offset=tuple(o * 8 for o in offset),
                                                overwrite=False, image_shape=image.shape)

    # Create a second output using the same_preid method
    output_path_same_preid = output_path.replace('_cremi.hdf', '_cremi_same_preid.hdf')

    # Add raw image data to the output file
    with h5py.File(output_path_same_preid, 'w') as h5_file:
        # Create volumes group and raw dataset
        h5_file.create_dataset('volumes/raw', data=image, compression='gzip')
        # Add offset attribute if needed - convert to nm resolution
        h5_file['volumes/raw'].attrs['offset'] = (0, 0, 0)
        h5_file['volumes/raw'].attrs['resolution'] = (8, 8, 8)  # Resolution in nm

    distances_same_preid = write_synapses_into_cremiformat_same_preid(synapses, output_path_same_preid,
                                                                      offset=None,
                                                                      overwrite=False)

    print(f"Conversion complete. Output files:\n{output_path}\n{output_path_same_preid}")

    # Create a third output: read the same_preid and pad the EM and adjust locations and save
    # Read the same_preid output file
    padded_output_path = write_padded_cremi_file(output_path_same_preid, output_shape=(600, 600, 600), offset=None)
