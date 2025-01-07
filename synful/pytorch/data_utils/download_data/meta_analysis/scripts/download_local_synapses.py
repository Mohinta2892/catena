"""
USE conda env `synful` in my smohinta's home in cardona-gpu1.
Adapted from Chris Barnes's n5 to mrc script to extend it's functionality to be able to convert n5 to zarr and hdf.
This script will create a .hdf files or .zarr files with raw and annotations for synapse positions in the CREMI format.
Synful expects coordinates in nm and not in pixels. Do not divide by voxel size.

Author: Samia Mohinta
Affiliation: University of Cambridge, UK
"""

from __future__ import annotations

from synful import synapse
import h5py
import os
from skimage import measure, segmentation
import logging
import pandas as pd
import numpy as np

import typing as tp
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path

import dask.array
import zarr
from zarr.n5 import N5FSStore
from tqdm import tqdm


def parse_subvolume(s):
    """
    Does not support negative dims!
    :param s:
    :return:
    """
    out = []
    for dim in s.split(","):
        dim = dim.strip()
        if dim == "...":
            out.append(Ellipsis)
            continue

        parts = []
        for part in dim.split(":"):
            if not part:
                parts.append(None)
            else:
                parts.append(int(part))
        if len(parts) > 2:
            raise ValueError("Each dimension can only have 2 or fewer parts")
        if (
                len(parts) == 2
                and parts[0] is not None
                and parts[1] is not None
                and parts[0] >= parts[1]
        ):
            raise ValueError("Cannot reverse dimensions")
        if ":" in dim:
            out.append(slice(*parts))
        else:
            out.append(parts[0])
    return tuple(out)


def parse_compression(fpath: Path):
    kwargs = dict()
    suff = Path(fpath).suffix
    if suff is None:
        return kwargs
    suff = suff.lower()
    if suff == ".bz2":
        kwargs["compression"] = "bzip2"
    elif suff == ".gz":
        kwargs["compression"] = "gzip"
    if kwargs:
        raise NotImplementedError("Compression is not supported")
    return kwargs


def parse_args(args=None):
    p = ArgumentParser()
    p.add_argument("container", help="path to N5 root")
    p.add_argument("dataset", help="path from N5 root to dataset")
    p.add_argument("outfile", type=Path, help="path to MRC file")
    p.add_argument("zarr", type=int, default=1,
                   help="Pass 1 to convert n5 to zarr, default converts raw to volumes/raw dataset")
    p.add_argument("mrc", type=int, help="Pass 1 to convert n5 to mrc: Credit Chris Barnes")
    p.add_argument(
        "-f", "--force", action="store_true", help="overwrite existing output file"
    )
    p.add_argument(
        "-s",
        "--subvolume",
        type=parse_subvolume,
        help=(
            "subvolume bounds (left-inclusive) as a string like "
            "'0:100,20:60,1000:20000'"
        ),
    )
    return p.parse_args(args)


def get_input(root, dataset):
    store = N5FSStore(root, mode="r")
    arr = zarr.Array(store, dataset, True)
    return dask.array.from_zarr(arr)


def to_zarr(z, outfile, resolution=(8, 8, 8), offset=(0, 0, 0), original_resolution=None):
    z = dask.array.rechunk(z, chunks=(256, 256, 256))  # "auto"

    dask.array.to_zarr(z, outfile, component='volumes/raw', overwrite=True)
    f = zarr.open(outfile, 'a')
    f['volumes/raw'].attrs['offset'] = offset
    f['volumes/raw'].attrs['resolution'] = resolution  # (8, 8, 8)
    if original_resolution is not None:
        f['volumes/raw'].attrs['original_resolution'] = original_resolution


def to_hdf(z, outfile, resolution=(8, 8, 8), offset=(0, 0, 0), original_resolution=None):
    z = dask.array.rechunk(z, chunks=(256, 256, 256))  # "auto"

    dask.array.to_hdf5(outfile, {'volumes/raw': z})
    f = h5py.File(outfile, 'a')
    f['volumes/raw'].attrs['offset'] = offset
    f['volumes/raw'].attrs['resolution'] = resolution  # (8, 8, 8)
    if original_resolution is not None:
        f['volumes/raw'].attrs['original_resolution'] = original_resolution


@contextmanager
def get_output(fpath, shape, dtype, force=False):
    _ = parse_compression(fpath)

    with mrcfile.new_mmap(fpath, shape, mode_from_dtype(dtype), overwrite=force) as f:
        yield f


def main(
        container,
        dataset,
        outfile,
        subvolume: tp.Optional[tuple[slice, ...]] = None,
        force=False,
        parsed=None,
        resolution=(8, 8, 8),
        offset=(0, 0, 0),
        invert=True
):
    z = get_input(container, dataset)
    # Be careful to think of the orientation of the volume, meaning pass zyx if the n5 is oriented as zyx.
    if subvolume:
        z = z[subvolume]

    # try and test it in napari if you see a weird crop in neuroglancer!!
    if invert:
        z = z[:, :, ::-1]  # flipped here to align with neuroglancer vizualization

    # if int(parsed.zarr):
    if outfile.endswith('.zarr'):  # crude
        to_zarr(z, outfile, resolution=resolution, offset=offset)
    elif outfile.endswith(('.hdf', '.hdf5', '.h5')):
        to_hdf(z, outfile, resolution=resolution, offset=offset)

    # if int(parsed.mrc):
    #     with get_output(outfile, z.shape, z.dtype, force) as out:
    #         dask.array.store(z, out.data)


def crop_neuron_path(container, dataset, outfile, skeleton_df, start_point, end_point, crop_range=256,
                     resolution=(8, 8, 8), offset=(0, 0, 0)):
    """
    Crops an EM volume around a neuron's path, defined on the XY plane and spanning between start and end Z coordinates.

    Parameters:
    - skeleton_df: DataFrame with columns 'x', 'y', 'z' for neuron skeleton coordinates.
    - start_point: Index of the start coordinate in the skeleton DataFrame.
    - end_point: Index of the end coordinate in the skeleton DataFrame.
    - crop_range: Range (in pixels) around the coordinates to include in the crop on the XY plane.

    Returns:
    - Cropped 3D numpy array of the EM volume around the neuron's path.
    """

    em_volume = get_input(container, dataset)

    # Extract coordinates for the specified segment
    segment_df = skeleton_df.loc[
        (skeleton_df['treenode_id'] >= start_point) & (skeleton_df['treenode_id'] <= end_point + 1)]
    print(f"len of df: {len(segment_df)}")

    # Crop the EM volume using the determined bounding box
    cropped_volume = []  # list of ndarrays, to stack later
    # start_z = segment_df['z-px'].iloc[0]
    for index, row in tqdm(segment_df.iterrows(), total=len(segment_df)):
        # make a np array to avoid rechunking when stacking dask arrays
        crop_roi = em_volume[row['z-px']:row['z-px'].shift(1), row['y-px'] - crop_range:row['y-px'] + crop_range,
                   row['x-px'] - crop_range:row['x-px'] + crop_range].compute()
        # start_z += 1
        cropped_volume.append(crop_roi)

    cropped_volume = np.stack(cropped_volume, axis=0)

    if resolution != (8, 8, 8):
        forced_res = (8, 8, 8)
    else:
        forced_res = resolution

    # recast to dask array for consistency
    to_zarr(dask.array.from_array(cropped_volume), outfile, original_resolution=resolution, resolution=forced_res,
            offset=offset)
    print(f"Saved file here: {outfile}")


# Function to write synapses into HDF5 format
def write_synapses_into_cremiformat(synapses, filename, offset=None, overwrite=False):
    logging.warning(
        "All orientations must be same, that is if coordinates are saved as XYZ, the EM vol should also be in XYZ")
    id_nr, ids, locations, partners, types = 0, [], [], [], []
    distances = []
    for syn in synapses:
        types.extend(['presynaptic_site', 'postsynaptic_site'])
        # ids.extend([np.array((syn.id_segm_pre, syn.id_segm_post))]) # we use original ids
        ids.extend([id_nr, id_nr + 1])  # generate ids
        # partners.extend([np.array((syn.id_segm_pre, syn.id_segm_post))]) # we use original ids
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


# Function to convert DataFrame to Synapse objects
def convert_to_synapses(df):
    synapses = []
    for _, row in df.iterrows():
        synapses.append(
            # synapse.Synapse(location_pre=(int(row['x_pre_roi']), int(row['y_pre_roi']), int(row['z_pre_roi'])),
            #                location_post=(int(row['x_post_roi']), int(row['y_post_roi']), int(row['z_post_roi'])),
            #                id_segm_pre=int(row['bodyId_pre']), id_segm_post=int(row['bodyId_post'])
            #                )
            synapse.Synapse(location_pre=(int(row['z_pre_roi']), int(row['y_pre_roi']), int(row['x_post_roi'])),
                            location_post=(int(row['z_post_roi']), int(row['y_post_roi']), int(row['x_post_roi'])),
                            id_segm_pre=int(row['bodyId_pre']), id_segm_post=int(row['bodyId_post'])
                            )
        )
    return synapses


def convert_to_pos_synapses(df, invert=True):
    synapses = []
    for _, row in df.iterrows():
        if invert:  # if we invert above the volume to visualise in napari, we have to invert here
            pre_coords = (int(row['x_pre_roi']), int(row['y_pre_roi']), int(row['z_pre_roi']))
            post_coords = (int(row['x_post_roi']), int(row['y_post_roi']), int(row['z_post_roi']))
        else:
            pre_coords = (int(row['z_pre_roi']), int(row['y_pre_roi']), int(row['x_pre_roi']))
            post_coords = (int(row['z_post_roi']), int(row['y_post_roi']), int(row['x_post_roi']))

        # Check if any coordinate is negative
        if all(coord >= 0 for coord in pre_coords + post_coords):
            synapses.append(
                synapse.Synapse(location_pre=pre_coords,
                                location_post=post_coords,
                                id_segm_pre=int(row['bodyId_pre']),
                                id_segm_post=int(row['bodyId_post'])
                                )
            )
    return synapses


# Ensure all arrays have the same length
def check_lengths_and_convert_to_synapses(data_dict, invert=True):
    lengths = [len(v) for v in data_dict.values()]
    if len(set(lengths)) != 1:
        print("Error: All arrays must be of the same length.")
        for key, value in data_dict.items():
            print(f"{key}: {len(value)}")
        raise ValueError("All arrays must be of the same length")

    # data_dict.update({'index': [1]})
    # print(data_dict)
    df = pd.DataFrame(data_dict)
    return convert_to_pos_synapses(df, invert=invert)  # convert_to_synapses may include negative coords


# Use the top 5 densely populated ROIs to create synapses and write to HDF5 files
totaldistances = []
overwrite = True
invert = True  # will reverse the x-axis, will not reverse the coords

top_roi_bboxes = pd.read_csv(
    "/net/fibserver1/raw/smohinta_data/local_synapses/bboxes/octo_syncube3_bbox1.csv")  # COORDS of bbox and Synapses here!

# container = '/net/zstore1/FIBSEM/MR1.4-3/registration/n5'  # which volume to crop from?
container = '/net/zstore1/achampion/alignment/output/1120/v0/n5/FIBSEM_L1120_FullCNS_8x8x8nm'  # which volume to crop from?
dataset = 's0'  # which resolution scale

# we can glob it, but the order is important.
cluster_files = ['/net/fibserver1/raw/smohinta_data/local_synapses/clusters_octo/synapsecube3_points.csv',
                 # '/net/fibserver1/raw/smohinta_data/local_synapses/clusters_mr143/2_LIKELYJ2_points.csv',
                 ]

counter = 5

for idx, roi in top_roi_bboxes.iterrows():
    #
    # if counter < 0:
    #     break

    # We have mirrored the x-axis in the to_hdf and to_zarr functions above
    x_min, x_max = int(roi['x_min']), int(roi['x_max'])
    y_min, y_max = int(roi['y_min']), int(roi['y_max'])
    z_min, z_max = int(roi['z_min']), int(roi['z_max'])

    # x_min, x_max = 8191, 8241
    # y_min, y_max = 7267, 7317
    # z_min, z_max = 12815, 12840

    # synapses
    syn_inputs = pd.read_csv(cluster_files[idx])

    # local n5 expects this zyx, but for neuroglancer x-axis has to be reversed to match catmaid view for MR143
    subvolume = f"{z_min}:{z_max},{y_min}:{y_max},{x_min}:{x_max}"
    ranges = subvolume.split(',')
    formatted_suffix = "z{}_y{}_x{}".format(
        ranges[0].replace(':', '-'),  # Format the first range
        ranges[1].replace(':', '-'),  # Format the second range
        ranges[2].replace(':', '-')  # Format the third range
    )

    # synapse_data = {
    #     'bodyId_pre': syn_inputs['pre_neuron'],
    #     'bodyId_post': syn_inputs['post_neuron'],
    #     'connectorId': syn_inputs['connector_id'],
    #     'x_pre_roi': (syn_inputs['pre_x'] // 8 - x_min).astype(int),  # this is because the files contain data in nm
    #     'x_post_roi': (syn_inputs['post_x'] // 8 - x_min).astype(int),
    #     'y_pre_roi': (syn_inputs['pre_y'] // 8 - y_min).astype(int),
    #     'y_post_roi': (syn_inputs['post_y'] // 8 - y_min).astype(int),
    #     'z_pre_roi': (syn_inputs['pre_z'] // 8 - z_min).astype(int),
    #     'z_post_roi': (syn_inputs['post_z'] // 8 - z_min).astype(int),
    #     'x_connector': (syn_inputs['connector_z'] // 8 - z_min).astype(int),
    #     'y_connector': (syn_inputs['connector_y'] // 8 - y_min).astype(int),
    #     'z_connector': (syn_inputs['connector_x'] // 8 - x_min).astype(int),
    # }

    print("Synful expects coordinates in nm and not in pixels. Do not divide by voxel size")
    synapse_data = {
        'bodyId_pre': syn_inputs['pre_neuron'],
        'bodyId_post': syn_inputs['post_neuron'],
        'connectorId': syn_inputs['connector_id'],
        # because x-axis is reversed, we have to reverse the coords of synapses for correct visualisation
        # but x != z in the coord space. Keep the axis as they are in the csv
        # 'x_pre_roi': (x_max - (syn_inputs['pre_x'] )).astype(int),
        'x_pre_roi': ((x_max - (syn_inputs['connector_x'] // 8)).astype(int)) * 8,
        'x_post_roi': ((x_max - (syn_inputs['post_x'] // 8)).astype(int)) * 8,
        # 'y_pre_roi': (syn_inputs['pre_y'] - y_min).astype(int),
        'y_pre_roi': ((syn_inputs['connector_y'] // 8 - y_min).astype(int)) * 8,
        'y_post_roi': ((syn_inputs['post_y'] // 8 - y_min).astype(int)) * 8,
        # 'z_pre_roi': (syn_inputs['pre_z']  - z_min).astype(int),
        'z_pre_roi': ((syn_inputs['connector_z'] // 8 - z_min).astype(int)) * 8,
        'z_post_roi': ((syn_inputs['post_z'] // 8 - z_min).astype(int)) * 8,
        'x_connector': ((x_max - (syn_inputs['connector_x'] // 8)).astype(int)) * 8,
        'y_connector': ((syn_inputs['connector_y'] // 8 - y_min).astype(int)) * 8,
        'z_connector': ((syn_inputs['connector_z'] // 8 - z_min).astype(int)) * 8,
    }

    print(synapse_data)

    # Check lengths and convert to Synapse objects
    synapses = check_lengths_and_convert_to_synapses(synapse_data, invert=False)
    outputpath = f"/net/fibserver1/raw/smohinta_data/local_synapses/sylee_local_syn_cubes/octo_cube3_{os.path.basename(cluster_files[idx]).split('.csv')[0]}"  # change the path here based on brain region
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    outputfile = f"{outputpath}/cutout{idx + 1}_{x_min}_{x_max}_y{y_min}_{y_max}_z{z_min}_{z_max}.hdf"  # synapses_x{x_min}-{x_max}_y{y_min}-{y_max}_z{z_min}-{z_max}

    distances = write_synapses_into_cremiformat(synapses, outputfile, overwrite=overwrite)

    if outputfile.endswith(('.h5', '.hdf', '.hdf5')):
        h5_file = h5py.File(outputfile, 'a')  # cannot be overwritten

        resolution = (8, 8, 8)  # even though it is 16nm in z, download it as 8 to fool the network
        offset = (0, 0, 0)
        main(
            container, dataset, outputfile, parse_subvolume(subvolume), resolution=resolution, offset=offset,
            invert=invert
            # parsed.force
        )

        # set the resolution in raw and labels
        print(f"resolution (8,8,8) and offset (0,0,0) are hardcoded, remember to change.")
        h5_file["volumes/raw"].attrs["resolution"] = resolution
        h5_file["volumes/raw"].attrs["offset"] = offset

        h5_file.close()

    elif outputfile.endswith('.zarr'):
        h5_file = zarr.open(outputfile, 'a')  # cannot be overwritten

        resolution = (8, 8, 8)  # even though it is 16nm in z, download it as 8 to fool the network
        offset = (0, 0, 0)
        main(
            container, dataset, outputfile, parse_subvolume(subvolume), resolution=resolution, offset=offset
            # parsed.force
        )

        # set the resolution in raw and labels
        print(f"resolution (8,8,8) and offset (0,0,0) are hardcoded, remember to change.")
        h5_file["volumes/raw"].attrs["resolution"] = resolution
        h5_file["volumes/raw"].attrs["offset"] = offset

        totaldistances.extend(distances)
        print(f'ROI {idx + 1}: {len(synapses)} synapses written to {outputfile}')

    # counter -= 1
