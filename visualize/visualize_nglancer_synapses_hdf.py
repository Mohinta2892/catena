"""
This script can be used to visualise hdf files containing synapse ground truths with neuron segmentation and clefts.
One needs to (un)comment portions of code corresponding to the datasets present in the hdf.
Tested on: CREMI unpadded datasets.
For padded datasets one must pass the correct voxel_offset to the segmentation layers (WIP).

Major Requirements and tested with:
python==3.10
neuroglancer==2.37

Original credits: Adapted and updated based on Julia Buhmann's script under synful_experiments.
Author: Samia Mohinta
Affiliation: Cardona lab, Cambridge UK
"""

import argparse
import itertools
import os

import daisy
import h5py
import zarr
import neuroglancer
import numpy as np
from funlib.persistence import open_ds  # open_ds is broken for hdf file loading;

# neuroglancer.set_server_bind_address('0.0.0.0')
ngid = itertools.count(start=1)


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


def ngLayer(data, res, oo=(0, 0, 0), tt='segmentation'):
    """Adds a local volume layer to the current state of the viewer."""
    return neuroglancer.LocalVolume(data, dimensions=res, volume_type=tt, voxel_offset=oo)


def add(s, a, name, shader=None, data=None):
    """
    This function from old version of this script has been retained for now to have compatibility to neuroglancer <=1.5.
    Will be removed eventually.
    """
    if a is None:
        return

    if shader == 'rgb':
        shader = """void main() { emitRGB(vec3(toNormalized(getDataValue(0)),
        toNormalized(getDataValue(1)), toNormalized(getDataValue(2)))); }"""

    kwargs = {}

    if shader is not None:
        kwargs['shader'] = shader
    data = a.data if data is None else data

    s.layers.append(name=name, layer=neuroglancer.LocalVolume(data, dimensions=dimensions, volume_type='image',
                                                              voxel_offset=(0, 0, 0)))


def open_ds_wrapper(path, ds_name):
    """ funlib.persistence `open_ds` is broken for hdf files.
    Issue raised in Github.
    """
    try:
        return open_ds(path, ds_name)
    except KeyError:
        print('dataset %s could not be loaded' % ds_name)
        return None


def add_cremi_synapses(s, filename, res):
    if filename.lower().endswith((".h5", ".hdf", ".hdf5")):
        # inelegant loading for now till open_ds gets fixed for hdfs (loads zarrs ok)
        locs, offsets = load_hdf5(filename, 'annotations/locations')
    elif filename.lower().endswith(".zarr"):
        locs, offsets = load_zarr(filename, 'annotations/locations')

    offset = 0

    # convert offsets to voxel coords
    offsets = [i / j for i, j in zip(offsets, res)]

    # flip locations if data gets loaded as xyz
    # locs = [np.flip(loc) + offset for loc in locs]
    # currently data is loaded as zyx
    locs = [loc + np.array(offsets, dtype=np.float32) for loc in locs]
    # locs = [loc + offset for loc in locs]
    print(locs)

    if filename.lower().endswith((".h5", ".hdf", ".hdf5")):
        partners, offset = load_hdf5(filename, 'annotations/presynaptic_site/partners')
        annotation_ids, offset = load_hdf5(filename, 'annotations/ids')
    elif filename.lower().endswith(".zarr"):
        partners, offset = load_zarr(filename, 'annotations/presynaptic_site/partners')
        annotation_ids, offset = load_zarr(filename, 'annotations/ids')

    (pre_sites, post_sites, connectors) = ([], [], [])
    for (pre, post) in partners:
        pre_index = int(np.where(pre == annotation_ids)[0][0])
        post_index = int(np.where(post == annotation_ids)[0][0])
        # print(pre_index, post_index)
        pre_site = locs[pre_index]
        post_site = locs[post_index]

        # pre_sites.append(neuroglancer.EllipsoidAnnotation(center=pre_site,
        #                                                   radii=(40, 40, 40),
        #                                                   id=next(ngid)))
        post_sites.append(neuroglancer.EllipsoidAnnotation(center=post_site,
                                                           radii=(40, 40, 40),
                                                           id=next(ngid)))

        pre_sites.append(neuroglancer.PointAnnotation(point=pre_site,
                                                      # radii=(40, 40, 40),
                                                      id=next(ngid)))
        # post_sites.append(neuroglancer.PointAnnotation(point=(100, 100, 100),
        #                                                    # radii=(40, 40, 40),
        #                                                    id=next(ngid)))
        connectors.append(
            neuroglancer.LineAnnotation(point_a=pre_site, point_b=post_site,
                                        id=next(ngid)))

    # print(f"Connectors: {connectors}")
    # print(f"pre_sites: {pre_sites}")
    # print(f"post sites: {post_sites}")

    s.layers.append(
        name="connectors",
        layer=neuroglancer.LocalAnnotationLayer(
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units="nm",
                scales=[1, 1, 1],
            ),
            annotation_relationships=['connectors'],
            # linked_segmentation_layer={'connectors': 'segmentation'},
            # filter_by_segmentation=['connectors'],
            ignore_null_segment_filter=False,
            annotation_properties=[
                neuroglancer.AnnotationPropertySpec(
                    id='color',
                    type='rgb',
                    default='#ffff00',
                )
            ],
            annotations=connectors
        )
    )

    s.layers.append(
        name="pre_sites",
        layer=neuroglancer.LocalAnnotationLayer(
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units="nm",
                scales=[1, 1, 1],
            ),
            annotation_relationships=['connectors'],
            linked_segmentation_layer={'pre_sites': 'segmentation'},
            filter_by_segmentation=['pre_sites'],
            ignore_null_segment_filter=False,
            annotation_properties=[
                neuroglancer.AnnotationPropertySpec(
                    id='color',
                    type='rgb',
                    default='#ff0000',  # '#ffff00',
                )
            ],
            annotations=pre_sites
        )
    )

    s.layers.append(
        name="post_sites",
        layer=neuroglancer.LocalAnnotationLayer(
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units="nm",
                scales=[1, 1, 1],
            ),
            annotation_relationships=['connectors'],
            # linked_segmentation_layer={'post_sites': 'segmentation'},
            # filter_by_segmentation=['post_sites'],
            ignore_null_segment_filter=False,
            annotation_properties=[
                neuroglancer.AnnotationPropertySpec(
                    id='color',
                    type='rgb',
                    default='#ff00ff'  # '#ff00ff',
                )
            ],
            annotations=post_sites
        )
    )


def resolution_tuple(resolution_str):
    # Custom function to parse a resolution string into a tuple of integers

    try:
        parts = resolution_str.split(',')
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("Resolution must contain exactly three integers separated by commas.")
        return tuple(int(part.strip()) for part in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("Resolution must contain only integers separated by commas.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sfile',
                        default="/media/samia/DATA/ark/dan-samia/lsd/funke/fafb/synapses/tencubes/gt_fafb_cubes_zarr/cube_1.zarr",
                        help="Synapse hdf file")

    parser.add_argument('-res', nargs='+', default="40, 4, 4",
                        help="Provide the resolution/voxel_size as comma separated numbers of the dataset in ZYX"
                             " (e.g., `40, 4, 4` ).")

    args = parser.parse_args()

    res = resolution_tuple(args.res)

    dimensions = neuroglancer.CoordinateSpace(
        names=["z", "y", "x"],
        units="nm",
        scales=res,
    )
    filename = args.sfile
    # inelegant loading for now till open_ds gets fixed for hdfs (loads zarrs ok)
    if filename.lower().endswith((".h5", ".hdf", ".hdf5")):
        # load raw
        raw, r_offset = load_hdf5(args.sfile, 'volumes/raw')
        # load neuron_ids
        neuron_ids, n_offset = load_hdf5(args.sfile, '/volumes/labels/neuron_ids')

        # load clefts
        clefts, c_offset = load_hdf5(args.sfile, '/volumes/labels/clefts')

    elif filename.lower().endswith(".zarr"):
        raw, r_offset = load_zarr(args.sfile, 'volumes/raw')

        # load neuron_ids
        neuron_ids, n_offset = load_zarr(args.sfile, '/volumes/labels/neuron_ids')

        # load clefts
        clefts, c_offset = load_zarr(args.sfile, '/volumes/labels/clefts')

    r_offset = [i / j for i, j in zip(r_offset, res)]
    n_offset = [i / j for i, j in zip(n_offset, res)]

    if clefts is not None:
        # replace a np.uint64(-3) to 0 as background, helps to visualise better
        clefts[clefts == np.array(-1).astype(np.uint64)] = 0
        c_offset = [i / j for i, j in zip(c_offset, res)]

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    # add raw
    s.layers.append(name='image', layer=ngLayer(raw, dimensions, tt='image', oo=r_offset))
    # add segmentations
    s.layers.append(name='neuron_ids', layer=ngLayer(neuron_ids, dimensions, tt='segmentation', oo=n_offset))
    if clefts is not None:
        s.layers.append(name='clefts', layer=ngLayer(clefts, dimensions, tt='segmentation', oo=c_offset))

    # add synapses; they are also in the same hdf file
    add_cremi_synapses(s, args.sfile, res)

    # adding states as per the old version of this script.
    # add(s, labels_mask, 'labels_mask')

# change viewer url
# print(viewer.__str__().replace('c04u01.int.janelia.org', '10.40.4.51'))

# for localhost
print(viewer)
