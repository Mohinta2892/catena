import itertools
import json
import sys
import logging

import daisy
import neuroglancer
import numpy as np
from funlib.show.neuroglancer import add_layer, ScalePyramid
from synful import database, synapse
from funlib.persistence import open_ds
import h5py

# neuroglancer.set_server_bind_address('0.0.0.0')
ngid = itertools.count(start=1)


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


def add_synapses(s, directory, source_roi, score_thr=0, radius=30):
    synapses = synapse.read_synapses_in_roi(directory, source_roi)

    pre_sites = []
    post_sites = []
    connectors = []
    below_score = 0
    for syn in synapses:
        if syn.score < score_thr:
            below_score += 1
        else:
            # pre_site = np.flip(syn.location_pre)
            # post_site = np.flip(syn.location_post)

            pre_site = syn.location_pre
            post_site = syn.location_post

            pre_sites.append(neuroglancer.EllipsoidAnnotation(center=pre_site,
                                                              radii=(
                                                                  radius, radius, radius),
                                                              id=next(ngid)))
            post_sites.append(neuroglancer.EllipsoidAnnotation(center=post_site,
                                                               radii=(
                                                                   radius, radius, radius),
                                                               id=next(ngid)))
            connectors.append(
                neuroglancer.LineAnnotation(point_a=pre_site, point_b=post_site,
                                            id=next(ngid)))

    # s.layers['connectors'] = neuroglancer.AnnotationLayer(
    #     dimensions=neuroglancer.CoordinateSpace(
    #         names=["z", "y", "x"],
    #         # names=["x", "y", "z"],
    #         units="nm",
    #         scales=[8, 8, 8]  # [1, 1, 1],
    #     ),
    #     filter_by_segmentation=False,
    #     annotation_color='#00ff00',
    #     annotations=connectors,
    # )
    # s.layers['pre_sites'] = neuroglancer.LocalAnnotationLayer(
    #     dimensions=neuroglancer.CoordinateSpace(
    #         names=["z", "y", "x"],
    #         # names=["x", "y", "z"],
    #         units="nm",
    #         scales=[8, 8, 8]  # [1, 1, 1],
    #     ),        filter_by_segmentation=False,
    #     annotation_color='#00ff00',
    #     annotations=pre_sites,
    # )
    # s.layers['post_sites'] = neuroglancer.AnnotationLayer(
    #     dimensions=neuroglancer.CoordinateSpace(
    #         names=["z", "y", "x"],
    #         # names=["x", "y", "z"],
    #         units="nm",
    #         scales=[8, 8, 8]  # [1, 1, 1],
    #     ),        filter_by_segmentation=False,
    #     annotation_color='#ff00ff',
    #     annotations=post_sites,
    # )

    s.layers.append(
        name="connectors",
        layer=neuroglancer.LocalAnnotationLayer(
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                # names=["x", "y", "z"],
                units="nm",
                scales=[1, 1, 1]  # [1, 1, 1],
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
                # names=["x", "y", "z"],
                units="nm",
                scales=[1, 1, 1]  # [1, 1, 1],
            ),
            annotation_relationships=['connectors'],
            linked_segmentation_layer={'pre_sites': 'segmentation'},
            filter_by_segmentation=['pre_sites'],
            ignore_null_segment_filter=False,
            annotation_properties=[
                neuroglancer.AnnotationPropertySpec(
                    id='color',
                    type='rgb',
                    default='red'  # '#ff0000',  # '#ffff00',
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
                # names=["x", "y", "z"],
                units="nm",
                scales=[1, 1, 1]  # [1, 1, 1],
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

    print(
        'filtered out {}/{} of synapses'.format(below_score,
                                                len(synapses)))
    print('displaying {} synapses'.format(len(post_sites)))


def add_cremi_synapses(s, filename, res=(8, 8, 8)):
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

        # Get indices of rows where pre matches the first column of annotation_ids
        pre_indices = np.where(annotation_ids == pre)[0]
        # Get indices of rows where post matches the second column of annotation_ids
        post_indices = np.where(annotation_ids == post)[0]

        # pre_index = int(np.where(pre == annotation_ids)[0][0])
        # post_index = int(np.where(post == annotation_ids)[0][0])

        # pre_indices = np.where(pre == annotation_ids)[0]
        # post_indices = np.where(post == annotation_ids)[0]

        if len(pre_indices) == 0 or len(post_indices) == 0:
            print(f"Skipping pair (pre: {pre}, post: {post}) - not found in annotation_ids")
            continue

        pre_index = int(pre_indices[0])
        post_index = int(post_indices[0])

        # print(pre_index, post_index)
        pre_site = locs[pre_index]
        post_site = locs[post_index]

        # get rid of the transpose if the data is already transposed
        # post_site = [post_site[2], post_site[1], post_site[0]]
        # pre_site = [pre_site[2], pre_site[1], pre_site[0]]

        # pre_sites.append(neuroglancer.EllipsoidAnnotation(center=pre_site,
        #                                                   radii=(40, 40, 40),
        #                                                   id=next(ngid)))
        post_sites.append(neuroglancer.PointAnnotation(point=post_site,
                                                       # radii=(10, 10, 10),
                                                       id=next(ngid)))

        pre_sites.append(neuroglancer.PointAnnotation(point=pre_site,
                                                      # radii=(10, 10, 10),
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
        name="gt_connectors",
        layer=neuroglancer.LocalAnnotationLayer(
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                # names=["x", "y", "z"],
                units="nm",
                scales=[1, 1, 1],  # [8, 8, 8]
            ),
            annotation_relationships=['gt_connectors'],
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
        name="gt_pre_sites",
        layer=neuroglancer.LocalAnnotationLayer(
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                # names=["x", "y", "z"],
                units="nm",
                scales=[1, 1, 1],  # [8, 8, 8]
            ),
            annotation_relationships=['gt_connectors'],
            linked_segmentation_layer={'gt_pre_sites': 'segmentation'},
            filter_by_segmentation=['pre_sites'],
            ignore_null_segment_filter=False,
            annotation_properties=[
                neuroglancer.AnnotationPropertySpec(
                    id='color',
                    type='rgb',
                    default='#FF0000'  # '#ff0000',  # '#ffff00',
                )
            ],
            annotations=pre_sites
        )
    )

    s.layers.append(
        name="gt_post_sites",
        layer=neuroglancer.LocalAnnotationLayer(
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                # names=["x", "y", "z"],
                units="nm",
                scales=[1, 1, 1],  # [8, 8, 8]
            ),
            annotation_relationships=['gt_connectors'],
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


# def add_cremi_synapses(s, filename):
#     offset = 0
#     locs = open_ds(filename, 'annotations/locations')
#
#     locs = [np.flip(loc) + offset for loc in locs.data]
#     partners = open_ds(filename, 'annotations/presynaptic_site/partners')
#     annotation_ids = open_ds(filename, 'annotations/ids').data
#
#     (pre_sites, post_sites, connectors) = ([], [], [])
#     distances = []
#     for (pre, post) in partners.data:
#         pre_index = int(np.where(pre == annotation_ids)[0][0])
#         post_index = int(np.where(post == annotation_ids)[0][0])
#         pre_site = locs[pre_index]
#         post_site = locs[post_index]
#
#         pre_sites.append(neuroglancer.EllipsoidAnnotation(center=pre_site,
#                                                           radii=(40, 40, 40),
#                                                           id=next(ngid)))
#         post_sites.append(neuroglancer.EllipsoidAnnotation(center=post_site,
#                                                            radii=(40, 40, 40),
#                                                            id=next(ngid)))
#         connectors.append(
#             neuroglancer.LineAnnotation(point_a=pre_site, point_b=post_site,
#                                         id=next(ngid)))
#         dist = np.linalg.norm(np.array(list(pre_site)) - np.array(list(post_site)))
#         distances.append(dist)
#     print(np.mean(distances), np.median(distances))
#     print(len(distances))
#     s.layers['connetors_gt'] = neuroglancer.AnnotationLayer(
#         voxel_size=(1, 1, 1),
#         filter_by_segmentation=False,
#         annotation_color='#ffff00',
#         annotations=connectors,
#     )
#     s.layers['pre_sites_gt'] = neuroglancer.AnnotationLayer(
#         voxel_size=(1, 1, 1),
#         filter_by_segmentation=False,
#         annotation_color='#00ff00',
#         annotations=pre_sites,
#     )
#     s.layers['post_sites_gt'] = neuroglancer.AnnotationLayer(
#         voxel_size=(1, 1, 1),
#         filter_by_segmentation=False,
#         annotation_color='#ff00ff',
#         annotations=post_sites,
#     )


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # trainingfile = '/groups/flyem/home/huangg/cln/exp/cx_smallcubes/synful/5_bf350.h5'
    trainingfile = '/media/samia/DATA/mounts/zstore1/catena/data/SYN_OCTO_CUBE1_RUN2/data_3d/train/cutout1_7827_9021_y5622_6798_z4441_5575.hdf'
    # neuron_ds = '/volumes/labels/neuron_ids'
    # mask_ds = 'volumes/masks/groundtruth'
    raw_ds = 'volumes/raw'
    # neuron = daisy.open_ds(trainingfile, neuron_ds)
    # mask = daisy.open_ds(trainingfile, mask_ds)
    raw = open_ds(trainingfile, raw_ds)

    inferencefile = '/media/samia/DATA/mounts/zstore1/synful/scripts/predict/output_predict_on_train/octo/setup03_octo_hemi/300000/cutout1_7827_9021_y5622_6798_z4441_5575.zarr'
    pred_post_syn = 'volumes/pred_syn_indicator'
    pred_post_dir = 'volumes/pred_partner_vectors'
    pred_post_syn = open_ds(inferencefile, pred_post_syn)
    pred_post_dir = open_ds(inferencefile, pred_post_dir)

    synapsedir = '/media/samia/DATA/mounts/zstore1/synful/scripts/predict/output_predict_on_train/train_syn_cube1_setup03_octo_hemi_300000/syn_cc_thr095000_sum'
    #
    gt_synfile = '/media/samia/DATA/mounts/zstore1/catena/data/SYN_OCTO_CUBE1_RUN2/data_3d/train/cutout1_7827_9021_y5622_6798_z4441_5575.hdf'
    #

    # Create a custom shader for direction vectors.
    nmfactor = 1
    clipvalue = 100
    pred_shader = """void main() {{ emitRGB(vec3((
        clamp(getDataValue(0)*{0}., -{1:.2f}, {1:.2f})+{1:.2f})/{2:.2f}, (
        clamp(getDataValue(1)*{0}., -{1:.2f}, {1:.2f})+{1:.2f})/{2:.2f}, (
        clamp(getDataValue(2)*{0}., -{1:.2f}, {1:.2f})+{1:.2f})/{2:.2f})); }}""".format(
        str(nmfactor), clipvalue, clipvalue * 2)

    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        # add_layer(s, neuron, 'neurons')
        # add_layer(s, mask, 'mask')
        add_layer(s, raw, 'raw')
        add_layer(s, pred_post_syn, 'pred_syn')
        add_layer(s, pred_post_dir, 'pred_dir', shader=pred_shader)
        add_synapses(s, synapsedir, raw.roi, score_thr=10)
        add_cremi_synapses(s, gt_synfile)
    print(viewer.__str__())
