"""
Because this is TF1.x please run train.py like CUDA_VISIBLE_DEVICES=1,2 python train.py
Otherwise it will spawn the graph across the all cards that are visible to it, but use only one as it does not support distributed training on multiple cards.
"""
from __future__ import print_function

import json
import math
import os
import pdb
import sys
import logging

try:
    import absl.logging

    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)

import gunpowder as gp
import numpy as np
import daisy
from generate_network import mknet
from synful.gunpowder import AddPartnerVectorMap, Hdf5PointsSource

# CREMI specific, download data from: www.cremi.org
data_dir = '/zstore/catena/data/SYNPAPER_FIBSEM_CLAHE/cross_4/data_3d/train'
data_dir_syn = data_dir
samples = [
#'WASPSYN23_train_vol0_syns_zyx_2217-2617_4038-4448_6335-6735_cremi_same_preid',
#'WASPSYN23_train_vol1_syns_zyx_3680-4096_2944-3360_4448-4864_cremi_same_preid',
#'WASPSYN23_train_vol2_syns_zyx_3776-4192_6048-6464_9248-9664_cremi_same_preid', # removed 10202 for neurips training
#'WASPSYN23_train_vol3_syns_zyx_5152-5568_3168-3584_8384-8800_cremi_same_preid',
#'WASPSYN23_train_vol4_syns_zyx_1920-2336_4832-5248_6528-6944_cremi_same_preid',
'HEMIBRAIN_synapses_x12437-13037_y27229-27829_z17176-17776',
'HEMIBRAIN_synapses_x15035-15635_y28559-29159_z9602-10202',
'HEMIBRAIN_synapses_x15082-15682_y31050-31650_z14555-15155', # removed 10202 for neurips training
'HEMIBRAIN_synapses_x21786-22386_y28978-29578_z18787-19387',
'HEMIBRAIN_synapses_x27262-27862_y31539-32139_z17577-18177',
'OCTO_cube1_v2_8083_8765_y5878_6542_z4697_5319',
'OCTO_cube3_calyx_v2_5603_6267_y3254_3890_z7464_8163',
'OCTO_cube2_v2_12485_13164_y6231_6901_z3971_4640',
'MANC_synapses_x14200-14800_y33000-33600_z44600-45200',
'MANC_synapses_x20200-20800_y34200-34800_z32200-32800',
'MANC_synapses_x23400-24000_y24000-24600_z14400-15000'

]
# cremi_roi = gp.Roi(np.array((1520, 3644, 3644)), np.array((5000, 5000, 5000)))
hemi_roi_1 = gp.Roi(np.array((0, 0, 0)), np.array((4800, 4800, 4800)))
hemi_roi_3 = gp.Roi(np.array((0, 0, 0)), np.array((4800, 4800, 4800)))
hemi_roi_4 = gp.Roi(np.array((0, 0, 0)), np.array((4800, 4800, 4800)))
hemi_roi_5 = gp.Roi(np.array((0, 0, 0)), np.array((4800, 4800, 4800))) #(3328, 3328, 3328)
hemi_roi_7 = gp.Roi(np.array((0, 0, 0)), np.array((4800, 4800, 4800)))
#hemi_roi_9 = gp.Roi(np.array((0, 0, 0)), np.array((4800, 4800, 4800)))
#hemi_roi_11 = gp.Roi(np.array((0, 0, 0)), np.array((4800, 4800, 4800)))
#hemi_roi_13 = gp.Roi(np.array((0, 0, 0)), np.array((4800, 4800, 4800)))
#hemi_roi_15 = gp.Roi(np.array((0, 0, 0)), np.array((4800, 4800, 4800)))
#hemi_roi_17 = gp.Roi(np.array((0, 0, 0)), np.array((4800, 4800, 4800)))
hemi_roi_19 = gp.Roi(np.array((0, 0, 0)), np.array((4976, 5312, 5456)))
hemi_roi_21 = gp.Roi(np.array((0, 0, 0)), np.array((5592, 5088, 5312)))
hemi_roi_23 = gp.Roi(np.array((0, 0, 0)), np.array((5352, 5360, 5432)))
hemi_roi_25 = gp.Roi(np.array((0, 0, 0)), np.array((4800, 4800, 4800)))
hemi_roi_27 = gp.Roi(np.array((0, 0, 0)), np.array((4800, 4800, 4800)))
hemi_roi_29 = gp.Roi(np.array((0, 0, 0)), np.array((4800, 4800, 4800)))
#hemi_roi_2 = gp.Roi(np.array((0, 0, 0)), np.array((13160, 17032, 12048)))
#hemi_roi_3 = gp.Roi(np.array((0, 0, 0)), np.array((14312, 13688, 13360)))
#hemi_roi_4 = gp.Roi(np.array((0, 0, 0)), np.array((12168, 14864, 11032)))
#hemi_roi_5 = gp.Roi(np.array((0, 0, 0)), np.array((12088, 14448, 13224)))
hemi_rois = [hemi_roi_1, hemi_roi_3, hemi_roi_4, hemi_roi_5, hemi_roi_7, hemi_roi_19, hemi_roi_21, hemi_roi_23, hemi_roi_25, hemi_roi_27, hemi_roi_29]


def create_source(sample, raw, presyn, postsyn, dummypostsyn, parameter,
                  gt_neurons=None,
                  roi=gp.Roi(np.array((1520, 3644, 3644)), np.array((5000, 5000, 5000))),
                  ):
    data_sources = tuple(
        (
            Hdf5PointsSource(
                os.path.join(data_dir_syn, sample + '.hdf'),
                datasets={presyn: 'annotations',
                          postsyn: 'annotations'},
                rois={
                    presyn: roi,
                    postsyn: roi
                }
            ),
            Hdf5PointsSource(
                os.path.join(data_dir_syn, sample + '.hdf'),
                datasets={
                    dummypostsyn: 'annotations'},
                rois={
                    # presyn: cremi_roi,
                    dummypostsyn: roi
                },
                kind='postsyn'
            ),
            gp.Hdf5Source(
                os.path.join(data_dir, sample + '.hdf'),
                datasets={
                    raw: 'volumes/raw',
                    # gt_neurons: 'volumes/labels/neuron_ids',
                },
                array_specs={
                    raw: gp.ArraySpec(interpolatable=True),
                    # gt_neurons: gp.ArraySpec(interpolatable=False),
                }
            )
        )
    )
    #source_pip = data_sources + gp.MergeProvider() + gp.Normalize(
    #    raw) + gp.RandomLocation()
    
    source_pip = data_sources + gp.MergeProvider() + gp.Normalize(
        raw) + gp.RandomLocation(ensure_nonempty=dummypostsyn,
                                 p_nonempty=parameter['reject_probability'])                       
                                
    return source_pip


def build_pipeline(parameter, augment=True):
    voxel_size = gp.Coordinate(parameter['voxel_size'])

    # Array Specifications.
    raw = gp.ArrayKey('RAW')
    gt_neurons = gp.ArrayKey('GT_NEURONS')
    gt_postpre_vectors = gp.ArrayKey('GT_POSTPRE_VECTORS')
    gt_post_indicator = gp.ArrayKey('GT_POST_INDICATOR')
    post_loss_weight = gp.ArrayKey('POST_LOSS_WEIGHT')
    vectors_mask = gp.ArrayKey('VECTORS_MASK')

    pred_postpre_vectors = gp.ArrayKey('PRED_POSTPRE_VECTORS')
    pred_post_indicator = gp.ArrayKey('PRED_POST_INDICATOR')

    grad_syn_indicator = gp.ArrayKey('GRAD_SYN_INDICATOR')
    grad_partner_vectors = gp.ArrayKey('GRAD_PARTNER_VECTORS')

    # Points specifications
    dummypostsyn = gp.PointsKey('DUMMYPOSTSYN')
    postsyn = gp.PointsKey('POSTSYN')
    presyn = gp.PointsKey('PRESYN')
    trg_context = 140  # AddPartnerVectorMap context in nm - pre-post distance

    with open('train_net_config.json',
              'r') as f:  # ' this is generated via generate_network.py whichs uses parameter.json
        net_config = json.load(f)

    input_size = gp.Coordinate(net_config['input_shape']) * voxel_size
    output_size = gp.Coordinate(net_config['output_shape']) * voxel_size

    request = gp.BatchRequest()
    request.add(raw, input_size)
    # request.add(gt_neurons, output_size)
    request.add(gt_postpre_vectors, output_size)
    request.add(gt_post_indicator, output_size)
    request.add(post_loss_weight, output_size)
    request.add(vectors_mask, output_size)
    request.add(dummypostsyn, output_size)

    for (key, request_spec) in request.items():
        print(key)
        print(request_spec.roi)
        request_spec.roi.contains(request_spec.roi)
    # slkfdms

    snapshot_request = gp.BatchRequest({
        pred_post_indicator: request[gt_postpre_vectors],
        pred_postpre_vectors: request[gt_postpre_vectors],
        grad_syn_indicator: request[gt_postpre_vectors],
        grad_partner_vectors: request[gt_postpre_vectors],
        vectors_mask: request[gt_postpre_vectors]
    })

    postsyn_rastersetting = gp.RasterizationSettings(
        radius=parameter['blob_radius'],
        mask=None,  # gt_neurons,
        mode=parameter['blob_mode'])

    pipeline = tuple([create_source(sample, raw,
                                    presyn, postsyn, dummypostsyn,
                                    parameter, gt_neurons=None, roi=roi) for sample, roi in
                      zip(samples, hemi_rois)])

    pipeline += gp.RandomProvider()
    if augment:
        pipeline += gp.ElasticAugment([40, 40, 40],  # for isotropic
                                      [2, 2, 2],
                                      [0, math.pi / 2.0],
                                      prob_slip=0.05,
                                      prob_shift=0.05,
                                      max_misalign=10,
                                      subsample=8)
        pipeline += gp.SimpleAugment(transpose_only=[0, 1, 2], mirror_only=[0, 1, 2])
        pipeline += gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1,
                                        z_section_wise=True)
    pipeline += gp.IntensityScaleShift(raw, 2, -1)
    pipeline += gp.RasterizePoints(postsyn, gt_post_indicator,
                                   gp.ArraySpec(voxel_size=voxel_size,
                                                dtype=np.int32),
                                   postsyn_rastersetting)

    spec = gp.ArraySpec(voxel_size=voxel_size)
    pipeline += AddPartnerVectorMap(
        src_points=postsyn,
        trg_points=presyn,
        array=gt_postpre_vectors,
        radius=parameter['d_blob_radius'],
        trg_context=trg_context,  # enlarge
        array_spec=spec,
        mask=None, #gt_neurons,
        pointmask=vectors_mask
    )
    pipeline += gp.BalanceLabels(labels=gt_post_indicator,
                                 scales=post_loss_weight,
                                 slab=(-1, -1, -1),
                                 clipmin=parameter['cliprange'][0],
                                 clipmax=parameter['cliprange'][1])
    if parameter['d_scale'] != 1:
        pipeline += gp.IntensityScaleShift(gt_postpre_vectors,
                                           scale=parameter['d_scale'], shift=0)
    pipeline += gp.PreCache(
        cache_size=8,
        num_workers=10)

    pipeline += gp.tensorflow.Train(
        './train_net',
        optimizer=net_config['optimizer'],
        loss=net_config['loss'],
        summary=net_config['summary'],
        log_dir='./tensorboard/',
        save_every=20000,  # saving space in zstore1
        log_every=10000,
        inputs={
            net_config['raw']: raw,
            net_config['gt_partner_vectors']: gt_postpre_vectors,
            net_config['gt_syn_indicator']: gt_post_indicator,
            net_config['vectors_mask']: vectors_mask,
            # Loss weights --> mask
            net_config['indicator_weight']: post_loss_weight,  # Loss weights
        },
        outputs={
            net_config['pred_partner_vectors']: pred_postpre_vectors,
            net_config['pred_syn_indicator']: pred_post_indicator,
        },
        gradients={
            net_config['pred_partner_vectors']: grad_partner_vectors,
            net_config['pred_syn_indicator']: grad_syn_indicator,
        },
    )
    # Visualize.
    pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)
    pipeline += gp.Snapshot({
        raw: 'volumes/raw',
        # gt_neurons: 'volumes/labels/neuron_ids',
        gt_post_indicator: 'volumes/gt_post_indicator',
        gt_postpre_vectors: 'volumes/gt_postpre_vectors',
        pred_postpre_vectors: 'volumes/pred_postpre_vectors',
        pred_post_indicator: 'volumes/pred_post_indicator',
        post_loss_weight: 'volumes/post_loss_weight',
        grad_syn_indicator: 'volumes/post_indicator_gradients',
        grad_partner_vectors: 'volumes/partner_vectors_gradients',
        vectors_mask: 'volumes/vectors_mask'
    },
        every=10000,
        #output_dir='/fibserver/syn_snapshots/syn_octo_cube1',
        output_filename='batch_{iteration}.hdf',
        compression_type='gzip',
        additional_request=snapshot_request)
    pipeline += gp.PrintProfilingStats(every=1000)

    print("Starting training...")
    max_iteration = parameter['max_iteration']
    with gp.build(pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)


if __name__ == "__main__":
    # Set to DEBUG to increase verbosity for
    # everything. logging.INFO --> logging.DEBUG
    logging.basicConfig(level=logging.INFO)

    # Example of how to only increase verbosity for specific python modules.
    logging.getLogger('gunpowder.nodes.rasterize_points').setLevel(
        logging.INFO)
    logging.getLogger('synful.gunpowder.hdf5_points_source').setLevel(
        logging.INFO)

    with open('parameter.json') as f:
        parameter = json.load(f)

    build_pipeline(parameter, augment=True)
