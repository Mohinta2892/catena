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
import glob
import h5py

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
data_dir = '/zstore/catena/data/SYNPAPER_TEM_CLAHE_RESIZED/RESIZED/cross_val/cross_val5/train'


data_dir_syn = data_dir

# --- AUTOMATED SAMPLE AND ROI LOADING START ---
samples = []
hemi_rois = []

# Look for both .hdf and .h5 extensions
file_patterns = [
    os.path.join(data_dir, '*.hdf'),
    os.path.join(data_dir, '*.h5')
]
files = []
for pattern in file_patterns:
    files.extend(glob.glob(pattern))
files.sort()  # Sort to ensure consistent order across runs

print(f"Found {len(files)} files in {data_dir}")

for filepath in files:
    filename = os.path.basename(filepath)
    # Assumes sample name is filename without extension
    sample_name = os.path.splitext(filename)[0]

    try:
        with h5py.File(filepath, 'r') as f:
            if 'volumes/raw' not in f:
                print(f"Skipping {sample_name}: 'volumes/raw' dataset not found.")
                continue

            raw_ds = f['volumes/raw']
            
            # 1. Get Shape (ZYX)
            shape = raw_ds.shape

            # 2. Get Resolution (ZYX)
            if 'resolution' not in raw_ds.attrs:
                raise ValueError(f"Sample {sample_name} missing 'resolution' attribute in 'volumes/raw'")
            resolution = tuple(raw_ds.attrs['resolution'])

            # 3. Get Offset (ZYX) - Default to (0,0,0) if not present
            offset = tuple(raw_ds.attrs.get('offset', (0, 0, 0)))

            # 4. Calculate ROI in physical units
            # Gunpowder ROI = (Offset, Shape in Physical Units)
            roi_offset = gp.Coordinate(offset)
            roi_shape = gp.Coordinate(shape) * gp.Coordinate(resolution)
            
            roi = gp.Roi(roi_offset, roi_shape)

            samples.append(sample_name)
            hemi_rois.append(roi)

            print(f"Loaded {sample_name}:")
            print(f"  - Shape: {shape}")
            print(f"  - Resolution: {resolution}")
            print(f"  - ROI: {roi}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        continue

if not samples:
    raise RuntimeError("No valid samples found. Please check data_dir path and file integrity.")


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
        pipeline += gp.ElasticAugment([4, 40, 40],  # for anisotropic
                                      [0, 2, 2],
                                      [0, math.pi / 2.0],
                                      prob_slip=0.05,
                                      prob_shift=0.05,
                                      max_misalign=10,
                                      subsample=8)
        pipeline += gp.SimpleAugment(transpose_only=[1, 2], mirror_only=[1, 2])
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
        save_every=5000,  # 30000
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
