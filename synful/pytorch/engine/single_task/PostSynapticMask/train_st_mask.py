# this makes it potentially compatible with future python versions
from __future__ import print_function

import json
import math
import os
import logging
import random
from funlib.geometry import Roi

from models.models import *
from models.losses import *

# from utils import *  # Squeeze and UnSqueeze are imported from this

try:
    import logging

    for handler in logging.root.handlers[:]:
        if handler.__class__.__name__ == 'StreamHandler':
            logging.root.removeHandler(handler)

    # Disable pre-init warning from absl.logging
    logging.root.manager.disable = logging.CRITICAL
except Exception as e:
    print(e)

import gunpowder as gp
import numpy as np
import daisy

from synful.add_ons.gp import AddPartnerVectorMap, Hdf5PointsSource, BatchRequest, ZarrPointsSource

import torch
from tqdm import tqdm

# we set a seed for reproducibility
torch.manual_seed(1961923)
np.random.seed(1961923)
random.seed(1961923)

# ensure that only the parent folder contains a backward slash at the beginning, otherwise path.join to
# prevent path.join from discarding the base folders
home = '/media/samia/DATA/ark'
# CREMI specific, download data from: www.cremi.org
data_dir = os.path.join(home, 'dan-samia/lsd/')

data_dir_syn = data_dir
print(data_dir_syn)

samples = [
    # 'funke/cremi/training/3D/cropped/hdf/sample_A_20160501',
    # 'funke/cremi/training/3D/cropped/hdf/sample_B_20160501',
    'funke/cremi/training/3D/padded/hdf/sample_C_padded_20160501',
    'funke/cremi/training/3D/padded/hdf/sample_A_padded_20160501',
    'funke/cremi/training/3D/padded/hdf/sample_B_padded_20160501',

]
# padded
cremi_roi = Roi(np.array((1520, 3644, 3644)), np.array((5000, 5000, 5000)))


# cropped
# cremi_roi = gp.Roi(np.array((0, 0, 0)), np.array((5000, 5000, 5000)))

def create_source(sample, raw,
                  presyn, postsyn, dummypostsyn,
                  parameter,
                  gt_neurons,
                  # cremi unpadded roi is set as default
                  roi=Roi(np.array((1520, 3644, 3644)), np.array((5000, 5000, 5000)))
                  ):
    data_sources = tuple(
        (
            # Hdf5PointsSource
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
                    dummypostsyn: roi
                },
                kind='postsyn'
            ),
            gp.Hdf5Source(
                os.path.join(data_dir_syn, sample + '.hdf'),
                datasets={
                    raw: 'volumes/raw',
                    # raw/label shape  (125, 1250, 1250) in zyx if cremi unpadded
                    # raw shape  (200, 1250, 1250) in zyx if cremi padded, label: (125,1250,1250)
                    gt_neurons: 'volumes/labels/neuron_ids',
                },
                array_specs={
                    raw: gp.ArraySpec(interpolatable=True),
                    gt_neurons: gp.ArraySpec(interpolatable=False),
                }
            )
        )
    )
    source_pip = data_sources + gp.MergeProvider() + gp.Normalize(
        raw) + gp.RandomLocation(ensure_nonempty=dummypostsyn,
                                 p_nonempty=parameter['reject_probability'])

    return source_pip


def build_pipeline(parameter, augment=True):
    voxel_size = gp.Coordinate(parameter['voxel_size'])

    # Array Specifications.
    raw = gp.ArrayKey('RAW')
    gt_neurons = gp.ArrayKey('GT_NEURONS')
    gt_post_indicator = gp.ArrayKey('GT_POST_INDICATOR')
    post_loss_weight = gp.ArrayKey('POST_LOSS_WEIGHT')

    pred_post_indicator = gp.ArrayKey('PRED_POST_INDICATOR')
    pred_post_indicator_sigmoid = gp.ArrayKey('PRED_POST_INDICATOR_SIGMOID')

    grad_syn_indicator = gp.ArrayKey('GRAD_SYN_INDICATOR')
    vectors_mask = gp.ArrayKey('VECTORS_MASK')
    gt_postpre_vectors = gp.ArrayKey('GT_POSTPRE_VECTORS')

    # Points specifications - Gunpowder Version: 1.0.0rc0.dev0
    # Since points roi is enlarged by AddPartnerVectorMap, add a dummy point
    # request (dummypostsyn; was core_points in
    # `https://github.com/funkelab/synful_experiments/blob/master/02_train/p_setup51/train.py`)
    # to avoid point empty batches downstream of AddPartnerVectorMap.
    # dummypostsyn = PointsKey('DUMMYPOSTSYN')
    dummypostsyn = gp.GraphKey('DUMMYPOSTSYN')
    # postsyn = PointsKey('POSTSYN')
    postsyn = gp.GraphKey('POSTSYN')
    # presyn = PointsKey('PRESYN')
    presyn = gp.GraphKey('PRESYN')
    # Flag : don't know what this does/means exactly
    trg_context = 140  # AddPartnerVectorMap context in nm - pre-post distance

    # Instead of train_net_config.json we read the parameter.json directly
    # replace `net_config` with `parameter` passed in as args
    with open('../single_task/PostSynapticMask/parameter.json', 'r') as f:
        net_config = json.load(f)
        print(net_config)

    input_size = gp.Coordinate(net_config['input_shape']) * voxel_size
    # Was previously calculated via a conv pass - now we either set it or implement the calc above
    net_config['output_shape'] = (54, 162, 162)
    output_size = gp.Coordinate(net_config['output_shape']) * voxel_size

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(gt_neurons, output_size)
    request.add(gt_post_indicator, output_size)
    # request.add(post_loss_weight, output_size)
    request.add(dummypostsyn, output_size)
    # request.add(pred_post_indicator, output_size)
    # request.add(pred_post_indicator_sigmoid, output_size)
    request.add(vectors_mask, output_size)
    request.add(gt_postpre_vectors, output_size)

    for (key, request_spec) in request.items():
        print(key)
        print(request_spec.roi)
        request_spec.roi.contains(request_spec.roi)
        # slkfdms

    # # # model, loss and optimizer instantiation
    # model = STMaskSynfulModel(
    #     in_channels=1,
    #     num_fmaps=net_config["fmap_num"],
    #     fmap_inc_factor=net_config["fmap_inc_factor"],
    #     downsample_factors=net_config["downsample_factors"],
    # )
    # print(model)
    #
    loss = initialize_loss(cfg)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.TRAIN.INITIAL_LR,
        betas=cfg.TRAIN.LR_BETAS)

    postsyn_rastersetting = gp.RasterizationSettings(
        radius=parameter['blob_radius'],
        mask=gt_neurons,
        mode=parameter['blob_mode'])

    pipeline = tuple([create_source(sample, raw,
                                    presyn, postsyn, dummypostsyn,
                                    parameter,
                                    gt_neurons
                                    ) for sample in
                      samples])

    pipeline += gp.RandomProvider()

    if augment:
        pipeline += gp.ElasticAugment((4, 40, 40),
                                      (0, 2, 2),
                                      [0, math.pi / 2.0],
                                      prob_slip=0.05,
                                      prob_shift=0.05,
                                      max_misalign=10,
                                      subsample=8)
        pipeline += gp.SimpleAugment(transpose_only=[1, 2], mirror_only=[1, 2])
        pipeline += gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1,
                                        z_section_wise=True)

    pipeline += gp.IntensityScaleShift(raw, 2, -1)
    # gt_post_indicator is the output array, if dtype is not provided, by default it should be np.uint8
    pipeline += gp.RasterizeGraph(postsyn, gt_post_indicator,
                                  gp.ArraySpec(voxel_size=voxel_size,
                                               dtype=np.uint8),
                                  postsyn_rastersetting)

    spec = gp.ArraySpec(voxel_size=voxel_size)
    pipeline += AddPartnerVectorMap(
        src_points=postsyn,
        trg_points=presyn,
        array=gt_postpre_vectors,
        radius=parameter['d_blob_radius'],
        trg_context=trg_context,  # enlarge
        array_spec=spec,
        mask=gt_neurons,
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
        cache_size=10,
        num_workers=2)

    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Unsqueeze([raw])

    pipeline += gp.Train(
        model=model,
        loss=loss,
        optimizer=optimizer,
        inputs={
            'input': raw,
        },
        loss_inputs={
            0: pred_post_indicator,
            1: gt_post_indicator,
            2: post_loss_weight
        },
        outputs={
            0: pred_post_indicator,
            1: pred_post_indicator_sigmoid
        },
        # gradients={
        #     0: grad_syn_indicator,
        # },
        save_every=1000,
        # log_dir='logs',
        # device='cuda:0'
    )

    # # # # Visualize.
    # pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)
    # pipeline += gp.Snapshot({
    #     raw: 'volumes/raw',
    #     gt_neurons: 'volumes/labels/neuron_ids',
    #     gt_post_indicator: 'volumes/gt_post_indicator',
    #     pred_post_indicator: 'volumes/pred_post_indicator',  # this is an output of the network - mask
    #     pred_post_indicator_sigmoid: 'volumes/post_syn_location',  # this is an output of the network - mask
    #     post_loss_weight: 'volumes/post_loss_weight',
    #     # grad_syn_indicator: 'volumes/post_indicator_gradients',
    # },
    #     every=1000,
    #     output_filename='batch_{iteration}.hdf',
    #     compression_type='gzip',
    #     # additional_request=snapshot_request
    # )
    # pipeline += gp.PrintProfilingStats(every=1000)

    print("Starting training...")
    max_iteration = parameter['max_iteration']
    with gp.build(pipeline) as b:
        for i in range(max_iteration):  # tqdm is meaningless unless adapted to checkpoint iterations as of now
            batch = b.request_batch(request)
            print(f"RAW --> {batch[raw].data}")
            print(f"gt neurons --> {np.unique(batch[gt_neurons].data)}")
            print(batch[gt_neurons].spec.roi)
            # print(f"RAW --> {batch[gt_postpre_vectors].data}")
            # print(f"Pred_post_ind --> {batch[pred_post_indicator].data}")
            # print(f"GT_post_indicator --> {np.nonzero(batch[gt_post_indicator].data)}")


if __name__ == "__main__":
    # Set to DEBUG to increase verbosity for
    # everything. logging.INFO --> logging.DEBUG
    logging.basicConfig(level=logging.INFO)

    # Example of how to only increase verbosity for specific python modules.
    logging.getLogger('gunpowder.nodes.rasterize_points').setLevel(
        logging.DEBUG)
    logging.getLogger('synful.gunpowder.hdf5_points_source').setLevel(
        logging.DEBUG)

    with open('../single_task/PostSynapticMask/parameter.json') as f:
        parameter = json.load(f)

    build_pipeline(parameter, augment=False)
