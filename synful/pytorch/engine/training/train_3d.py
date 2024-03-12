# this makes it potentially compatible with future python versions
from __future__ import print_function

import json
import math
import os
import logging
import random
from funlib.geometry import Roi, Coordinate
from glob import glob
from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from models.models import *
from models.losses import *

from add_ons.gp import AddPartnerVectorMap, Hdf5PointsSource  # , BatchRequest

from synful.models import CalculateModelSummary, initialize_model, initialize_loss

# we set a seed for reproducibility
torch.manual_seed(1961923)
np.random.seed(1961923)
random.seed(1961923)

# padded
cremi_roi = Roi(np.array((1520, 3644, 3644)), np.array((5000, 5000, 5000)))

# cropped
# cremi_roi = gp.Roi(np.array((0, 0, 0)), np.array((5000, 5000, 5000)))

def create_source(sample, raw,
                  presyn, postsyn, dummypostsyn,
                  cfg,
                  gt_neurons,
                  # cremi unpadded roi is set as default
                  roi=Roi(np.array((1520, 3644, 3644)), np.array((5000, 5000, 5000)))
                  ):
    data_sources = tuple(
        (
            # Hdf5PointsSource
            Hdf5PointsSource(
                sample,
                datasets={presyn: 'annotations',
                          postsyn: 'annotations'},
                rois={
                    presyn: roi,
                    postsyn: roi
                }
            ),
            Hdf5PointsSource(
                sample,
                datasets={
                    dummypostsyn: 'annotations'},
                rois={
                    dummypostsyn: roi
                },
                kind='postsyn'
            ),
            Hdf5Source(
                sample,
                datasets={
                    raw: 'volumes/raw',
                    # raw/label shape  (125, 1250, 1250) in zyx if cremi unpadded
                    # raw shape  (200, 1250, 1250) in zyx if cremi padded, label: (125,1250,1250)
                    gt_neurons: 'volumes/labels/neuron_ids',
                },
                array_specs={
                    raw: ArraySpec(interpolatable=True),
                    gt_neurons: ArraySpec(interpolatable=False),
                }
            )
        )
    )
    source_pip = data_sources + MergeProvider() + Normalize(
        raw) + RandomLocation(ensure_nonempty=dummypostsyn,
                              p_nonempty=cfg.MODEL.REJECT_PROBABILITY)

    return source_pip


def train_until(cfg):
    logging.basicConfig(filename=f"./logs/train_logs.txt",  # always overwrite??
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        # set to logging.INFO for fewer details
                        level=logging.DEBUG if cfg.SYSTEM.VERBOSE else logging.INFO)
    module_logger = logging.getLogger(__name__)

    # Warning: Hard-coding, we know we must read `training` data from `data_3d`
    data_dir = os.path.join(cfg.DATA.HOME, cfg.DATA.DATA_DIR_PATH, cfg.DATA.BRAIN_VOL, "data_3d", "train")
    module_logger.debug(f"data_dir {data_dir}")

    # Todo: add this to doc
    module_logger.debug(f"If you are wondering why data_dir is missing your root dir, troubleshoot tip here:"
                        f" https://stackoverflow.com/questions/1945920/why-doesnt-os-path-join-work-in-this-case")

    #  list all files from the above path - assume they are hdf
    samples = glob(f"{data_dir}/*.hdf")
    module_logger.debug(f"samples {samples}")

    voxel_size = Coordinate(cfg.MODEL.VOXEL_SIZE)

    # Array Specifications.
    raw = ArrayKey('RAW')
    gt_neurons = ArrayKey('GT_NEURONS')  # labels

    gt_post_indicator = ArrayKey('GT_POST_INDICATOR')  # post-synaptic mask
    post_loss_weight = ArrayKey('POST_LOSS_WEIGHT')  # loss mask based on where the post-syn mask is placed??
    grad_syn_indicator = ArrayKey('GRAD_SYN_INDICATOR')  # store gradients for post-syn mask
    pred_post_indicator = ArrayKey('PRED_POST_INDICATOR')  # predicted Logits for post-syn mask
    pred_post_indicator_sigmoid = ArrayKey('PRED_POST_INDICATOR_SIGMOID')  # Sigmoid(Logits) for post-syn mask

    gt_postpre_vectors = ArrayKey('GT_POSTPRE_VECTORS')  # pre-post syn vectors
    vectors_mask = ArrayKey('VECTORS_MASK')  # pre-post vector weights for loss calc
    pred_postpre_vectors = ArrayKey('PRED_POSTPRE_VECTORS')  # predicted pre-post vectors

    # Points specifications - Gunpowder Version: 1.0.0rc0.dev0
    # Since points roi is enlarged by AddPartnerVectorMap, add a dummy point
    # request (dummypostsyn; was core_points in
    # `https://github.com/funkelab/synful_experiments/blob/master/02_train/p_setup51/train.py`)
    # to avoid point empty batches downstream of AddPartnerVectorMap.
    dummypostsyn = GraphKey('DUMMYPOSTSYN')
    postsyn = GraphKey('POSTSYN')
    presyn = GraphKey('PRESYN')
    # Flag : don't know what this does/means exactly
    trg_context = 140  # AddPartnerVectorMap context in nm - pre-post distance

    input_size = Coordinate(cfg.MODEL.INPUT_SHAPE) * voxel_size
    output_size = Coordinate(cfg.MODEL.OUTPUT_SHAPE) * voxel_size

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(gt_neurons, output_size)
    request.add(gt_post_indicator, output_size)
    request.add(post_loss_weight, output_size)
    request.add(dummypostsyn, output_size)
    request.add(pred_post_indicator, output_size)
    # request.add(pred_post_indicator_sigmoid, output_size)
    request.add(vectors_mask, output_size)
    request.add(gt_postpre_vectors, output_size)
    if cfg.TRAIN.MODEL_TYPE not in ["SynMT1", "STVEC"]:
        request.add(pred_postpre_vectors, output_size)

    for (key, request_spec) in request.items():
        # print(key)
        # print(request_spec.roi)
        request_spec.roi.contains(request_spec.roi)

        # Initialize the model
    if cfg.TRAIN.MODEL_TYPE not in ["ACMASK", "ACVEC"]:
        model = initialize_model(cfg)
        calc_shape_obj = CalculateModelSummary(model, cfg)
        output_shape = calc_shape_obj.calculate_output_shape()[-len(cfg.MODEL.INPUT_SHAPE):]

    module_logger.debug("Model")
    print(f"Model Parameters: {(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6):.3f}M")
    loss = initialize_loss(cfg)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.TRAIN.INITIAL_LR,
        betas=cfg.TRAIN.LR_BETAS)

    postsyn_rastersetting = RasterizationSettings(
        radius=cfg.MODEL.BLOB_RADIUS,
        mask=gt_neurons,
        mode=cfg.MODEL.BLOB_MODE)

    pipeline = tuple([create_source(sample, raw,
                                    presyn, postsyn, dummypostsyn,
                                    cfg,
                                    gt_neurons
                                    ) for sample in
                      samples])

    pipeline += RandomProvider()

    if cfg.TRAIN.AUGMENT:
        pipeline += ElasticAugment((4, 40, 40),
                                   (0, 2, 2),
                                   [0, math.pi / 2.0],
                                   prob_slip=0.05,
                                   prob_shift=0.05,
                                   max_misalign=10,
                                   subsample=8)
        pipeline += SimpleAugment(transpose_only=[1, 2], mirror_only=[1, 2])
        pipeline += IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1,
                                     z_section_wise=True)

    pipeline += IntensityScaleShift(raw, 2, -1)
    # gt_post_indicator is the output array, if dtype is not provided, by default it should be np.uint8
    pipeline += RasterizeGraph(postsyn, gt_post_indicator,
                               ArraySpec(voxel_size=voxel_size,
                                         dtype=np.uint8),
                               postsyn_rastersetting)

    spec = ArraySpec(voxel_size=voxel_size)
    pipeline += AddPartnerVectorMap(
        src_points=postsyn,
        trg_points=presyn,
        array=gt_postpre_vectors,
        radius=cfg.MODEL.D_BLOB_RADIUS,
        trg_context=trg_context,  # enlarge
        array_spec=spec,
        mask=gt_neurons,
        pointmask=vectors_mask
    )

    pipeline += BalanceLabels(labels=gt_post_indicator,
                              scales=post_loss_weight,
                              slab=(-1, -1, -1),
                              clipmin=cfg.MODEL.CLIPRANGE[0],
                              clipmax=cfg.MODEL.CLIPRANGE[1])

    if cfg.MODEL.D_SCALE != 1:
        pipeline += IntensityScaleShift(gt_postpre_vectors,
                                        scale=cfg.MODEL.D_SCALE, shift=0)
    pipeline += PreCache(
        cache_size=10,
        num_workers=2)

    # customize the loss inputs and outputs here based on model type
    if cfg.TRAIN.MODEL_TYPE == "SynMT1":
        loss_inputs = {
            0: pred_post_indicator,
            1: gt_post_indicator,
            2: post_loss_weight,
            3: pred_postpre_vectors,
            4: gt_postpre_vectors,
            5: vectors_mask
        }

        outputs = {
            0: pred_post_indicator,
            1: pred_postpre_vectors
        }
        snapshot_ds = {
            raw: 'raw',
            gt_neurons: 'labels',
            gt_post_indicator: 'gt_post_indicator',
            pred_post_indicator: 'pred_post_indicator',
            gt_postpre_vectors: 'gt_postpre_vectors',
            pred_postpre_vectors: 'pred_postpre_vectors',
        }
    elif cfg.TRAIN.MODEL_TYPE == "STMASK":
        loss_inputs = {
            0: pred_post_indicator,
            1: gt_post_indicator,
            2: post_loss_weight
        }

        outputs = {
            0: pred_post_indicator
        }
        snapshot_ds = {
            raw: 'raw',
            gt_neurons: 'labels',
            gt_post_indicator: 'gt_post_indicator',
            pred_post_indicator: 'pred_post_indicator',
            gt_postpre_vectors: 'gt_postpre_vectors',
        }

    elif cfg.TRAIN.MODEL_TYPE == "STVEC":
        loss_inputs = {
            0: pred_postpre_vectors,
            1: gt_postpre_vectors,
            2: vectors_mask
        }

        outputs = {
            0: pred_postpre_vectors
        }

        snapshot_ds = {
            raw: 'raw',
            gt_neurons: 'labels',
            pred_postpre_vectors: 'pred_postpre_vectors',
            gt_postpre_vectors: 'gt_postpre_vectors',
            vectors_mask: 'vectors_mask'
        }

    # shape c(1) x d x h x w
    pipeline += Unsqueeze([raw])
    # batch_size == 1; shape b(1) x c(1) x d x h x w
    pipeline += Stack(cfg.TRAIN.BATCH_SIZE)
    train = Train(
        model=model,
        loss=loss,
        optimizer=optimizer,
        inputs={
            'x': raw,
        },
        loss_inputs=loss_inputs,
        outputs=outputs,
        save_every=cfg.TRAIN.SAVE_EVERY,
        log_dir=cfg.MODEL.LOG_DIR,
        device=cfg.TRAIN.DEVICE,
        checkpoint_folder=cfg.MODEL.CKPT_FOLDER,
        use_wandb=cfg.PREPROCESS.USE_WANDB  # Todo: add functionality or remove
    )
    pipeline += train

    pipeline += Squeeze([raw], axis=None)
    squeeze_output_list = [raw, gt_post_indicator, gt_postpre_vectors]
    # have to squeeze selectively now
    if cfg.TRAIN.MODEL_TYPE in ["STMASK", "SynMT1"]:
        squeeze_output_list.extend([pred_post_indicator])

    elif cfg.TRAIN.MODEL_TYPE in ["STVEC", "SynMT1"]:
        squeeze_output_list.extend([pred_postpre_vectors])

    # raw shape: c x d x h x w ---> d x h x w;
    # affs/lsds: b x c x d x h x w --> c x d x h x w
    pipeline += Squeeze(squeeze_output_list, axis=None)

    # # Visualize.
    pipeline += IntensityScaleShift(raw, 0.5, 0.5)
    pipeline += Snapshot(snapshot_ds,
                         every=1000,
                         output_filename='batch_{iteration}.hdf',
                         compression_type='gzip',
                         # additional_request=snapshot_request
                         )

    # pipeline += PrintProfilingStats(every=1000)

    print("Starting training...")
    max_iteration = cfg.TRAIN.EPOCHS
    with build(pipeline) as b:
        for i in (
                pbar := tqdm(range(train.iteration, max_iteration), desc=f"Model resumed from {train.iteration}",
                             )):
            batch = b.request_batch(request)
            pbar.set_postfix({"loss": batch.loss})


if __name__ == "__main__":
    # Set to DEBUG to increase verbosity for
    # everything. logging.INFO --> logging.DEBUG
    logging.basicConfig(level=logging.INFO)

    # Example of how to only increase verbosity for specific python modules.
    logging.getLogger('gunpowder.nodes.rasterize_points').setLevel(
        logging.DEBUG)
    logging.getLogger('synful.gunpowder.hdf5_points_source').setLevel(
        logging.DEBUG)

    # with open('../single_task/PostSynapticMask/parameter.json') as f:
    #     parameter = json.load(f)
    #
    # build_pipeline(parameter, augment=False)
