# this makes it potentially compatible with future python versions
from __future__ import print_function

import json
import math
import os
import logging
import random

from funlib.geometry import Roi, Coordinate
from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import sys
import logging


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from models.models import *
from models.losses import *

from add_ons.gp import AddPartnerVectorMap, Hdf5PointsSource  # , BatchRequest
from add_ons.detection import SynapseExtractionParameters
from add_ons.gp import ExtractSynapses
from add_ons.gp import IntensityScaleShiftClip


# we set a seed for reproducibility
torch.manual_seed(1961923)
np.random.seed(1961923)
random.seed(1961923)


def predict(cfg):
    logging.basicConfig(filename=f"./logs/train_logs.txt",  # always overwrite??
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        # set to logging.INFO for fewer details
                        level=logging.DEBUG if cfg.SYSTEM.VERBOSE else logging.INFO)
    module_logger = logging.getLogger(__name__)

    # Warning: Hard-coding, we know we must read `training` data from `data_3d`
    data_dir = os.path.join(cfg.DATA.HOME, cfg.DATA.DATA_DIR_PATH, cfg.DATA.BRAIN_VOL, "data_3d", "test")
    module_logger.debug(f"data_dir {data_dir}")

    # Todo: add this to doc
    module_logger.debug(f"If you are wondering why data_dir is missing your root dir, troubleshoot tip here:"
                        f" https://stackoverflow.com/questions/1945920/why-doesnt-os-path-join-work-in-this-case")

    extractionsettings = SynapseExtractionParameters(
        extract_type=cfg.PREDICT.EXTRACT_TYPE,
        cc_threshold=cfg.PREDICT.CC_THRESHOLD,
        loc_type=cfg.PREDICT.LOC_TYPE,
        score_thr=cfg.PREDICT.SCORE_THRESHOLD,
        score_type=cfg.PREDICT.SCORE_TYPE,
        nms_radius=cfg.PREDICT.NMS_RADIUS
    )

    voxel_size = Coordinate(cfg.MODEL.VOXEL_SIZE)

    # Array Specifications.
    raw = ArrayKey('RAW')

    pred_post_indicator = ArrayKey('PRED_POST_INDICATOR')  # predicted Logits for post-syn mask
    pred_postpre_vectors = ArrayKey('PRED_POSTPRE_VECTORS')  # predicted pre-post vectors

    # Points specifications - Gunpowder Version: 1.0.0rc0.dev0
    # Since points roi is enlarged by AddPartnerVectorMap, add a dummy point
    # request (dummypostsyn; was core_points in
    # `https://github.com/funkelab/synful_experiments/blob/master/02_train/p_setup51/train.py`)
    # to avoid point empty batches downstream of AddPartnerVectorMap.
    pred_postsyn = GraphKey('POSTSYN')
    pred_presyn = GraphKey('PRESYN')

    input_size = Coordinate(cfg.MODEL.INPUT_SHAPE) * voxel_size
    output_size = Coordinate(cfg.MODEL.OUTPUT_SHAPE) * voxel_size
    if not (type(cfg.PREDICT.SYNAPSE_CONTEXT) == list or type(cfg.PREDICT.SYNAPSE_CONTEXT) == tuple):
        synapse_context = [cfg.PREDICT.SYNAPSE_CONTEXT] * 3
    output_size -= Coordinate(synapse_context) * 2

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(pred_presyn, output_size)
    request.add(pred_postsyn, output_size)

    # Initialize the model
    model = initialize_model(cfg)
    calc_shape_obj = CalculateModelSummary(model, cfg)
    output_shape = calc_shape_obj.calculate_output_shape()[-len(cfg.MODEL.INPUT_SHAPE):]
    module_logger.debug("Model")
    model.eval()

    # Hdf5Source
    if cfg.DATA.SAMPLE.endswith('.hdf') or cfg.DATA.SAMPLE.endswith('.h5'):
        data_sources = Hdf5Source(
            cfg.DATA.SAMPLE,
            datasets={
                raw: cfg.DATA.RAW
            },
            array_specs={
                raw: ArraySpec(interpolatable=True),
            }
        )
    # ZarrSource
    elif cfg.DATA.SAMPLE.endswith('.zarr') or cfg.DATA.SAMPLE.endswith('.n5'):
        data_sources = ZarrSource(
            cfg.DATA.SAMPLE,
            datasets={
                raw: cfg.DATA.RAW
            },
            array_specs={
                raw: ArraySpec(interpolatable=True),
            }
        )
    else:
        raise RuntimeError('UNKNOWN input data format {}'.format(cfg.DATA.SAMPLE))

    # context added/removed
    context = (input_size - output_size) / 2
    module_logger.debug(f"input_size: {input_size}; output_size: {output_size}")
    # create an output roi anew based on context
    with build(data_sources):
        raw_roi = data_sources.spec[raw].roi
    total_output_roi = raw_roi.grow(-context, -context)
    module_logger.debug(f"Total output ROI {total_output_roi} and context {context}")

    pipeline = data_sources
    pipeline += Pad(raw, size=None)
    pipeline += Normalize(raw)
    pipeline += IntensityScaleShift(raw, 2, -1)

    # higher values here?
    pipeline += PreCache(
        cache_size=8,  # 40
        num_workers=8)  # 7?

    # customize the loss inputs and outputs here based on model type
    if cfg.TRAIN.MODEL_TYPE == "SynMT1":

        outputs = {
            0: pred_post_indicator,
            1: pred_postpre_vectors
        }

    elif cfg.TRAIN.MODEL_TYPE == "STMASK":
        outputs = {
            0: pred_post_indicator
        }

    elif cfg.TRAIN.MODEL_TYPE == "STVEC":

        outputs = {
            0: pred_postpre_vectors
        }

    # shape c(1) x d x h x w
    pipeline += Unsqueeze([raw])
    # batch_size == 1; shape b(1) x c(1) x d x h x w
    pipeline += Stack(cfg.TRAIN.BATCH_SIZE)
    pipeline += Predict(
        model=model,
        checkpoint=cfg.TRAIN.CHECKPOINT,
        inputs={
            'x': raw,  # key should as in the forward defined in the models.py
        },
        outputs=outputs,  # selectively pass output based on model type
        device=cfg.TRAIN.DEVICE
    )

    # # D_SCALE??
    if cfg.MODEL.D_SCALE != 1 and cfg.MODEL.D_SCALE is not None:
        pipeline += IntensityScaleShift(pred_postpre_vectors,
                                        scale=1. / cfg.MODEL.D_SCALE, shift=0)

    pipeline += Squeeze([raw], axis=None)
    squeeze_output_list = [raw]
    # have to squeeze selectively now
    if cfg.TRAIN.MODEL_TYPE in ["STMASK", "SynMT1"]:
        squeeze_output_list.extend([pred_post_indicator])

    if cfg.TRAIN.MODEL_TYPE in ["STVEC", "SynMT1"]:
        squeeze_output_list.extend([pred_postpre_vectors])

    # raw shape: c x d x h x w ---> d x h x w;
    # raw shape: c x d x h x w ---> d x h x w;
    # affs/lsds: b x c x d x h x w --> c x d x h x w
    pipeline += Squeeze(squeeze_output_list, axis=None)

    pipeline += ExtractSynapses(pred_post_indicator,
                                pred_postpre_vectors,
                                pred_postsyn, pred_presyn,
                                settings=extractionsettings,
                                context=cfg.PREDICT.SYNAPSE_CONTEXT,
                                out_dir=cfg.DATA.OUTFILE)

    pipeline += Scan(request)

    print("Starting prediction...")
    with build(pipeline):
        pipeline.request_batch(BatchRequest())
    print("Prediction finished")


if __name__ == "__main__":
    # Set to DEBUG to increase verbosity for
    # everything. logging.INFO --> logging.DEBUG
    logging.basicConfig(level=logging.INFO)

    # Example of how to only increase verbosity for specific python modules.
    logging.getLogger('gunpowder.nodes.rasterize_points').setLevel(
        logging.DEBUG)
    logging.getLogger('synful.gunpowder.hdf5_points_source').setLevel(
        logging.DEBUG)


