from __future__ import print_function

import logging
import os
import sys

import gunpowder as gp
from gunpowder.ext import torch
import numpy as np
import pymongo
from yacs.config import CfgNode as CN  # default config
import argparse
from funlib.geometry import Roi, Coordinate
from pathlib import Path
import glob
import random
import datetime
import math

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from models.models import *
from models.losses import *
from config.config import get_cfg_defaults

from add_ons.gp import AddPartnerVectorMap, Hdf5PointsSource  # , BatchRequest
from add_ons.detection import SynapseExtractionParameters
from add_ons.gp import ExtractSynapses
from add_ons.gp import IntensityScaleShiftClip


torch.backends.cudnn.benchmark = True

# we set a seed for reproducibility
torch.manual_seed(1961923)
np.random.seed(1961923)
random.seed(1961923)


def rename_keys(original_config, key_mapping):
    for new_key, old_key in key_mapping.items():
        if hasattr(original_config, old_key):
            original_config[new_key] = getattr(original_config, old_key)

    return original_config


def block_done_callback(
        db_host,
        db_name,
        block,
        start,
        duration,
        worker_config=None):
    print("Recording block-done for %s" % (block,))

    # print('Trying to connect!')
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    collection = db['blocks_predicted']

    # print('Connected!')

    # print(dict(worker_config))
    document = dict(worker_config)
    document.update({
        'block_id': block.block_id,
        'read_roi': (block.read_roi.get_begin(), block.read_roi.get_shape()),
        'write_roi': (block.write_roi.get_begin(), block.write_roi.get_shape()),
        'start': start,
        'duration': duration
    })

    x = collection.insert_one(document)
    # print('------Inserted ID------ \n', x.inserted_id)
    # print('------Document------ \n', document)

    print("Recorded block-done for %s" % (block,))


def predict(cfg):

    logging.basicConfig(filename=f"./logs/predict_scan_logs_{datetime.datetime.now()}.txt",
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        # set to logging.INFO for fewer details
                        level=logging.DEBUG if cfg.SYSTEM.VERBOSE else logging.INFO)
    module_logger = logging.getLogger(__name__)

    out_properties = {
        "pred_syn_indicator_out": {
            "dsname": "pred_syn_indicator",
            "dtype": "uint8",
            "scale": 255
        },
        "pred_partner_vectors": {
            "dtype": "int8",
            "scale": 0.25
        }
    }

    # simplify this structure
    data_dir = os.path.join(cfg.DATA.HOME, cfg.DATA.DATA_DIR_PATH, cfg.DATA.BRAIN_VOL)
    module_logger.debug(f"data_dir {data_dir}")

    # Initialize the model
    model = initialize_model(cfg)
    model.eval()

    voxel_size = Coordinate(cfg.MODEL.VOXEL_SIZE)
    input_size = Coordinate(cfg.MODEL.INPUT_SHAPE) * voxel_size
    output_size = Coordinate(cfg.MODEL.OUTPUT_SHAPE) * voxel_size
    # context added/removed
    context = (input_size - output_size) / 2
    module_logger.debug(f"input_size: {input_size}; output_size: {output_size}")


    raw = gp.ArrayKey('RAW')
    pred_postpre_vectors = gp.ArrayKey('PRED_POSTPRE_VECTORS')
    pred_post_indicator = gp.ArrayKey('PRED_POST_INDICATOR')

    chunk_request = gp.BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(pred_postpre_vectors, output_size)
    chunk_request.add(pred_post_indicator, output_size)

    d_property = out_properties[
        'pred_partner_vectors'] if 'pred_partner_vectors' in out_properties else None
    m_property = out_properties[
        'pred_syn_indicator_out'] if 'pred_syn_indicator_out' in out_properties else None

    # Hdf5Source
    if cfg.DATA.SAMPLE.endswith('.hdf') or cfg.DATA.SAMPLE.endswith('.h5'):
        data_sources = gp.Hdf5Source(
            cfg.DATA.SAMPLE,
            datasets={
                raw: cfg.DATA.RAW
            },
            array_specs={
                raw: gp.ArraySpec(interpolatable=True),
            }
        )
    # ZarrSource
    elif cfg.DATA.SAMPLE.endswith('.zarr') or cfg.DATA.SAMPLE.endswith('.n5'):
        data_sources = gp.ZarrSource(
            cfg.DATA.SAMPLE,
            datasets={
                raw: cfg.DATA.RAW
            },
            array_specs={
                raw: gp.ArraySpec(interpolatable=True),
            }
        )
    else:
        raise RuntimeError('UNKNOWN input data format {}'.format(cfg.DATA.SAMPLE))

    # create an output roi anew based on context
    with gp.build(data_sources):
        raw_roi = data_sources.spec[raw].roi
    total_output_roi = raw_roi.grow(-context, -context)
    module_logger.debug(f"Total output ROI {total_output_roi} and context {context}")

    pipeline = data_sources

    pipeline += gp.Pad(raw, size=None)

    pipeline += gp.Normalize(raw)

    pipeline += gp.IntensityScaleShift(raw, 2, -1)

    pipeline += gp.torch.Predict(
        model=model,
        checkpoint=cfg.TRAIN.CHECKPOINT,
        inputs={
            'x': raw
        },
        outputs={
            0: pred_post_indicator,
            1: pred_postpre_vectors
        })
    # d_scale = parameters['d_scale'] if 'd_scale' in parameters else None
    if cfg.MODEL.D_SCALE != 1 and cfg.MODEL.D_SCALE is not None:
        d_scale = cfg.MODEL.D_SCALE
        # if d_scale != 1 and d_scale is not None:
        pipeline += gp.IntensityScaleShift(pred_postpre_vectors,
                                           1. / d_scale,
                                           0)  # Map back to nm world.
    if m_property is not None and 'scale' in m_property:
        if m_property['scale'] != 1:
            pipeline += gp.IntensityScaleShift(pred_post_indicator,
                                               m_property['scale'], 0)
    if d_property is not None and 'scale' in d_property:
        pipeline += gp.IntensityScaleShift(pred_postpre_vectors,
                                           d_property['scale'], 0)
    if d_property is not None and 'dtype' in d_property:
        assert d_property['dtype'] == 'int8' or d_property[
            'dtype'] == 'float32', 'predict not adapted to dtype {}'.format(
            d_property['dtype'])
        if d_property['dtype'] == 'int8':
            pipeline += IntensityScaleShiftClip(pred_postpre_vectors,
                                                1, 0, clip=(-128, 127))

    pipeline += gp.ZarrWrite(
        dataset_names={
            pred_post_indicator: 'volumes/pred_syn_indicator',
            pred_postpre_vectors: 'volumes/pred_partner_vectors',
        },
        output_dir=os.path.dirname(cfg.DATA.OUTFILE),
        output_filename=os.path.basename(cfg.DATA.OUTFILE),
        dataset_dtypes= {pred_post_indicator: gp.ArraySpec(roi=total_output_roi),
                         pred_postpre_vectors: gp.ArraySpec(roi=total_output_roi)}
    )

    pipeline += gp.PrintProfilingStats(every=10)

    pipeline += gp.DaisyRequestBlocks(
        chunk_request,
        roi_map={
            raw: 'read_roi',
            pred_postpre_vectors: 'write_roi',
            pred_post_indicator: 'write_roi'
        },
        num_workers=2,
        block_done_callback=lambda b, s, d: block_done_callback(
            cfg.DATA.DB_HOST,
            cfg.DATA.DB_NAME,
            b, s, d, None))

    print("Starting prediction...")
    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())
    print("Prediction finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('gunpowder.nodes.hdf5like_write_base').setLevel(
        logging.DEBUG)

    """ Whatever comes from the master file must be passed to the predict above.
    Since, it is originally like:
    ``` subprocess.run["python", "predict_worker.py", "config_file"] ```
    this has to be adjusted such that the cfg can be read directly.
    """
    config_file = sys.argv[1]

    # parse the args file to become cfg
    cfg = CN()
    # Allow creating new keys recursively.: https://github.com/rbgirshick/yacs/issues/25
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file)
    predict(cfg)
