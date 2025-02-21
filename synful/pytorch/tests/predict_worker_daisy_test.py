import datetime
import math
import numpy as np
import os
from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *
# this needs to change for lsd > 0.1.3 : from lsd.train.gp import AddLocalShapeDescriptor
# from lsd.gp import AddLocalShapeDescriptor
# from lsd.train.gp import AddLocalShapeDescriptor
import argparse
import yaml
# add current directory to path and allow absolute imports - this is a terrible for now
import sys
from pathlib import Path
import ast
from tqdm import tqdm
from glob import glob
import random
import torch
import pymongo

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.models import *
from models.losses import *
from add_ons.funlib_persistence.persistence_utils import *
from add_ons.gp import AddPartnerVectorMap, Hdf5PointsSource  # , BatchRequest
from add_ons.detection import SynapseExtractionParameters
from add_ons.gp import ExtractSynapses
from add_ons.gp import IntensityScaleShiftClip


from config.config_predict import get_cfg_defaults  # import but do no use

from funlib.persistence import prepare_ds, open_ds
from yacs.config import CfgNode as CN
# from add_ons.gp.batch_zarr_write import ZarrWrite

torch.backends.cudnn.benchmark = True

# we set a seed for reproducibility
torch.manual_seed(1961923)
np.random.seed(1961923)
random.seed(1961923)


def block_done_callback(db_host, db_name, block, start, duration, db_collection_name='blocks_predicted',
                        worker_config=None):
    print("Recording block-done for %s" % (block,))

    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    collection = db[db_collection_name]

    # document = dict(worker_config)
    document = dict()
    document.update(
        {
            "block_id": block.block_id,
            "read_roi": (block.read_roi.get_begin(), block.read_roi.get_shape()),
            "write_roi": (block.write_roi.get_begin(), block.write_roi.get_shape()),
            "start": start,
            "duration": duration,
        }
    )

    collection.insert_one(document)

    print("Recorded block-done for %s" % (block,))


def predict(cfg):
    logging.basicConfig(filename=f"./logs/predict_scan_logs_{datetime.datetime.now()}.txt",
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        # set to logging.INFO for fewer details
                        level=logging.DEBUG if cfg.SYSTEM.VERBOSE else logging.INFO)
    module_logger = logging.getLogger(__name__)

    # simplify this structure
    data_dir = os.path.join(cfg.DATA.HOME, cfg.DATA.DATA_DIR_PATH, cfg.DATA.BRAIN_VOL)
    module_logger.debug(f"data_dir {data_dir}")

    # Todo: add this to doc
    module_logger.debug(f"If you are wondering why data_dir is missing your root dir, troubleshoot tip here:"
                        f" https://stackoverflow.com/questions/1945920/why-doesnt-os-path-join-work-in-this-case")

    # Initialize the model and put it in eval mode for prediction
    model = initialize_model(cfg)
    model.eval()
    module_logger.debug("Model")
    print(model)
    print(f"Model Parameters: {(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6):.3f}M")

    # copied from https://github.com/funkelab/lsd_experiments/blob/master/hemi/02_train/setup03/train.py
    raw = ArrayKey('RAW')
    # experimental addition to prevent inference resin/blank input data
    # labels_mask = ArrayKey('LABELS_MASK')
    # initialise both here but request selectively based on model type
    pred_postpre_vectors = ArrayKey('PRED_POSTPRE_VECTORS')
    pred_post_indicator = ArrayKey('PRED_POST_INDICATOR')

    # must be cast as gunpowder Coordinates
    voxel_size = Coordinate(cfg.MODEL.VOXEL_SIZE)
    input_shape = Coordinate(cfg.MODEL.INPUT_SHAPE) + Coordinate(cfg.MODEL.GROW_INPUT)
    output_shape = Coordinate(cfg.MODEL.OUTPUT_SHAPE) + Coordinate(cfg.MODEL.GROW_INPUT)
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size
    # context added/removed
    context = (input_size - output_size) / 2
    module_logger.debug(f"input_size: {input_size}; output_size: {output_size}")

    # initialise inference dataset
    data_sources = Hdf5Source(
        cfg.DATA.SAMPLE,
        datasets={
            raw: 'volumes/raw',
            # experimental addition to prevent inference resin/blank input data
            # labels_mask: 'volumes/labels/labels_mask'
        },
        array_specs={
            raw: ArraySpec(interpolatable=True),
        }
    )

    # create an output roi anew based on context
    with build(data_sources):
        raw_roi = data_sources.spec[raw].roi
    total_output_roi = raw_roi.grow(-context, -context)
    module_logger.debug(f"Total output ROI {total_output_roi} and context {context}")

    # masking
    try:
        if cfg.DATA.MASK_FILE is not None and cfg.DATA.MASK_DS is not None:
            logging.info(f"Reading mask from {cfg.DATA.MASK_FILE} and {cfg.DATA.MASK_DS}")
            # mask = open_ds(cfg.DATA.MASK_FILE, cfg.DATA.MASK_DS) # just read the file, we cannot upsample here
            # mask_data = get_mask_data_in_roi(mask=mask, roi=raw_roi, target_voxel_size=voxel_size)

            mask = ArrayKey('MASK')
            mask_source = ZarrSource(
                cfg.DATA.MASK_FILE,
                datasets={
                    mask: cfg.DATA.MASK_DS,
                    # experimental addition to prevent inference resin/blank input data
                    # labels_mask: 'volumes/labels/labels_mask'
                },
                array_specs={
                    mask: ArraySpec(interpolatable=False),
                }
            )
        else:
            mask_source = None
    except Exception as e:
        print(e)

    # this is scan_request
    request = BatchRequest()
    request.add(raw, input_size)

    if cfg.TRAIN.MODEL_TYPE in ["SynMT1", "STMASK"]:
        request.add(pred_post_indicator, output_size)
    if cfg.TRAIN.MODEL_TYPE in ["SynMT1", "AFF"]:
        request.add(pred_postpre_vectors, output_size)

    # if not os.path.exists(cfg.DATA.OUTFILE):
    #     os.makedirs(os.path.dirname(cfg.DATA.OUTFILE), exist_ok=False)

    print(f"Saving outputs to {cfg.DATA.OUTFILE}")

    # Creating datasets in output zarr
    # Hard-code warning: the ds keys in the out-zarr are hardcoded for now, hence will ensure same output format
    out_raw = "volumes/raw"
    # print(out_raw)

    if cfg.TRAIN.MODEL_TYPE in ["SynMT1", "STMASK"]:
        out_lsds = "volumes/pred_indicator"

    if cfg.TRAIN.MODEL_TYPE in ["SynMT1", "AFF"]:
        out_affs = "volumes/pred_vectors"
        # if cfg.DATA.INVERT_PRED_AFFS and cfg.TRAIN.MODEL_TYPE in ["MTLSD", "AFF"]:
        #     # this will choose a max affinity channel, cast to uint8 and invert
        #     out_inv_affs = "volumes/inverted_pred_affs"

    train_pipeline = data_sources

    # train_pipeline += ZarrWrite(
    #     dataset_names={
    #         raw: out_raw},
    #     output_dir=os.path.dirname(cfg.DATA.OUTFILE),
    #     output_filename=os.path.basename(cfg.DATA.OUTFILE),
    #     dataset_dtypes={
    #         raw: ArraySpec(roi=raw_roi)})

    train_pipeline += Normalize(raw)

    train_pipeline += IntensityScaleShift(raw, cfg.MODEL.INTENSITYSCALESHIFT_SCALE[0],
                                          cfg.MODEL.INTENSITYSCALESHIFT_SHIFT[0])

    train_pipeline += Unsqueeze([raw])
    train_pipeline += Stack(1)
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
            0: pred_postpre_vectors,
        }

    module_logger.debug(f"cuda device {cfg.TRAIN.DEVICE}")
    train_pipeline += Predict(
        model=model,
        checkpoint=cfg.TRAIN.CHECKPOINT,
        inputs={
            'x': raw,  # key should as in the forward defined in the models.py
        },
        outputs=outputs,  # selectively pass output based on model type
        spawn_subprocess=True,
        device=cfg.TRAIN.DEVICE
    )
    # shape: b x c x d x h x w -->  c x d x h x w
    train_pipeline += Squeeze([raw])
    squeeze_output_list = [raw]
    # have to squeeze selectively now
    if cfg.TRAIN.MODEL_TYPE in ["SynMT1", "STMASK"]:
        squeeze_output_list.extend([pred_post_indicator])

    if cfg.TRAIN.MODEL_TYPE in ["SynMT1", "STVEC"]:
        squeeze_output_list.extend([pred_postpre_vectors])

    # raw shape: c x d x h x w ---> d x h x w;
    # affs/lsds: b x c x d x h x w --> c x d x h x w
    train_pipeline += Squeeze(squeeze_output_list)

    # property hard-coding for tests
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

    d_property = out_properties[
        'pred_partner_vectors'] if 'pred_partner_vectors' in out_properties else None
    m_property = out_properties[
    'pred_syn_indicator_out'] if 'pred_syn_indicator_out' in out_properties else None

    if cfg.MODEL.D_SCALE != 1 and cfg.MODEL.D_SCALE is not None:
        d_scale = cfg.MODEL.D_SCALE
        # if d_scale != 1 and d_scale is not None:
        train_pipeline += IntensityScaleShift(pred_postpre_vectors,
                                        1. / d_scale,
                                    0)  # Map back to nm world.
    if m_property is not None and 'scale' in m_property:
        if m_property['scale'] != 1:
            train_pipeline += IntensityScaleShift(pred_post_indicator,
                                            m_property['scale'], 0)
    if d_property is not None and 'scale' in d_property:
        train_pipeline += IntensityScaleShift(pred_postpre_vectors,
                                        d_property['scale'], 0)
    if d_property is not None and 'dtype' in d_property:
        assert d_property['dtype'] == 'int8' or d_property[
            'dtype'] == 'float32', 'predict not adapted to dtype {}'.format(
            d_property['dtype'])
        if d_property['dtype'] == 'int8':
            train_pipeline += IntensityScaleShiftClip(pred_postpre_vectors,
                                                1, 0, clip=(-128, 127))

    if cfg.TRAIN.MODEL_TYPE in ["SynMT1", "STMASK"]:
        train_pipeline += ZarrWrite(
            dataset_names={
                pred_post_indicator: out_lsds},
            output_dir=os.path.dirname(cfg.DATA.OUTFILE),
            output_filename=os.path.basename(cfg.DATA.OUTFILE),
            dataset_dtypes={
                pred_post_indicator: ArraySpec(roi=total_output_roi)})

    if cfg.TRAIN.MODEL_TYPE in ["SynMT1", "STVEC"]:
        train_pipeline += ZarrWrite(
            dataset_names={
                pred_postpre_vectors: out_affs},
            output_dir=os.path.dirname(cfg.DATA.OUTFILE),
            output_filename=os.path.basename(cfg.DATA.OUTFILE),
            dataset_dtypes={
                pred_postpre_vectors: ArraySpec(roi=total_output_roi)})

    # train_pipeline += Scan(request)

    if cfg.TRAIN.MODEL_TYPE in ["SynMT1", "STMASK"]:
        daisy_roi_map = {raw: "read_roi", pred_post_indicator: "write_roi"}
    if cfg.TRAIN.MODEL_TYPE in ["SynMT1", "STVEC"]:
        daisy_roi_map = {raw: "read_roi", pred_post_indicator: "write_roi", pred_postpre_vectors: "write_roi"}

    train_pipeline += DaisyRequestBlocks(
        request,
        roi_map=daisy_roi_map,
        num_workers=cfg.SYSTEM.NUM_WORKERS,
        block_done_callback=lambda b, s, d: block_done_callback(
            cfg.DATA.DB_HOST, cfg.DATA.DB_NAME, b, s, d, db_collection_name=cfg.DATA.DB_COLLECTION_NAME,
            worker_config=None
        ),
    )

    with build(train_pipeline) as b:
        # passing an empty request allows to scan through the input data automatically
        b.request_batch(BatchRequest())


if __name__ == '__main__':
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
