import datetime
import math
import numpy as np
import os
from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *
# this needs to change for lsd > 0.1.3 : from lsd.train.gp import AddLocalShapeDescriptor
# from lsd.gp import AddLocalShapeDescriptor
from lsd.train.gp import AddLocalShapeDescriptor
import argparse
import yaml
# add current directory to path and allow absolute imports - this is a terrible for now
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from models.models import *
from models.losses import *
from add_ons.gp.gp_utils import *
from add_ons.gp.reject_if_empty import RejectIfEmpty
from add_ons.funlib_persistence.persistence_utils import *
import ast
from tqdm import tqdm
from glob import glob
from config.config_predict import get_cfg_defaults  # import but do no use
import random
import torch
from funlib.persistence import prepare_ds
import pymongo
from yacs.config import CfgNode as CN



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
    logging.basicConfig(filename=f"./logs/predict_scan_logs_{datetime.date}_{datetime.time}.txt",
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
    pred_affs = ArrayKey('PRED_AFFS')
    pred_lsds = ArrayKey('PRED_LSDS')

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
    data_sources = ZarrSource(
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

    # this is scan_request
    request = BatchRequest()
    request.add(raw, input_size)
    # experimental addition to prevent inference resin/blank input data
    # request.add(labels_mask, input_size)

    if cfg.TRAIN.MODEL_TYPE in ["MTLSD", "LSD"]:
        request.add(pred_lsds, output_size)
    if cfg.TRAIN.MODEL_TYPE in ["MTLSD", "AFF"]:
        request.add(pred_affs, output_size)

    # if not os.path.exists(cfg.DATA.OUTFILE):
    #     os.makedirs(os.path.dirname(cfg.DATA.OUTFILE), exist_ok=False)

    print(f"Saving outputs to {cfg.DATA.OUTFILE}")

    # Creating datasets in output zarr
    # Hard-code warning: the ds keys in the out-zarr are hardcoded for now, hence will ensure same output format
    out_raw = "volumes/raw"
    # print(out_raw)
    
    if cfg.TRAIN.MODEL_TYPE in ["MTLSD", "LSD"]:
        out_lsds = "volumes/pred_lsds"

    if cfg.TRAIN.MODEL_TYPE in ["MTLSD", "AFF"]:
        out_affs = "volumes/pred_affs"
        if cfg.DATA.INVERT_PRED_AFFS and cfg.TRAIN.MODEL_TYPE in ["MTLSD", "AFF"]:
            # this will choose a max affinity channel, cast to uint8 and invert
            out_inv_affs = "volumes/inverted_pred_affs"

    train_pipeline = data_sources
    # train_pipeline += RejectIfEmpty(gt=labels_mask, p=1) # experimental to skip through resin and blank chunks

    train_pipeline += ZarrWrite(
        dataset_names={
            raw: out_raw},
        output_dir=os.path.dirname(cfg.DATA.OUTFILE),
        output_filename=os.path.basename(cfg.DATA.OUTFILE),
        dataset_dtypes={
            raw: ArraySpec(roi=raw_roi)})

    train_pipeline += Normalize(raw)

    train_pipeline += IntensityScaleShift(raw, cfg.MODEL.INTENSITYSCALESHIFT_SCALE[0],
                                          cfg.MODEL.INTENSITYSCALESHIFT_SHIFT[0])

    train_pipeline += Unsqueeze([raw])
    train_pipeline += Stack(cfg.TRAIN.BATCH_SIZE)
    # customize the loss inputs and outputs here based on model type
    if cfg.TRAIN.MODEL_TYPE == "MTLSD":
        outputs = {
            0: pred_lsds,
            1: pred_affs
        }
    elif cfg.TRAIN.MODEL_TYPE == "LSD":
        outputs = {
            0: pred_lsds
        }

    elif cfg.TRAIN.MODEL_TYPE == "AFF":
        outputs = {
            0: pred_affs,
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
    if cfg.TRAIN.MODEL_TYPE in ["AFF", "MTLSD"]:
        squeeze_output_list.extend([pred_affs])

    if cfg.TRAIN.MODEL_TYPE in ["LSD", "MTLSD"]:
        squeeze_output_list.extend([pred_lsds])

    # raw shape: c x d x h x w ---> d x h x w;
    # affs/lsds: b x c x d x h x w --> c x d x h x w
    train_pipeline += Squeeze(squeeze_output_list)

    if cfg.TRAIN.MODEL_TYPE in ["MTLSD", "AFF"]:
        train_pipeline += ZarrWrite(
            dataset_names={
                pred_affs: out_affs},
            output_dir=os.path.dirname(cfg.DATA.OUTFILE),
            output_filename=os.path.basename(cfg.DATA.OUTFILE),
            dataset_dtypes={
                pred_affs: ArraySpec(roi=total_output_roi)})

    if cfg.DATA.INVERT_PRED_AFFS:
        train_pipeline += ChooseMaxAffinityValue(pred_affs)
        train_pipeline += EnsureUInt8(pred_affs)
        train_pipeline += InvertAffPred(pred_affs)

        train_pipeline += ZarrWrite(
            dataset_names={
                pred_affs: out_inv_affs},
            output_dir=os.path.dirname(cfg.DATA.OUTFILE),
            output_filename=os.path.basename(cfg.DATA.OUTFILE),
            dataset_dtypes={
                pred_affs: ArraySpec(roi=total_output_roi)})

    if cfg.TRAIN.MODEL_TYPE in ["MTLSD", "LSD"]:
        train_pipeline += ZarrWrite(
            dataset_names={
                pred_lsds: out_lsds},
            output_dir=os.path.dirname(cfg.DATA.OUTFILE),
            output_filename=os.path.basename(cfg.DATA.OUTFILE),
            dataset_dtypes={
                pred_lsds: ArraySpec(roi=total_output_roi)})

    # train_pipeline += Scan(request)

    if cfg.TRAIN.MODEL_TYPE in ["MTLSD", "AFF"]:
        daisy_roi_map = {raw: "read_roi", pred_affs: "write_roi"}
    if cfg.TRAIN.MODEL_TYPE in ["MTLSD", "LSD"]:
        daisy_roi_map = {raw: "read_roi", pred_affs: "write_roi", pred_lsds: "write_roi"}

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
    # have to use this since :
    # RuntimeError: Cannot re-initialize CUDA in forked subprocess.
    # To use CUDA with multiprocessing, you must use the 'spawn' start method
    # import torch
    # torch.multiprocessing.set_start_method('spawn')
    # import multiprocessing
    # multiprocessing.set_start_method('spawn')
    config_file = sys.argv[1]

    # parse the args file to become cfg
    cfg = CN()
    # Allow creating new keys recursively.: https://github.com/rbgirshick/yacs/issues/25
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file)
    predict(cfg)
