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
from models.models import *
from models.losses import *
from add_ons.gp.gp_utils import *
from add_ons.funlib_persistence.persistence_utils import *
import ast
from tqdm import tqdm
from glob import glob
# from config.config_predict import get_cfg_defaults  # import but do no use
import random
import torch
from funlib.persistence import prepare_ds

torch.backends.cudnn.benchmark = True

# we set a seed for reproducibility
torch.manual_seed(1961923)
np.random.seed(1961923)
random.seed(1961923)


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
    prepare_predict_datasets(cfg, dtype=np.uint8, voxel_size=voxel_size, ds_key=out_raw, ds_size=input_size,
                             total_roi=total_output_roi)
    print(out_raw)

    if cfg.TRAIN.MODEL_TYPE in ["MTLSD", "LSD"]:
        out_lsds = "volumes/pred_lsds"
        prepare_predict_datasets(cfg, dtype=np.float32, ds_key=out_lsds,
                                 ds_size=output_size, num_channels=10,
                                 total_roi=total_output_roi,
                                 voxel_size=voxel_size)  # shape C(10) x D x H x W

    if cfg.TRAIN.MODEL_TYPE in ["MTLSD", "AFF"]:
        out_affs = "volumes/pred_affs"
        prepare_predict_datasets(cfg, dtype=np.float32,
                                 ds_key=out_affs, ds_size=output_size,
                                 num_channels=len(
                                     cfg.TRAIN.NEIGHBORHOOD),
                                 total_roi=total_output_roi,
                                 voxel_size=voxel_size)

        if cfg.DATA.INVERT_PRED_AFFS and cfg.TRAIN.MODEL_TYPE in ["MTLSD", "AFF"]:
            # this will choose a max affinity channel, cast to uint8 and invert
            out_inv_affs = "volumes/inverted_pred_affs"
            prepare_predict_datasets(cfg, dtype=np.uint8, ds_key=out_inv_affs, ds_size=output_size,
                                     num_channels=1, total_roi=total_output_roi,
                                     voxel_size=voxel_size)

        train_pipeline = data_sources

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

        train_pipeline += Predict(
            model=model,
            checkpoint=cfg.TRAIN.CHECKPOINT,
            inputs={
                'x': raw,  # key should as in the forward defined in the models.py
            },
            outputs=outputs,  # selectively pass output based on model type
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

        train_pipeline += Scan(request)

        with build(train_pipeline) as b:
            # passing an empty request allows to scan through the input data automatically
            b.request_batch(BatchRequest())


if __name__ == '__main__':
    """
    cfg = get_cfg_defaults()

    if not os.path.exists('./train_config.yaml'):
        print("Please run train.py -h to pass minimal args. Model falls back to hard-coded augmentations."
              " You can make a config.yaml file instead to gain more control (add link to docs here)!")

        parser = argparse.ArgumentParser()
        parser.add_argument("--fib", default=0, type=int, help="If you want to train FIBSEM `isotropic` pass True,"
                                                               " default: 0:False")
        parser.add_argument("--device", default="cuda:6", type=str, help="Specify device you want to train on,"
                                                                         " default: `cuda:0`")
        parser.add_argument("--epochs", default=400000, type=int, help="Specify device you want to train on, "
                                                                       " default: `400000`")

        # tip: returns a dict like this to maintain access consistency above
        args = vars(parser.parse_args())

    else:
        # read yaml file in current directory
        with open('./train_config.yaml', 'r') as stream:
            yaml_data = yaml.load(stream, Loader=yaml.FullLoader)

            # args is now a dict
            args = yaml_data

    iterations = args['epochs']
    train_until(iterations, args)
    """
    pass
