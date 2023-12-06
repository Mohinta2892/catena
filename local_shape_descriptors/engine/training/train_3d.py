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
from add_ons.gp.reject_if_empty import RejectIfEmpty
from add_ons.gp.gp_utils import EnsureUInt8
import ast
from tqdm import tqdm
from glob import glob
import random

# we set a seed for reproducibility
torch.manual_seed(1961923)
np.random.seed(1961923)
random.seed(1961923)
torch.backends.cudnn.benchmark = True


def train_until(max_iteration, cfg):
    logging.basicConfig(filename=f"./logs/train_logs.txt",  # always overwrite??
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        # set to logging.INFO for fewer details
                        level=logging.DEBUG if cfg.SYSTEM.VERBOSE else logging.INFO)
    module_logger = logging.getLogger(__name__)

    # Warning: Hard-coding, we know we must read `training` data from `data_3d`
    data_dir = os.path.join(cfg.DATA.HOME, cfg.DATA.DATA_DIR_PATH, cfg.DATA.BRAIN_VOL, "data_3d", "train")
    logger.debug(f"data_dir {data_dir}")

    # Todo: add this to doc
    module_logger.debug(f"If you are wondering why data_dir is missing your root dir, troubleshoot tip here:"
                        f" https://stackoverflow.com/questions/1945920/why-doesnt-os-path-join-work-in-this-case")

    #  list all files from the above path - assume they are zarrs
    samples = glob(f"{data_dir}/*.zarr")
    logger.debug(f"samples {samples}")

    # Check if preprocessed data exists
    if cfg.PREPROCESS.HISTOGRAM_MATCH is not None:
        """ This is set, hence the data files must exist in the preprocessed_3d directory!"""
        preprocessed_data_dir = os.path.join(cfg.DATA.HOME, cfg.DATA.DATA_DIR_PATH, "preprocessed_3d")
        # TODO: will be different when this a list with multiple BRAIN_VOLS
        preprocessed_samples = glob(f"{preprocessed_data_dir}/{cfg.DATA.BRAIN_VOL}_*/*.zarr")
        samples.extend(preprocessed_samples)

    # sanity check: view samples list
    print(f"Samples: \n {samples}")

    # Initialize the model
    if cfg.TRAIN.MODEL_TYPE not in ["ACLSD", "ACRLSD"]:
        model = initialize_model(cfg)
        calc_shape_obj = CalculateModelSummary(model, cfg)
        output_shape = calc_shape_obj.calculate_output_shape()[-len(cfg.MODEL.INPUT_SHAPE):]
    elif cfg.TRAIN.MODEL_TYPE in ["ACLSD", "ACRLSD"] and cfg.TRAIN.LSD_EPOCHS is None:
        # In auto-context mode, the in_channels for affinity == local shape descriptors returned
        model_lsd, model = initialize_model(cfg)
        # LSDModel needs to used in eval mode
        model_lsd.eval()
        calc_shape_obj = CalculateModelSummary(model_lsd, cfg)
        # expected output shape from LSD to be input to Aff
        pretrained_lsd_shape = calc_shape_obj.calculate_output_shape()  # BxCxDxHxW
        calc_shape_obj = CalculateModelSummary(model, cfg)
        output_shape = calc_shape_obj.calculate_output_shape(pretrained_lsd_shape)[-len(cfg.MODEL.INPUT_SHAPE):]
    elif cfg.TRAIN.MODEL_TYPE in ["ACLSD", "ACRLSD"] and cfg.TRAIN.LSD_EPOCHS is not None:
        # In auto-context mode, the in_channels for affinity == local shape descriptors returned
        model, _ = initialize_model(cfg)
        calc_shape_obj = CalculateModelSummary(model, cfg)
        # expected output shape from LSD to be input to Aff
        output_shape = Coordinate(*calc_shape_obj.calculate_output_shape()[-len(cfg.MODEL.INPUT_SHAPE):])  # BxCxDXHxW

    module_logger.debug("Model")
    print(f"Model Parameters: {(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6):.3f}M")
    # initialize the loss
    loss = initialize_loss(cfg)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.TRAIN.INITIAL_LR,
        betas=cfg.TRAIN.LR_BETAS)

    # copied from https://github.com/funkelab/lsd_experiments/blob/master/hemi/02_train/setup03/train.py
    raw = ArrayKey('RAW')
    labels = ArrayKey('GT_LABELS')

    pred_affs = ArrayKey('PRED_AFFS')
    gt_affs = ArrayKey('GT_AFFS')
    affs_weights = ArrayKey('AFFS_WEIGHTS')
    affs_mask = ArrayKey('GT_AFFINITIES_MASK')
    pred_lsds = ArrayKey('PRED_LSDS')
    gt_lsds = ArrayKey('GT_LSDS')
    lsds_weights = ArrayKey('LSDS_WEIGHTS')
    # it is mandatory to have mask now, however a way should be found to reduce RAM use with input masks
    # TODO: find inside gunpowder
    labels_mask = ArrayKey('GT_LABELS_MASK')

    if cfg.TRAIN.MODEL_TYPE in ["ACLSD", "ACRLSD"] and cfg.TRAIN.LSD_EPOCHS is None:
        pretrained_lsd = ArrayKey("PRETRAINED_LSD")

    # must be cast as gunpowder Coordinates
    voxel_size = Coordinate(cfg.MODEL.VOXEL_SIZE)
    input_shape = Coordinate(cfg.MODEL.INPUT_SHAPE)
    # assert (cfg.TRAIN.LSD_EPOCHS is None) == (output_shape == cfg.MODEL.OUTPUT_SHAPE), \
    #     "When cfg.TRAIN.LSD_EPOCHS is None, output_shape should be equal to cfg.MODEL.OUTPUT_SHAPE."

    if cfg.TRAIN.LSD_EPOCHS is None:
        output_shape = Coordinate(cfg.MODEL.OUTPUT_SHAPE)

    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(gt_lsds, output_size)
    request.add(lsds_weights, output_size)
    # if we do not do this selectively, snapshot requests every key otherwise
    # TODO: could replace with `ZarrWrite()`, check dacapo.
    request.add(gt_affs, output_size)
    request.add(affs_weights, output_size)
    request.add(affs_mask, output_size)
    if cfg.TRAIN.MODEL_TYPE == "MTLSD":
        request.add(pred_affs, output_size)
        request.add(pred_lsds, output_size)
    elif cfg.TRAIN.MODEL_TYPE in ["ACLSD", "ACRLSD"] and cfg.TRAIN.LSD_EPOCHS is not None:
        request.add(pred_lsds, output_size)
    elif cfg.TRAIN.MODEL_TYPE in ["ACLSD", "ACRLSD"] and cfg.TRAIN.LSD_EPOCHS is None:
        pretrained_lsd_size = Coordinate(*pretrained_lsd_shape[2:]) * voxel_size
        request.add(pretrained_lsd, pretrained_lsd_size)
        request.add(pred_affs, output_size)
    elif cfg.TRAIN.MODEL_TYPE == "LSD":
        request.add(pred_lsds, output_size)
    elif cfg.TRAIN.MODEL_TYPE == "AFF":
        request.add(pred_affs, output_size)

    # Assume worst case (rotation augmentation by 45 degrees) and pad
    # by half the length of the diagonal of the network output size
    # Copied from https://github.com/funkelab/lsd_experiments/blob/master/fib25/02_train/setup02/train.py

    p = int(round(np.sqrt(np.sum([i * i for i in output_shape])) / 2))

    # Ensure that our padding is the closest multiple of our resolution
    labels_padding = Coordinate([j * round(i / j) for i, j in zip([p, p, p], list(voxel_size))])
    print('Labels padding:', labels_padding)

    # all input volumes are assumed to have been saved as .zarr
    # todo: add conversion scripts documentation link here
    data_sources = tuple(
        ZarrSource(
            os.path.join(data_dir, sample),
            datasets={
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
                labels_mask: 'volumes/labels/labels_mask',
            },
            array_specs={
                raw: ArraySpec(interpolatable=True),
                labels: ArraySpec(interpolatable=False),
                labels_mask: ArraySpec(interpolatable=False)
            }
        ) +
        Normalize(raw) +
        Pad(raw, None) +
        Pad(labels, labels_padding) +
        Pad(labels_mask, labels_padding) +
        # RandomLocation(min_masked=0.5, mask=labels_mask) +  # always 20% masked in
        RandomLocation() +  # without masking
        RejectIfEmpty(gt=labels, p=1)  # reject empty batches where labels/gt are fully empty
        for sample in samples
    )

    train_pipeline = data_sources

    # chooses data sources randomly
    train_pipeline += RandomProvider()

    if cfg.TRAIN.AUGMENT:
        # TODO: ElasticAugment is replaced by DefectAugment from version 1.3.0 (PattonW), deform() is broken!
        train_pipeline += ElasticAugment(
            control_point_spacing=cfg.MODEL.CONTROL_POINT_SPACING,
            jitter_sigma=cfg.MODEL.JITTER_SIGMA[0] if isinstance(cfg.MODEL.JITTER_SIGMA,
                                                                 list) else cfg.MODEL.JITTER_SIGMA,
            rotation_interval=cfg.MODEL.ROTATION_INTERVAL,
            prob_slip=cfg.MODEL.PROB_SLIP[0] if isinstance(cfg.MODEL.PROB_SLIP, list) else cfg.MODEL.PROB_SLIP,
            prob_shift=cfg.MODEL.PROB_SHIFT[0] if isinstance(cfg.MODEL.PROB_SHIFT, list) else cfg.MODEL.PROB_SHIFT,
            max_misalign=cfg.MODEL.MAX_MISALIGN[0] if isinstance(cfg.MODEL.MAX_MISALIGN,
                                                                 list) else cfg.MODEL.MAX_MISALIGN,
            subsample=cfg.MODEL.SUBSAMPLE)

        train_pipeline += SimpleAugment(transpose_only=cfg.MODEL.TRANSPOSE)

        # double elastic deformation is applied as per LSD paper for FIBSEM datasets:Supplementary Table 10
        if cfg.DATA.FIB:
            train_pipeline += ElasticAugment(
                control_point_spacing=cfg.MODEL.CONTROL_POINT_SPACING,
                jitter_sigma=cfg.MODEL.JITTER_SIGMA[0] if isinstance(cfg.MODEL.JITTER_SIGMA,
                                                                     list) else cfg.MODEL.JITTER_SIGMA,
                rotation_interval=cfg.MODEL.ROTATION_INTERVAL,
                prob_slip=cfg.MODEL.PROB_SLIP[0] if isinstance(cfg.MODEL.PROB_SLIP, list) else cfg.MODEL.PROB_SLIP,
                prob_shift=cfg.MODEL.PROB_SHIFT[0] if isinstance(cfg.MODEL.PROB_SHIFT, list) else cfg.MODEL.PROB_SHIFT,
                max_misalign=cfg.MODEL.MAX_MISALIGN[0] if isinstance(cfg.MODEL.MAX_MISALIGN,
                                                                     list) else cfg.MODEL.MAX_MISALIGN,
                subsample=cfg.MODEL.SUBSAMPLE)

        train_pipeline += IntensityAugment(raw, cfg.MODEL.INTENSITYAUG_SCALE_MIN, cfg.MODEL.INTENSITYAUG_SCALE_MAX,
                                           cfg.MODEL.INTENSITYAUG_SHIFT_MIN, cfg.MODEL.INTENSITYAUG_SHIFT_MAX)

    train_pipeline += GrowBoundary(
        labels,
        mask=labels_mask,
        steps=cfg.MODEL.GROWBOUNDARY_STEPS)

    train_pipeline += AddLocalShapeDescriptor(
        labels,
        gt_lsds,
        lsds_mask=lsds_weights,  # was mask in 0.1, now lsds_mask from > 0.1.3
        sigma=cfg.MODEL.LSD_SIGMA,
        downsample=cfg.MODEL.LSD_DOWNSAMPLE)

    train_pipeline += AddAffinities(
        cfg.TRAIN.NEIGHBORHOOD,
        labels=labels,
        labels_mask=labels_mask,
        affinities=gt_affs,
        affinities_mask=affs_mask)

    train_pipeline += BalanceLabels(
        gt_affs,
        affs_weights,
        affs_mask)

    if cfg.TRAIN.AUGMENT:
        train_pipeline += IntensityScaleShift(raw, cfg.MODEL.INTENSITYSCALESHIFT_SCALE[0],
                                              cfg.MODEL.INTENSITYSCALESHIFT_SHIFT[0])

    train_pipeline += Unsqueeze([raw])
    train_pipeline += Stack(cfg.TRAIN.BATCH_SIZE)

    train_pipeline += PreCache(
        cache_size=cfg.SYSTEM.CACHE_SIZE,
        num_workers=cfg.SYSTEM.NUM_WORKERS)

    # customize the loss inputs and outputs here based on model type
    if cfg.TRAIN.MODEL_TYPE == "MTLSD":
        loss_inputs = {
            0: pred_lsds,
            1: gt_lsds,
            2: lsds_weights,
            3: pred_affs,
            4: gt_affs,
            5: affs_weights
        }

        outputs = {
            0: pred_lsds,
            1: pred_affs
        }
        snapshot_ds = {
            raw: 'raw',
            labels: 'labels',
            gt_affs: 'gt_affs',
            gt_lsds: 'gt_lsds',
            pred_affs: 'pred_affs',
            pred_lsds: 'pred_lsds'
        }
    elif cfg.TRAIN.MODEL_TYPE == "LSD":
        loss_inputs = {
            0: pred_lsds,
            1: gt_lsds,
            2: lsds_weights
        }

        outputs = {
            0: pred_lsds
        }
        snapshot_ds = {
            raw: 'raw',
            labels: 'labels',
            gt_affs: 'gt_affs',
            gt_lsds: 'gt_lsds',
            pred_lsds: 'pred_lsds'
        }

    elif cfg.TRAIN.MODEL_TYPE == "AFF":
        loss_inputs = {
            0: pred_affs,
            1: gt_affs,
            2: affs_weights
        }

        outputs = {
            0: pred_affs
        }

        snapshot_ds = {
            raw: 'raw',
            labels: 'labels',
            gt_affs: 'gt_affs',
            gt_lsds: 'gt_lsds',
            pred_affs: 'pred_affs'
        }

    if cfg.TRAIN.MODEL_TYPE in ["ACLSD", "ACRLSD"] and cfg.TRAIN.LSD_EPOCHS is not None:

        loss_inputs = {
            0: pred_lsds,
            1: gt_lsds,
            2: lsds_weights
        }

        outputs = {
            0: pred_lsds
        }
        snapshot_ds = {
            raw: 'raw',
            labels: 'labels',
            gt_affs: 'gt_affs',
            gt_lsds: 'gt_lsds',
            pred_lsds: 'pred_lsds'}

        train = Train(
            model=model,
            loss=loss,
            optimizer=optimizer,
            inputs={
                # key should as in the forward defined in the models.py
                'x': raw
            },
            loss_inputs=loss_inputs,  # selectively pass loss input based on model type
            outputs=outputs,  # selectively pass output based on model type
            save_every=cfg.TRAIN.SAVE_EVERY,
            log_dir=cfg.MODEL.LOG_DIR,
            device=cfg.TRAIN.DEVICE,
            checkpoint_folder=cfg.MODEL.CKPT_FOLDER,
            use_wandb=cfg.PREPROCESS.USE_WANDB)  # set this flag

    elif cfg.TRAIN.MODEL_TYPE in ["ACLSD", "ACRLSD"] and cfg.TRAIN.LSD_EPOCHS is None:
        loss_inputs = {
            0: pred_affs,
            1: gt_affs,
            2: affs_weights
        }

        outputs = {
            0: pred_affs
        }

        snapshot_ds = {
            raw: 'raw',
            labels: 'labels',
            gt_affs: 'gt_affs',
            gt_lsds: 'gt_lsds',
            pred_affs: 'pred_affs'
        }

        # LSDModel() should be in eval() model
        predict_lsd_node = Predict(
            model=model_lsd,
            checkpoint=cfg.TRAIN.CHECKPOINT_AC,
            inputs={'x': raw},
            outputs={0: pretrained_lsd},
        )

        train_pipeline += predict_lsd_node + EnsureUInt8(pretrained_lsd) + Normalize(pretrained_lsd)

        # model should AFFModel() now
        train = Train(
            model=model,
            loss=loss,
            optimizer=optimizer,
            inputs={
                # key should as in the forward defined in the models.py
                'x': raw if cfg.TRAIN.MODEL_TYPE not in ["ACLSD", "ACRLSD"] else pretrained_lsd
            },
            loss_inputs=loss_inputs,  # selectively pass loss input based on model type
            outputs=outputs,  # selectively pass output based on model type
            save_every=cfg.TRAIN.SAVE_EVERY,
            log_dir=cfg.MODEL.LOG_DIR,
            device=cfg.TRAIN.DEVICE,
            checkpoint_folder=cfg.MODEL.CKPT_FOLDER,
            use_wandb=cfg.PREPROCESS.USE_WANDB)  # set this flag

    else:
        train = Train(
            model=model,
            loss=loss,
            optimizer=optimizer,
            inputs={
                # key should as in the forward defined in the models.py
                'x': raw if cfg.TRAIN.MODEL_TYPE not in ["ACLSD", "ACRLSD"] else pretrained_lsd
            },
            loss_inputs=loss_inputs,  # selectively pass loss input based on model type
            outputs=outputs,  # selectively pass output based on model type
            save_every=cfg.TRAIN.SAVE_EVERY,
            log_dir=cfg.MODEL.LOG_DIR,
            device=cfg.TRAIN.DEVICE,
            checkpoint_folder=cfg.MODEL.CKPT_FOLDER,
            use_wandb=cfg.PREPROCESS.USE_WANDB)  # set this flag

    train_pipeline += train

    # shape: b x c x d x h x w -->  c x d x h x w
    train_pipeline += Squeeze([raw], axis=None)
    squeeze_output_list = [raw, gt_affs, gt_lsds]
    # have to squeeze selectively now
    if cfg.TRAIN.MODEL_TYPE in ["AFF", "MTLSD"]:
        squeeze_output_list.extend([pred_affs])

    elif cfg.TRAIN.MODEL_TYPE in ["LSD", "MTLSD"]:
        squeeze_output_list.extend([pred_lsds])

    # raw shape: c x d x h x w ---> d x h x w;
    # affs/lsds: b x c x d x h x w --> c x d x h x w
    train_pipeline += Squeeze(squeeze_output_list, axis=None)

    train_pipeline += IntensityScaleShift(raw, cfg.MODEL.INTENSITYSCALESHIFT_SCALE[1],
                                          cfg.MODEL.INTENSITYSCALESHIFT_SHIFT[1])

    train_pipeline += Snapshot(snapshot_ds,
                               dataset_dtypes={
                                   labels: np.uint64,
                                   gt_affs: np.float32
                               },
                               every=cfg.TRAIN.SAVE_EVERY,
                               output_dir=cfg.MODEL.OUTPUT_DIR,
                               output_filename='batch_{iteration}.zarr'  # default: snapshot filename
                               )

    with build(train_pipeline) as b:
        for i in (
                pbar := tqdm(range(train.iteration, max_iteration), desc=f"Model resumed from {train.iteration}",
                             )):
            batch = b.request_batch(request)
            pbar.set_postfix({"loss": batch.loss})


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
