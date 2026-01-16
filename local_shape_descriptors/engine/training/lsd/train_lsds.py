"""
Adapted from Arlo Sheridan's pytorch example.
Note: We have to create separate run scripts for training lsd, affs and mtlsd, ac(r)lsd models
since the io pipeline differs for them.
"""
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
import ast
from tqdm import tqdm
from glob import glob
from config.config import get_cfg_defaults  # import but do no use

# we set a seed for reproducibility
torch.manual_seed(0)


def train_until(max_iteration, cfg):
    logging.basicConfig(filename=f"train_logs.txt",
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG if cfg.SYSTEM.VERBOSE else logging.INFO)  # set to logging.INFO for fewer details
    module_logger = logging.getLogger(__name__)
    torch.backends.cudnn.benchmark = True

    data_dir = os.path.join(cfg.DATA.HOME, cfg.DATA.DATA_DIR_PATH, cfg.DATA.BRAIN_VOL)
    logger.debug(f"data_dir {data_dir}")

    # Todo: add this to doc
    module_logger.debug(f"If you are wondering why data_dir is missing your root dir, troubleshoot tip here:"
                        f" https://stackoverflow.com/questions/1945920/why-doesnt-os-path-join-work-in-this-case")

    #  list all files from the above path - assume they are zarrs
    samples = glob(f"{data_dir}/*.zarr")
    logger.debug(f"samples {samples}")

    # Initialize the model
    model = MtlsdModel(
        in_channels=cfg.MODEL.IN_CHANNELS,
        num_fmaps=cfg.MODEL.NUM_FMAPS,
        fmap_inc_factor=cfg.MODEL.FMAP_INC_FACTOR,
        downsample_factors=cfg.MODEL.DOWNSAMPLE_FACTORS,
        kernel_size_down=cfg.MODEL.KERNEL_SIZE_DOWN,
        kernel_size_up=cfg.MODEL.KERNEL_SIZE_UP,
        num_fmaps_out=cfg.MODEL.NUM_FMAPS_OUT,
        nhood=len(cfg.TRAIN.NEIGHBORHOOD),
        use_2d=cfg.DATA.DIM_2D)

    module_logger.debug("Model")
    print(model)
    print(f"Model Parameters: {(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6):.3f}M")
    loss = WeightedMSELoss()

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

    # must be cast as gunpowder Coordinates
    voxel_size = Coordinate(cfg.MODEL.VOXEL_SIZE)
    input_shape = Coordinate(cfg.MODEL.INPUT_SHAPE)
    output_shape = Coordinate(cfg.MODEL.OUTPUT_SHAPE)
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(gt_lsds, output_size)
    request.add(lsds_weights, output_size)
    request.add(pred_lsds, output_size)
    request.add(gt_affs, output_size)
    request.add(affs_weights, output_size)
    request.add(affs_mask, output_size)
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
        RandomLocation(min_masked=0.5, mask=labels_mask)  # always 50% masked
        # RandomLocation() # without masking
        for sample in samples
    )

    train_pipeline = data_sources

    train_pipeline += RandomProvider()

    train_pipeline += ElasticAugment(
        control_point_spacing=cfg.MODEL.CONTROL_POINT_SPACING,
        jitter_sigma=cfg.MODEL.JITTER_SIGMA[0] if isinstance(cfg.MODEL.JITTER_SIGMA, list) else cfg.MODEL.JITTER_SIGMA,
        rotation_interval=cfg.MODEL.ROTATION_INTERVAL,
        prob_slip=cfg.MODEL.PROB_SLIP[0] if isinstance(cfg.MODEL.PROB_SLIP, list) else cfg.MODEL.PROB_SLIP,
        prob_shift=cfg.MODEL.PROB_SHIFT[0] if isinstance(cfg.MODEL.PROB_SHIFT, list) else cfg.MODEL.PROB_SHIFT,
        max_misalign=cfg.MODEL.MAX_MISALIGN[0] if isinstance(cfg.MODEL.MAX_MISALIGN, list) else cfg.MODEL.MAX_MISALIGN,
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
        affinities=gt_affs,
        affinities_mask=affs_mask)

    train_pipeline += BalanceLabels(
        gt_affs,
        affs_weights,
        affs_mask)

    train_pipeline += IntensityScaleShift(raw, cfg.MODEL.INTENSITYSCALESHIFT_SCALE[0],
                                          cfg.MODEL.INTENSITYSCALESHIFT_SHIFT[0])

    train_pipeline += Unsqueeze([raw])
    train_pipeline += Stack(cfg.TRAIN.BATCH_SIZE)

    train_pipeline += PreCache(
        cache_size=cfg.SYSTEM.CACHE_SIZE,
        num_workers=cfg.SYSTEM.NUM_WORKERS)

    train_pipeline += Train(
        model=model,
        loss=loss,
        optimizer=optimizer,
        inputs={
            'x': raw,  # key should as in the forward defined in the models.py
        },
        loss_inputs={
            0: pred_lsds,
            1: gt_lsds,
            2: lsds_weights,
            3: pred_affs,
            4: gt_affs,
            5: affs_weights
        },
        outputs={
            0: pred_lsds,
            1: pred_affs
        },
        save_every=cfg.TRAIN.SAVE_EVERY,
        log_dir=cfg.MODEL.LOG_DIR,
        device=cfg.TRAIN.DEVICE,
        checkpoint_folder=cfg.MODEL.CKPT_FOLDER)

    train_pipeline += Squeeze([raw])
    train_pipeline += Squeeze([raw, gt_affs, pred_affs, gt_lsds, pred_lsds])

    train_pipeline += IntensityScaleShift(raw, cfg.MODEL.INTENSITYSCALESHIFT_SCALE[1],
                                          cfg.MODEL.INTENSITYSCALESHIFT_SHIFT[1])

    train_pipeline += Snapshot({
        raw: 'raw',
        labels: 'labels',
        gt_affs: 'gt_affs',
        gt_lsds: 'gt_lsds',
        pred_affs: 'pred_affs',
        pred_lsds: 'pred_lsds'
    },
        dataset_dtypes={
            labels: np.uint64,
            gt_affs: np.float32
        },
        every=cfg.TRAIN.SAVE_EVERY,
        output_dir=cfg.MODEL.OUTPUT_DIR,
        output_filename='batch_{iteration}.zarr'  # default: snapshot filename
    )

    with build(train_pipeline) as b:
        for i in tqdm(range(max_iteration)):
            b.request_batch(request)


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
