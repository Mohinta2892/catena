import math
import numpy as np
import os
from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *
from lsd.gp import AddLocalShapeDescriptor
import argparse
import yaml
from models import *
from losses import *
import ast
from tqdm import tqdm
from glob import glob

logging.basicConfig(level=logging.DEBUG)  # set to logging.INFO for fewer details
module_logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True


# using Long Range `LR` affinities just so to test mutex, which apparently works better with LR affinities
# neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1], [-3, 0, 0], [0, -3, 0], [0, 0, -3],
#                  [-9, 0, 0], [0, -9, 0], [0, 0, -9]]


def unpack_hyperparams(args):
    """ Explicit unpacking for hyper-params passed via the config.yaml file"""
    if args["data"]["fib"]:
        in_channels = args["data_augmentation_iso"]["in_channels"]
        num_fmaps = args["data_augmentation_iso"]["num_fmaps"]
        fmap_inc_factor = args["data_augmentation_iso"]["fmap_inc_factor"]
        downsample_factors = args["data_augmentation_iso"]["downsample_factors"]
        num_fmaps_out = args["data_augmentation_iso"]["num_fmaps_out"]
        kernel_size_down = args["data_augmentation_iso"]["kernel_size_down"]

        kernel_size_up = args["data_augmentation_iso"]["kernel_size_up"]
        control_point_spacing = args["data_augmentation_iso"]["control_point_spacing"]
        jitter_sigma = args["data_augmentation_iso"]["jitter_sigma"]
        prob_slip = args["data_augmentation_iso"]["prob_slip"]
        prob_shift = args["data_augmentation_iso"]["prob_shift"]
        max_misalign = args["data_augmentation_iso"]["max_misalign"]
        transpose = args["data_augmentation_iso"]["transpose"]
        input_shape = Coordinate(args["data_augmentation_iso"]["input_shape"])
        output_shape = Coordinate(args["data_augmentation_iso"]["output_shape"])
        voxel_size = Coordinate(args["data_augmentation_iso"]["voxel_size"])
        if 'log_dir' not in args["data_augmentation_iso"].keys() or len(args["data_augmentation_iso"]["log_dir"]):
            log_dir = '/home/log-3dmtlsd-hemi-onlyori'
        if 'ckpt_folder' not in args["data_augmentation_iso"].keys() or len(
                args["data_augmentation_iso"]["ckpt_folder"]):
            checkpoint_folder = "/home/checkpoints-3dmtlsd-hemi-onlyori"
        if 'output_dir' not in args["data_augmentation_iso"].keys() or len(
                args["data_augmentation_iso"]["output_dir"]):
            output_dir = "/home/snapshots_-3dmtlsd-hemi-onlyori"

    else:
        in_channels = args["data_augmentation_aniso"]["in_channels"]
        num_fmaps = args["data_augmentation_aniso"]["num_fmaps"]
        fmap_inc_factor = args["data_augmentation_aniso"]["fmap_inc_factor"]
        downsample_factors = args["data_augmentation_aniso"]["downsample_factors"]
        num_fmaps_out = args["data_augmentation_aniso"]["num_fmaps_out"]
        kernel_size_down = args["data_augmentation_aniso"]["kernel_size_down"]

        kernel_size_up = args["data_augmentation_aniso"]["kernel_size_up"]
        control_point_spacing = args["data_augmentation_aniso"]["control_point_spacing"]
        jitter_sigma = args["data_augmentation_aniso"]["jitter_sigma"]
        prob_slip = args["data_augmentation_aniso"]["prob_slip"]
        prob_shift = args["data_augmentation_aniso"]["prob_shift"]
        max_misalign = args["data_augmentation_aniso"]["max_misalign"]
        transpose = args["data_augmentation_aniso"]["transpose"]
        input_shape = Coordinate(args["data_augmentation_aniso"]["input_shape"])
        output_shape = Coordinate(args["data_augmentation_aniso"]["output_shape"])
        voxel_size = Coordinate(args["data_augmentation_aniso"]["voxel_size"])
        if 'log_dir' not in args["data_augmentation_aniso"].keys() or len(args["data_augmentation_aniso"]["log_dir"]):
            log_dir = '/home/log-3dmtlsd-cremi-cropped-z30-170'
        if 'ckpt_folder' not in args["data_augmentation_aniso"].keys() or len(
                args["data_augmentation_aniso"]["ckpt_folder"]):
            checkpoint_folder = "/home/checkpoints_3dmtlsd-cremi-cropped-z30-170"
        if 'output_dir' not in args["data_augmentation_aniso"].keys() or len(
                args["data_augmentation_aniso"]["output_dir"]):
            output_dir = "/home/snapshots_3dmtlsd-cremi-cropped-z30-170"

    neighbourhood = args["train"]["neighbourhood"]
    batch_size = args["train"]["batch_size"]
    save_every = args["train"]["save_every"]


def train_until(max_iteration, args):
    data_dir = os.path.join("/", args["data"]["home"], args["data"]["data_dir_path"], args["data"]["brain_vol"])
    # todo: add this to doc
    module_logger.debug(f"If you are wondering why data_dir is missing your root dir, troubleshoot tip here:"
                        f" https://stackoverflow.com/questions/1945920/why-doesnt-os-path-join-work-in-this-case")

    #  list all files from the above path - assume they are zarrs
    samples = glob(f"{data_dir}/*.zarr")

    if args["data"]["fib"]:
        """Instantiate with isotropic kernel sizes"""
        model = MtlsdModel(
            args["data_augmentation_iso"]["in_channels"],
            args["data_augmentation_iso"]["num_fmaps"],
            args["data_augmentation_iso"]["fmap_inc_factor"],
            ast.literal_eval(args["data_augmentation_iso"]["downsample_factors"]),
            ast.literal_eval(args["data_augmentation_iso"]["kernel_size_down"]),
            ast.literal_eval(args["data_augmentation_iso"]["kernel_size_up"]),
            args["data_augmentation_iso"]["num_fmaps_out"])
    else:
        """Instantiate with anisotropic kernel sizes"""
        model = MtlsdModel(
            args["data_augmentation_aniso"]["in_channels"],  # int
            args["data_augmentation_aniso"]["num_fmaps"],  # int
            args["data_augmentation_aniso"]["fmap_inc_factor"],  # int
            ast.literal_eval(args["data_augmentation_aniso"]["downsample_factors"]),  # list(lists)
            ast.literal_eval(args["data_augmentation_aniso"]["kernel_size_down"]),  # list(lists)
            ast.literal_eval(args["data_augmentation_aniso"]["kernel_size_up"]),
            args["data_augmentation_aniso"]["num_fmaps_out"])  # int

    module_logger.debug("Model")
    print(model)
    print(f"Model Parameters: {(sum(p.numel() for p in model.parameters() if p.requires_grad))}")
    loss = WeightedMSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ast.literal_eval(args["optimizer"]["initial_lr"]),
        betas=ast.literal_eval(args["optimizer"]["betas"]))

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
    labels_mask = ArrayKey('GT_LABELS_MASK')

    if args["data"]["fib"]:
        # in/out shapes are different if in/out vols as isotropic or anisotropic
        voxel_size = ast.literal_eval(args["data_augmentation_iso"]["voxel_size"])
        input_shape = ast.literal_eval(args["data_augmentation_iso"]["input_shape"])
        output_shape = ast.literal_eval(args["data_augmentation_iso"]["output_shape"])
    else:
        # in/out shapes are different if in/out vols as isotropic or anisotropic
        voxel_size = ast.literal_eval(args["data_augmentation_aniso"]["voxel_size"])
        input_shape = ast.literal_eval(args["data_augmentation_aniso"]["input_shape"])
        output_shape = ast.literal_eval(args["data_augmentation_aniso"]["output_shape"])

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
        RandomLocation(min_masked=0.5, mask=labels_mask)
        # RandomLocation() # without masking
        for sample in samples
    )

    train_pipeline = data_sources

    train_pipeline += RandomProvider()

    train_pipeline += ElasticAugment(
        control_point_spacing=control_point_spacing,
        jitter_sigma=jitter_sigma,
        rotation_interval=[0, math.pi / 2.0],
        prob_slip=prob_slip,
        prob_shift=prob_shift,
        max_misalign=max_misalign,
        subsample=8)

    train_pipeline += SimpleAugment(transpose_only=transpose)  # transpose_only=[1, 2]

    if int(args.fib):
        train_pipeline += ElasticAugment(
            control_point_spacing=control_point_spacing,
            jitter_sigma=(2, 2, 2),
            rotation_interval=[0, math.pi / 2.0],
            prob_slip=0.1,
            prob_shift=0.1,
            max_misalign=1,
            subsample=8)

    train_pipeline += IntensityAugment(raw, 0.8, 1.2, -0.2, 0.2)
    train_pipeline += GrowBoundary(
        labels,
        steps=1)

    train_pipeline += AddLocalShapeDescriptor(
        labels,
        gt_lsds,
        mask=lsds_weights,
        sigma=80,
        downsample=2)

    train_pipeline += AddAffinities(
        neighborhood,
        labels=labels,
        affinities=gt_affs,
        affinities_mask=affs_mask)

    train_pipeline += BalanceLabels(
        gt_affs,
        affs_weights,
        affs_mask)

    train_pipeline += IntensityScaleShift(raw, 2, -1)

    train_pipeline += Unsqueeze([raw])
    train_pipeline += Stack(batch_size)

    train_pipeline += PreCache(
        cache_size=40,
        num_workers=10)

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
        save_every=1000,
        log_dir=log_dir,
        device=args.device,
        checkpoint_folder=checkpoint_folder)

    train_pipeline += Squeeze([raw])
    train_pipeline += Squeeze([raw, gt_affs, pred_affs, gt_lsds, pred_lsds])

    train_pipeline += IntensityScaleShift(raw, 0.5, 0.5)

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
        every=100,
        output_dir=output_dir,
        output_filename='batch_{iteration}.zarr'  # snapshot filename
    )

    with build(train_pipeline) as b:
        for i in tqdm(range(max_iteration)):
            b.request_batch(request)


if __name__ == '__main__':

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
