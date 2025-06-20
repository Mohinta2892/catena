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
import datetime
from funlib.learn.torch.models import Vgg3D


# we set a seed for reproducibility
r_seed = 1961923
torch.manual_seed(r_seed)
np.random.seed(r_seed)
random.seed(r_seed)
torch.backends.cudnn.benchmark = False  # set this to False, when deterministic=True
torch.backends.cudnn.deterministic = True


class AddLabel(gp.BatchFilter):

    def __init__(self, array_key, label):
        self.array_key = array_key
        self.label = label

    def setup(self):

        self.provides(self.array_key, gp.ArraySpec(nonspatial=True))

    def prepare(self, request):
        pass

    def process(self, batch, request):

        array = gp.Array(np.array(self.label), spec=gp.ArraySpec(nonspatial=True))
        batch = gp.Batch()
        batch[self.array_key] = array
        return batch


class Accuracy(gp.BatchFilter):
    """Accumulate the accuracy of the model on the batches."""

    def __init__(self, prediction_key, label_key, log_every=100, summary_writer=None):
        self.prediction_key = prediction_key
        self.label_key = label_key
        self.total = 0
        self.correct = 0
        self.counter = 0
        self.log_every = log_every
        self.summary_writer = summary_writer

    def setup(self):
        pass

    def prepare(self, request):
        pass

    def process(self, batch, request):
        prediction = batch[self.prediction_key].data.argmax(axis=1)
        label = batch[self.label_key].data
        self.total += len(label)
        self.correct += np.sum(prediction == label)
        self.counter += 1
        if self.counter % self.log_every == 0:
            # Reset the counters
            if self.summary_writer:
                self.summary_writer.add_scalar(
                    "accuracy", self.correct / self.total, self.counter
                )
            self.total = 0
            self.correct = 0
        return batch


def train(
        max_iteration, cfg
        # model=None,
        # experiment_dir: Path = None,
        # file_path: str = None,
        # num_iterations=100000,
        # batch_size=8,
        # save_every=50000,
        # snapshot_every=50000,
        # log_every=100,
        # input_shape=(80, 80, 80),
        # lr=1e-4,
        # num_cache_workers=12,
        # # Data
        # container: str = None,
        # dataset: str = None,
        # voxel_size=(8, 8, 8),
        # num_transmitters: int = 3,
        # coordinate_order: str = "zyx",
):
    """
    Train a classifier model on the synapse data.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    file_path : str
        Path to the feather file containing the ground truth data.
    log_dir : str
        Directory to save the logs.
    num_iterations : int
        Number of iterations to train the model.
    batch_size : int
        Batch size.
    save_every : int
        Save the model every `save_every` iterations.
    snapshot_every : int
        Save the snapshots every `snapshot_every` iterations.
    log_every : int
        Log the training every `log_every` iterations.
    input_shape : tuple
        Shape of the input to the model.
    voxel_size : tuple
        Voxel size of the input data.
    lr : float
        Learning rate.
    coordinate_order : str
        Order of the coordinates to use, e.g. "zyx" or "xyz".
    """
    # Metadata
    input_shape = gp.Coordinate(input_shape)
    voxel_size = gp.Coordinate(voxel_size)
    input_size = input_shape * voxel_size

    # logging.info("Reading data...")
    # df = pd.read_feather(file_path)
    # # Multiply z, y, x by voxel size
    # df[["z", "y", "x"]] = df[["z", "y", "x"]].mul(voxel_size)
    # # Make sure that we have the right number of transmitters
    # assert len(df["neurotransmitter"].unique()) == num_transmitters

    logging.info(f"Input shape (voxels): {input_shape}")
    logging.info(f"Voxel size: {voxel_size}")
    logging.info(f"Input size (nm): {input_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    logging.info("Setting up pipeline...")
    raw = gp.ArrayKey("RAW")
    label = gp.ArrayKey("LABEL")
    prediction = gp.ArrayKey("PREDICTION")

    # read data
    # Warning: Hard-coding, we know we must read `training` data from `data_3d`
    # all sub-folders with transmitters should be under train potentially. Example: data_3d/train/acetylcholine/*.hdf
    data_dir = os.path.join(cfg.DATA.HOME, cfg.DATA.DATA_DIR_PATH, cfg.DATA.BRAIN_VOL, "data_3d", "train")

    logging.info("Creating pipeline...")
    # this will be a list of all hdf files for all transmitters
    neurotransmitter_sources = glob(f"{data_dir}/**/*.hdf")
    module_logger.debug(f"samples {samples}")

    # build the sources with one-hot labels?

    # # _________________________________Sampling Transmitters______________________________________
    # for neurotransmitter in tqdm(range(num_transmitters), total=num_transmitters):
    #     # Get coordinates of a certain transmitter type
    #     synapse_locations = df[df["neurotransmitter"] == neurotransmitter]
    #     #
    #     coordinates = synapse_locations[list(coordinate_order)]
    #     locations = coordinates.values

        # neurotransmitter_source = gp.ZarrSource(
        #     container,
        #     datasets={raw: dataset},
        #     array_specs={raw: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size)},
        # )
        #
        # if len(locations) > 0:
        #     neurotransmitter_source += gp.SpecifiedLocation(
        #         locations, choose_randomly=True, jitter=(10, 10, 10)
        #     )
        #
        #     neurotransmitter_source += AddLabel(label, neurotransmitter)

        # neurotransmitter_sources.append(neurotransmitter_source)

    pipeline = tuple(neurotransmitter_sources) + gp.RandomProvider()

    # _________________________________Rest of Process______________________________________
    pipeline += gp.Normalize(raw)
    # NOTE: We do not use a elastic/deformation augmentation here
    # This is because this augmentation is slow, and we assume that there are enough
    # separate data points to learn from.
    # In case you have a small dataset, you might want to add this augmentation.
    # pipeline += gp.DeformAugment(
    #     control_point_spacing=(40, 40, 40),
    #     jitter_sigma=(5.0, 5.0, 5.0),
    #     graph_raster_voxel_size=(1, 1, 1),
    #     spatial_dims=3,
    #     subsample=1,
    # )
    pipeline += gp.SimpleAugment()
    pipeline += gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)
    pipeline += gp.IntensityScaleShift(raw, 2, -1)
    pipeline += gp.PreCache(num_workers=num_cache_workers)
    # add a channel dimension to raw
    pipeline += gp.Stack(batch_size)
    pipeline += gp.Unsqueeze([raw], axis=1)

    train = gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={0: raw},
        loss_inputs={0: prediction, 1: label},
        outputs={0: prediction},
        array_specs={prediction: gp.ArraySpec(nonspatial=True)},
        save_every=save_every,
        log_dir=log_dir,
        log_every=log_every,
        checkpoint_basename=checkpoint_basename,
    )
    pipeline += train
    pipeline += Accuracy(
        prediction, label, log_every, summary_writer=train.summary_writer
    )

    pipeline += gp.IntensityScaleShift(
        raw, 0.5, 0.5
    )  # Scale back to original values for visualization
    pipeline += gp.Snapshot(
        {raw: "raw", label: "label", prediction: "prediction"},
        output_filename="snapshot_{iteration}.zarr",
        every=snapshot_every,
        output_dir=snapshot_dir,
    )

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request[label] = gp.ArraySpec(nonspatial=True)
    request[prediction] = gp.ArraySpec(nonspatial=True)

    logging.info("Training model...")
    with gp.build(pipeline):
        for i in tqdm(range(num_iterations), total=num_iterations):
            batch = pipeline.request_batch(request)
