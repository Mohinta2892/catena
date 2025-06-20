import logging
import numpy as np
import os
import h5py
from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *
from tqdm import tqdm
from glob import glob
import random

# Import models from your models directory
from models.vgg3d import Vgg3D
from models.resnet3d import ResNet3D

# Set a seed for reproducibility
r_seed = 1961923
torch.manual_seed(r_seed)
np.random.seed(r_seed)
random.seed(r_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class AddLabel(BatchFilter):
    """Adds a specified label to the batch."""

    def __init__(self, array_key, label):
        self.array_key = array_key
        self.label = label

    def setup(self):
        self.provides(self.array_key, ArraySpec(nonspatial=True))

    def process(self, batch, request):
        batch[self.array_key] = Array(np.array(self.label, dtype=np.int64), spec=ArraySpec(nonspatial=True))
        return batch

#
# class AddLabel(BatchFilter):
#
#     def __init__(self, array_key, label):
#         self.array_key = array_key
#         self.label = label
#
#     def setup(self):
#
#         self.provides(self.array_key, ArraySpec(nonspatial=True))
#
#     def prepare(self, request):
#         pass
#
#     def process(self, batch, request):
#
#         array = Array(np.array(self.label), spec=ArraySpec(nonspatial=True))
#         batch = Batch() # add this dummy batch instead, why?
#         batch[self.array_key] = array
#         return batch

class Accuracy(BatchFilter):
    """Accumulates and logs the model accuracy."""

    def __init__(self, prediction_key, label_key, log_every=100, summary_writer=None):
        self.prediction_key = prediction_key
        self.label_key = label_key
        self.total = 0
        self.correct = 0
        self.counter = 0
        self.log_every = log_every
        self.summary_writer = summary_writer

    def process(self, batch, request):
        prediction = batch[self.prediction_key].data.argmax(axis=1)
        label = batch[self.label_key].data
        self.total += len(label)
        self.correct += np.sum(prediction == label)
        self.counter += 1
        if self.counter > 0 and self.counter % self.log_every == 0:
            accuracy = self.correct / self.total if self.total > 0 else 0
            if self.summary_writer:
                self.summary_writer.add_scalar("accuracy", accuracy, self.counter)
            logging.info(f"Accuracy at iteration {self.counter}: {accuracy:.4f}")
            self.total = 0
            self.correct = 0
        return batch


def initialize_model(cfg, num_classes):
    """
    Initializes the classification model based on the configuration.
    """
    model_type = cfg.TRAIN.MODEL_TYPE
    logging.info(f"Initializing model of type: {model_type}")

    if model_type == 'VGG':
        model = Vgg3D(
            input_size=cfg.MODEL_VGG.INPUT_SIZE,
            fmaps=cfg.MODEL_VGG.FMAPS,
            downsample_factors=cfg.MODEL_VGG.DOWNSAMPLE_FACTORS,
            fmap_inc=cfg.MODEL_VGG.FMAP_INC,
            n_convolutions=cfg.MODEL_VGG.N_CONVOLUTIONS,
            output_classes=num_classes,
            input_fmaps=8
        )
    elif model_type == 'RESNET':
        model = ResNet3D(
            output_classes=num_classes,
            input_channels=cfg.MODEL_RESNET.INPUT_CHANNELS,
            start_channels=cfg.MODEL_RESNET.START_CHANNELS
        )
    else:
        raise ValueError(f"Unknown model type specified in config: {model_type}")

    return model


def create_data_source(hdf_file, raw_key, voxel_size):
    """
    Creates a Gunpowder data source for a given HDF5 file, centering
    patches on pre-synaptic locations.
    """
    module_logger = logging.getLogger(__name__)

    try:
        with h5py.File(hdf_file, 'r') as f:
            if 'volumes/raw' not in f:
                module_logger.warning(f"'volumes/raw' not found in {hdf_file}, skipping file.")
                return None
            if 'annotations/locations' not in f or 'annotations/types' not in f:
                module_logger.warning(f"Annotations not found in {hdf_file}, skipping file.")
                return None

            # Read locations and types
            locations = f['annotations/locations'][:]
            types = f['annotations/types'][:]

            # Filter for pre-synaptic sites
            # Assuming 'presynaptic_site' is encoded as 1, adjust if necessary
            presynaptic_indices = np.where(types == b'presynaptic_site')[0]
            presynaptic_locations = locations[presynaptic_indices]

            if presynaptic_locations.shape[0] == 0:
                module_logger.warning(f"No pre-synaptic locations found in {hdf_file}, skipping.")
                return None

    except Exception as e:
        module_logger.error(f"Could not read data from {hdf_file}: {e}")
        return None

    # Create a source for the raw data
    raw_source = Hdf5Source(
        hdf_file,
        datasets={raw_key: 'volumes/raw'},
        array_specs={raw_key: ArraySpec(interpolatable=True, voxel_size=voxel_size)}
    )

    # Combine raw source with specified pre-synaptic locations
    source_node = (
            raw_source +
            Pad(raw_key, None) +
            SpecifiedLocation(presynaptic_locations, choose_randomly=True, jitter=Coordinate((10, 10, 10)))
    )

    return source_node


def train_until(cfg):
    """
    Main training function.
    """
    module_logger = logging.getLogger(__name__)

    # --- Extract parameters from config ---
    input_shape = Coordinate(cfg.MODEL_VGG.INPUT_SIZE if cfg.TRAIN.MODEL_TYPE == 'VGG' else cfg.MODEL_RESNET.INPUT_SIZE)
    voxel_size = Coordinate(cfg.DATA.VOXEL_SIZE)
    input_size = input_shape * voxel_size

    # --- Define Gunpowder ArrayKeys ---
    raw = ArrayKey("RAW")
    label = ArrayKey("LABEL")
    prediction = ArrayKey("PREDICTION")

    # --- Data Loading ---
    data_dir = os.path.join(cfg.DATA.HOME, cfg.DATA.DATA_DIR_PATH, cfg.DATA.BRAIN_VOL, "data_3d", "train")

    neurotransmitter_classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_label = {name: i for i, name in enumerate(neurotransmitter_classes)}
    num_transmitters = len(neurotransmitter_classes)

    if num_transmitters == 0:
        raise RuntimeError(f"No data folders found in {data_dir}")
    module_logger.info(f"Found {num_transmitters} classes: {class_to_label}")

    # --- Model, Loss, and Optimizer Setup ---
    device = torch.device(cfg.TRAIN.DEVICE)
    model = initialize_model(cfg, num_classes=num_transmitters)
    model.to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.INITIAL_LR)

    module_logger.info(f"Input shape (voxels): {input_shape}")
    module_logger.info(f"Voxel size: {voxel_size}")
    module_logger.info(f"Input size (nm): {input_size}")

    # --- Gunpowder Pipeline Construction ---
    sources = []
    for class_name, label_id in class_to_label.items():
        class_dir = os.path.join(data_dir, class_name)
        hdf_files = glob(os.path.join(class_dir, '*.h*'))
        for hdf_file in tqdm(hdf_files, desc=f"loading hdf files for {class_name}", total=len(hdf_files)):
            source = create_data_source(hdf_file, raw, voxel_size)
            if source:
                source += AddLabel(label, label_id)
                sources.append(source)

    if not sources:
        raise RuntimeError("No valid data sources found. Check your data and paths.")

    pipeline = tuple(sources) + RandomProvider()

    # --- Augmentations ---
    pipeline += Normalize(raw)
    pipeline += SimpleAugment()
    pipeline += IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=False)
    pipeline += IntensityScaleShift(raw, 2, -1)
    pipeline += PreCache(num_workers=cfg.SYSTEM.NUM_WORKERS, cache_size=cfg.SYSTEM.CACHE_SIZE)

    # --- Prepare for training ---
    pipeline += Stack(cfg.TRAIN.BATCH_SIZE)
    pipeline += Unsqueeze([raw], axis=1)

    train_node = Train(
        model, loss, optimizer,
        inputs={'x': raw},
        loss_inputs={0: prediction, 1: label},
        outputs={0: prediction},
        array_specs={prediction: ArraySpec(nonspatial=True)},
        save_every=cfg.TRAIN.SAVE_EVERY,
        log_dir=cfg.LOGGING.LOG_DIR,  # TODO: this needs to change to checkpoint_dir
    )
    pipeline += train_node

    # --- Monitoring and Snapshots ---
    pipeline += Accuracy(
        prediction, label, log_every=100, summary_writer=train_node.summary_writer
    )
    pipeline += Squeeze([raw])
    pipeline += IntensityScaleShift(raw, 0.5, 0.5)
    pipeline += Snapshot(
        {raw: "raw", label: "label"},
        output_filename="snapshot_{iteration}.zarr",
        every=cfg.TRAIN.SNAPSHOT_EVERY,
        output_dir=cfg.LOGGING.SNAPSHOT_DIR,
    )

    # --- Request and Run Training ---
    request = BatchRequest()
    request.add(raw, input_size)
    request[label] = ArraySpec(nonspatial=True)
    request[prediction] = ArraySpec(nonspatial=True)

    module_logger.info("Starting training...")
    max_iteration = cfg.TRAIN.EPOCHS
    with build(pipeline):
        for i in tqdm(range(train_node.iteration, max_iteration), total=max_iteration, initial=train_node.iteration):
            pipeline.request_batch(request)
