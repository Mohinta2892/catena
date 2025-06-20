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


class Accuracy(BatchFilter):
    """
    Accumulates and logs the model accuracy, both overall and per-class.
    """
    def __init__(self, prediction_key, label_key, class_to_label, log_every=100, summary_writer=None):
        self.prediction_key = prediction_key
        self.label_key = label_key
        self.class_to_label = class_to_label
        self.label_to_class = {v: k for k, v in self.class_to_label.items()}
        self.log_every = log_every
        self.summary_writer = summary_writer

        self.num_classes = len(class_to_label)
        self.counter = 0

        # Initialize counters for overall and per-class stats
        self._reset_counters()

    def _reset_counters(self):
        """Resets all statistics counters to zero."""
        self.total_correct = 0
        self.total_count = 0
        self.class_correct = {i: 0 for i in range(self.num_classes)}
        self.class_total = {i: 0 for i in range(self.num_classes)}

    def process(self, batch, request):
        predictions = batch[self.prediction_key].data.argmax(axis=1)
        labels = batch[self.label_key].data

        self.total_count += len(labels)
        self.total_correct += np.sum(predictions == labels)

        # Update per-class stats
        for i in range(len(labels)):
            label_val = labels[i]
            pred_val = predictions[i]
            self.class_total[label_val] += 1
            if pred_val == label_val:
                self.class_correct[label_val] += 1

        self.counter += 1

        if self.counter > 0 and self.counter % self.log_every == 0:

            # Log overall accuracy
            overall_accuracy = self.total_correct / self.total_count if self.total_count > 0 else 0
            logging.info(f"Accuracy at iteration {self.counter}: {overall_accuracy:.4f}")
            if self.summary_writer:
                self.summary_writer.add_scalar("accuracy/overall", overall_accuracy, self.counter)

            # Log per-class accuracy
            for i in range(self.num_classes):
                class_name = self.label_to_class[i]
                total = self.class_total[i]
                correct = self.class_correct[i]
                accuracy = correct / total if total > 0 else 0
                logging.info(f"  - Accuracy for class '{class_name}' ({i}): {accuracy:.4f} ({correct}/{total})")
                if self.summary_writer:
                    self.summary_writer.add_scalar(f"accuracy/{class_name}", accuracy, self.counter)

            # Reset for the next logging interval
            self._reset_counters()

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
            input_fmaps=1
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

            locations = f['annotations/locations'][:]
            types = f['annotations/types'][:]

            presynaptic_indices = np.where(types == b'presynaptic_site')[0]
            presynaptic_locations = locations[presynaptic_indices]

            if presynaptic_locations.shape[0] == 0:
                module_logger.warning(f"No pre-synaptic locations found in {hdf_file}, skipping.")
                return None

    except Exception as e:
        module_logger.error(f"Could not read data from {hdf_file}: {e}")
        return None

    raw_source = Hdf5Source(
        hdf_file,
        datasets={raw_key: 'volumes/raw'},
        array_specs={raw_key: ArraySpec(interpolatable=True, voxel_size=voxel_size)}
    )

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

    input_shape = Coordinate(cfg.MODEL_VGG.INPUT_SIZE if cfg.TRAIN.MODEL_TYPE == 'VGG' else cfg.MODEL_RESNET.INPUT_SIZE)
    voxel_size = Coordinate(cfg.DATA.VOXEL_SIZE)
    input_size = input_shape * voxel_size

    raw = ArrayKey("RAW")
    label = ArrayKey("LABEL")
    prediction = ArrayKey("PREDICTION")

    data_dir = os.path.join(cfg.DATA.HOME, cfg.DATA.DATA_DIR_PATH, cfg.DATA.BRAIN_VOL, "data_3d", "train")

    neurotransmitter_classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_label = {name: i for i, name in enumerate(neurotransmitter_classes)}
    num_transmitters = len(neurotransmitter_classes)

    if num_transmitters == 0:
        raise RuntimeError(f"No data folders found in {data_dir}")
    module_logger.info(f"Found {num_transmitters} classes: {class_to_label}")

    device = torch.device(cfg.TRAIN.DEVICE)
    model = initialize_model(cfg, num_classes=num_transmitters)
    model.to(device)

    # --- Log Model Summary ---
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    module_logger.info(f"Model: {model}")
    module_logger.info(f"Number of trainable parameters: {num_params / 1e6:.2f}M")

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.INITIAL_LR)

    module_logger.info(f"Input shape (voxels): {input_shape}")
    module_logger.info(f"Voxel size: {voxel_size}")
    module_logger.info(f"Input size (nm): {input_size}")

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

    pipeline += Normalize(raw)
    pipeline += SimpleAugment()
    pipeline += IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=False)
    pipeline += IntensityScaleShift(raw, 2, -1)
    pipeline += PreCache(num_workers=cfg.SYSTEM.NUM_WORKERS, cache_size=cfg.SYSTEM.CACHE_SIZE)

    pipeline += Stack(cfg.TRAIN.BATCH_SIZE)
    pipeline += Unsqueeze([raw], axis=1)

    train_node = Train(
        model, loss, optimizer,
        inputs={'x': raw},
        loss_inputs={0: prediction, 1: label},
        outputs={0: prediction},
        array_specs={prediction: ArraySpec(nonspatial=True)},
        save_every=cfg.TRAIN.SAVE_EVERY,
        log_dir=cfg.LOGGING.LOG_DIR,
        checkpoint_folder=cfg.MODEL.CKPT_DIR,
    )
    pipeline += train_node

    pipeline += Accuracy(
        prediction, label, class_to_label, log_every=100, summary_writer=train_node.summary_writer
    )
    pipeline += Squeeze([raw])
    pipeline += IntensityScaleShift(raw, 0.5, 0.5)
    pipeline += Snapshot(
        {raw: "raw", label: "label"},
        output_filename="snapshot_{iteration}.zarr",
        every=cfg.TRAIN.SNAPSHOT_EVERY,
        output_dir=cfg.LOGGING.SNAPSHOT_DIR,
    )

    request = BatchRequest()
    request.add(raw, input_size)
    request[label] = ArraySpec(nonspatial=True)
    request[prediction] = ArraySpec(nonspatial=True)

    module_logger.info("Starting training...")
    max_iteration = cfg.TRAIN.EPOCHS
    with build(pipeline):
        for i in tqdm(range(train_node.iteration, max_iteration), total=max_iteration, initial=train_node.iteration):
            pipeline.request_batch(request)
