import logging
import numpy as np
import os
import h5py
import pickle
from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *
from tqdm import tqdm
from glob import glob
import random
import math

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
        self.iteration_counter = 0
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

        for i in range(len(labels)):
            label_val = labels[i]
            pred_val = predictions[i]
            self.class_total[label_val] += 1
            if pred_val == label_val:
                self.class_correct[label_val] += 1

        self.iteration_counter += 1

        if self.iteration_counter > 0 and self.iteration_counter % self.log_every == 0:
            overall_accuracy = self.total_correct / self.total_count if self.total_count > 0 else 0
            logging.info(f"[TRAIN] Overall Accuracy at iteration {self.iteration_counter}: {overall_accuracy:.4f}")
            if self.summary_writer:
                global_step = batch.iteration
                self.summary_writer.add_scalar("accuracy/train_overall", overall_accuracy, global_step)

            for i in range(self.num_classes):
                class_name = self.label_to_class[i]
                total = self.class_total[i]
                correct = self.class_correct[i]
                accuracy = correct / total if total > 0 else 0
                logging.info(f"  - [TRAIN] Accuracy for class '{class_name}': {accuracy:.4f} ({correct}/{total})")
                if self.summary_writer:
                    self.summary_writer.add_scalar(f"accuracy/train_{class_name}", accuracy, global_step)

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

    # --- Load Data ---
    train_files_dict = {}
    if cfg.DATA.USE_SPLIT_FILE:
        module_logger.info(f"Attempting to load data split from {cfg.DATA.SPLIT_FILE}")
        try:
            with open(cfg.DATA.SPLIT_FILE, 'rb') as f:
                split_dict = pickle.load(f)
            train_files_dict = split_dict['train']
        except FileNotFoundError:
            module_logger.error(f"Split file not found at {cfg.DATA.SPLIT_FILE} but USE_SPLIT_FILE was True. Aborting.")
            return
    else:
        module_logger.info("USE_SPLIT_FILE is False. Scanning data directory for all files...")
        data_dir = os.path.join(cfg.DATA.HOME, cfg.DATA.DATA_DIR_PATH, cfg.DATA.BRAIN_VOL, "data_3d", "train")
        neurotransmitter_classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        for class_name in neurotransmitter_classes:
            class_dir = os.path.join(data_dir, class_name)
            files = glob(os.path.join(class_dir, '*.h*'))
            train_files_dict[class_name] = files

    class_to_label = {name: i for i, name in enumerate(sorted(train_files_dict.keys()))}
    num_transmitters = len(class_to_label)

    if num_transmitters == 0:
        raise RuntimeError(f"No training data found. Check your configuration and data paths.")
    module_logger.info(f"Found {num_transmitters} classes: {class_to_label}")

    device = torch.device(cfg.TRAIN.DEVICE)
    model = initialize_model(cfg, num_classes=num_transmitters)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    module_logger.info(f"Model: {model}")
    module_logger.info(f"Number of trainable parameters: {num_params / 1e6:.2f}M")

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.INITIAL_LR)

    module_logger.info(f"Input shape (voxels): {input_shape}, Voxel size: {voxel_size}, Input size (nm): {input_size}")

    sources = []
    for class_name, file_list in train_files_dict.items():
        label_id = class_to_label[class_name]
        for hdf_file in tqdm(file_list, desc=f"Loading training files for {class_name}"):
            source = create_data_source(hdf_file, raw, voxel_size)
            if source:
                sources.append(source + AddLabel(label, label_id))

    if not sources:
        raise RuntimeError("No valid data sources found from split file. Check your data and paths.")

    train_pipeline = (
            tuple(sources) + RandomProvider() +
            Normalize(raw) + SimpleAugment() +
            IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=False) +
            IntensityScaleShift(raw, 2, -1) +
            PreCache(num_workers=cfg.SYSTEM.NUM_WORKERS, cache_size=cfg.SYSTEM.CACHE_SIZE) +
            Stack(cfg.TRAIN.BATCH_SIZE) + Unsqueeze([raw], axis=1)
    )

    train_node = Train(
        model, loss, optimizer,
        inputs={'x': raw}, loss_inputs={0: prediction, 1: label}, outputs={0: prediction},
        array_specs={prediction: ArraySpec(nonspatial=True)},
        save_every=cfg.TRAIN.SAVE_EVERY, log_dir=cfg.LOGGING.LOG_DIR,
        checkpoint_folder=cfg.LOGGING.CKPT_DIR
    )

    train_pipeline += train_node
    train_pipeline += Accuracy(prediction, label, class_to_label, log_every=100, summary_writer=train_node.summary_writer)
    train_pipeline += Snapshot({raw: "raw", label: "label"}, output_filename="snapshot_train_{iteration}.zarr", every=cfg.TRAIN.SNAPSHOT_EVERY, output_dir=cfg.LOGGING.SNAPSHOT_DIR)

    request = BatchRequest()
    request.add(raw, input_size)
    request[label] = ArraySpec(nonspatial=True)
    request[prediction] = ArraySpec(nonspatial=True)

    module_logger.info("Starting training...")
    max_iteration = cfg.TRAIN.EPOCHS
    with build(train_pipeline):
        for i in tqdm(range(train_node.iteration, max_iteration), total=max_iteration, initial=train_node.iteration):
            train_pipeline.request_batch(request)
