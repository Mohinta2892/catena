import torch
import h5py
import numpy as np
import pickle
import os
import argparse
import logging
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from yacs.config import CfgNode as CN
from gunpowder import *
from gunpowder.torch import Predict

# Assuming your models are in a directory that can be imported
from models.vgg3d import Vgg3D
from models.resnet3d import ResNet3D


class CollectPredictions(BatchFilter):
    """
    A custom gunpowder node to collect batch data into a list.
    This is useful for gathering all predictions during inference.
    """

    def __init__(self, prediction_array, label_array, class_names):
        self.prediction_array = prediction_array
        self.label_array = label_array
        self.class_names = class_names
        self.results = []

    def process(self, batch, request):
        probabilities = batch[self.prediction_array].data
        true_label_id = batch[self.label_array].data.item()

        # Get the ROI to find the center coordinate
        roi = request[self.prediction_array].roi
        center_nm = roi.get_center()

        result_row = {
            'true_label': self.class_names[true_label_id],
            'location_z': center_nm[0],
            'location_y': center_nm[1],
            'location_x': center_nm[2],
            **{f'prob_{name}': prob for name, prob in zip(self.class_names, probabilities[0])}
        }
        self.results.append(result_row)


def create_data_source(hdf_file, raw_key, voxel_size):
    """
    Creates a Gunpowder data source for a given HDF5 file, providing
    patches centered on pre-synaptic locations.
    (This function is copied from train_3d.py for consistency)
    """
    module_logger = logging.getLogger(__name__)

    try:
        with h5py.File(hdf_file, 'r') as f:
            if 'volumes/raw' not in f:
                module_logger.warning(f"'volumes/raw' not found in {hdf_file}, skipping file.")
                return None, 0
            if 'annotations/locations' not in f or 'annotations/types' not in f:
                module_logger.warning(f"Annotations not found in {hdf_file}, skipping file.")
                return None, 0

            locations = f['annotations/locations'][:]
            types = f['annotations/types'][:]
            presynaptic_indices = np.where(types == b'presynaptic_site')[0]
            presynaptic_locations = locations[presynaptic_indices]

            if presynaptic_locations.shape[0] == 0:
                module_logger.warning(f"No pre-synaptic locations found in {hdf_file}, skipping.")
                return None, 0

    except Exception as e:
        module_logger.error(f"Could not read data from {hdf_file}: {e}")
        return None, 0

    raw_source = Hdf5Source(
        hdf_file,
        datasets={raw_key: 'volumes/raw'},
        array_specs={raw_key: ArraySpec(interpolatable=True, voxel_size=voxel_size)}
    )

    # Create a gunpowder source that provides all specified locations
    location_source = SpecifiedLocation(presynaptic_locations)

    source_node = (
            (raw_source, location_source) +
            MergeProvider() +
            Pad(raw_key, None)
    )

    return source_node, len(presynaptic_locations)


def initialize_model(cfg, num_classes):
    """
    Initializes the classification model based on the configuration.
    """
    model_type = cfg.TRAIN.MODEL_TYPE
    if model_type == 'VGG':
        return Vgg3D(
            input_size=cfg.MODEL_VGG.INPUT_SIZE,
            output_classes=num_classes,
            fmaps=cfg.MODEL_VGG.FMAPS,
            downsample_factors=cfg.MODEL_VGG.DOWNSAMPLE_FACTORS,
            fmap_inc=cfg.MODEL_VGG.FMAP_INC,
            n_convolutions=cfg.MODEL_VGG.N_CONVOLUTIONS,
            input_fmaps=1
        )
    elif model_type == 'RESNET':
        return ResNet3D(
            output_classes=num_classes,
            input_channels=cfg.MODEL_RESNET.INPUT_CHANNELS,
            start_channels=cfg.MODEL_RESNET.START_CHANNELS
        )
    raise ValueError(f"Unknown model type: {model_type}")


def predict(args, cfg):
    """
    Run inference on the validation set from the split file using Gunpowder.
    """
    logging.info("Starting inference...")

    # --- Load Data Split ---
    with open(args.split_file, 'rb') as f:
        split_dict = pickle.load(f)

    val_files_dict = split_dict['val']
    class_names = sorted(val_files_dict.keys())
    class_to_label = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    logging.info(f"Found {num_classes} validation classes: {class_names}")

    # --- Setup Model ---
    device = torch.device(cfg.TRAIN.DEVICE)
    model = initialize_model(cfg, num_classes)

    logging.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # --- Define Gunpowder Keys and request ---
    raw = ArrayKey('RAW')
    prediction = ArrayKey('PREDICTION')
    label = ArrayKey('LABEL')
    patch_shape = Coordinate(cfg.MODEL_VGG.INPUT_SIZE if cfg.TRAIN.MODEL_TYPE == 'VGG' else cfg.MODEL_RESNET.INPUT_SIZE)
    voxel_size = Coordinate(cfg.DATA.VOXEL_SIZE)
    input_size = patch_shape * voxel_size

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(prediction, (1, num_classes))  # Request shape for non-spatial output
    request[label] = ArraySpec(nonspatial=True)

    # --- Create sources and count total samples ---
    sources = []
    total_samples = 0
    for class_name, file_list in val_files_dict.items():
        label_id = class_to_label[class_name]
        for hdf_file in file_list:
            source_node, num_locs = create_data_source(hdf_file, raw, voxel_size)
            if source_node:
                sources.append(source_node + AddLabel(label, label_id))
                total_samples += num_locs

    if not sources:
        logging.error("No valid validation sources could be created.")
        return

    pipeline = tuple(sources) + RandomProvider()

    # --- Define the Inference Pipeline ---
    pipeline += Normalize(raw)
    pipeline += Unsqueeze([raw])
    pipeline += Predict(
        model=model,
        inputs={'x': raw},
        outputs={prediction: prediction},
        array_specs={prediction: ArraySpec(nonspatial=True)},
    )
    # Add a custom node to collect the predictions
    collection_node = CollectPredictions(prediction, label, class_names)
    pipeline += collection_node
    # Iterate over all locations provided by the sources
    pipeline += Scan(reference=request)

    # --- Run Inference ---
    logging.info(f"Running inference on {total_samples} samples...")
    with build(pipeline):
        for i in tqdm(range(total_samples), desc="Running inference"):
            pipeline.request_batch(request)

    # --- Save Results ---
    results_df = pd.DataFrame(collection_node.results)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logging.info(f"Inference results saved to {output_path}")


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Run inference on held-out neurotransmitter data.")
    parser.add_argument('--config_file', type=str, required=True, help='Path to the project config file.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint file (.pt).')
    parser.add_argument('--split_file', type=str, required=True,
                        help='Path to the .pkl file containing the train/val split.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output CSV with predictions.')
    args = parser.parse_args()

    # Load config
    cfg = CN()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    predict(args, cfg)


if __name__ == '__main__':
    main()
