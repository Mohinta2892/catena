import os
import glob
import random
import pickle
import logging
from pathlib import Path
import argparse
import math


def create_split(data_dir, split_ratio, output_file, random_seed):
    """
    Splits HDF5 data files into training and validation sets for each class
    and saves the file paths to a pickle file.

    Args:
        data_dir (str): The root directory containing class subfolders.
        split_ratio (float): The proportion of data to be used for training.
        output_file (str): The path to save the output .pkl file.
        random_seed (int): The random seed for shuffling to ensure reproducibility.
    """
    logging.info(f"Scanning data directory: {data_dir}")

    # Discover classes from subdirectories
    try:
        neurotransmitter_classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    except FileNotFoundError:
        logging.error(f"Data directory not found: {data_dir}")
        return

    if not neurotransmitter_classes:
        logging.error(f"No subdirectories found in {data_dir}")
        return

    logging.info(f"Found classes: {neurotransmitter_classes}")

    random.seed(random_seed)

    # This dictionary will hold the final split
    split_dict = {'train': {}, 'val': {}}

    for class_name in neurotransmitter_classes:
        class_dir = os.path.join(data_dir, class_name)
        files = glob.glob(os.path.join(class_dir, '*.h*'))

        if not files:
            logging.warning(f"No HDF5 files found for class '{class_name}' in {class_dir}")
            continue

        random.shuffle(files)

        split_index = int(math.ceil(len(files) * split_ratio))
        train_files = files[:split_index]
        val_files = files[split_index:]

        split_dict['train'][class_name] = train_files
        split_dict['val'][class_name] = val_files

        logging.info(f"  Class '{class_name}': {len(train_files)} train, {len(val_files)} val")

    # Save the split dictionary to a pickle file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(split_dict, f)

    logging.info(f"Split information successfully saved to {output_path}")


def main():
    """Main function to run the script from the command line.
    To Run:
    --data_dir /media/samia/DATA/mounts/cephfs/catena/helpers/neurotransmitter/sylee_neurotrans_cubes_hemi2025/data_3d/train
    --split_ratio 0.8
    --output_file /media/samia/DATA/mounts/cephfs/catena/helpers/neurotransmitter/sylee_neurotrans_cubes_hemi2025/data_3d/train/data_split/data_split.pkl
    --seed 42
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Create train/validation split for neurotransmitter data.")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the root data directory with class subfolders.')
    parser.add_argument('--split_ratio', type=float, default=0.9, help='Ratio of data to use for training.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output pickle file.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()

    create_split(args.data_dir, args.split_ratio, args.output_file, args.seed)


if __name__ == '__main__':
    main()
