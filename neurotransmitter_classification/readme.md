# Synister Re-Implementation for Neurotransmitter Classification

### Neurotransmitter Classification with Synister
This is a very initial reimplementation of the Synister project. It has been restructured to work with local datasets that have been carefully curated from publicly available adult fly brain datasets. These local datasets are much smaller in size, with the biggest dataset containing around 1000 examples for each of the major neurotransmitters: `acetylcholine`, `serotonin`, `dopamine`, `glutamate`, `gaba`, `octopamine`, and `tyramine`.

## Overview

This project provides a complete pipeline to train deep learning models (VGG or ResNet) to classify neurotransmitter types from electron microscopy images at synaptic sites. It includes scripts for:
-   **Data Preparation**: Splitting raw data into training and testing sets.
-   **Training**: Training a model on the prepared data.
-   **Prediction**: Running inference on new or held-out data from various sources.
-   **Evaluation**: Calculating accuracy and other metrics from prediction results.

## Directory Structure

The project is organized as follows:

```bash
.
├── config/
│   ├── config.py           # Main configuration for training
│   └── config_predict.py   # Configuration for prediction
├── data_utils/
│   └── pre_process/
│       └── split_data.py   # Script to create train/test splits
├── engine/
│   ├── predict/
│   │   └── predict_3d.py   # Core prediction logic
│   └── train/
│       └── train_3d.py     # Core training logic
├── gunpowder_nodes/
|   └── mongo_source.py     # Custom Gunpowder node for MongoDB
├── models/
│   ├── resnet3d.py         # 3D ResNet model definition
│   └── vgg3d.py            # 3D VGG model definition
├── scripts/
│   ├── create_split.py     # Script to generate data splits
│   ├── predict.py          # Core prediction script
│   └── evaluate.py         # Script to evaluate predictions
├── predicter.py            # Launcher for prediction runs
├── trainer.py              # Launcher for training runs
└── readme.md

```

## Getting Started

### Prerequisites
-   Conda (for managing the Python environment)
-   Access to a raw data volume (Zarr or N5 format) / HDF5 files containing synapse locations in the CREMI format.
-   Optionally, a MongoDB instance for large-scale data sourcing synapse pre-synapse points.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda create -n synister python=3.8
    conda activate synister
    ```

3.  **Install dependencies:**
    *(Note: You will need to create a `requirements.txt` file based on your project's specific dependencies, including `gunpowder`, `torch`, `yacs`, `pandas`, `sklearn`, `tqdm`, `h5py`, and `pymongo`.)*
    ```bash
    pip install -r requirements.txt
    ```

## Step 1: Data Preparation (Optional)

If you wish to create a reproducible train/test split from your HDF5 files, you can use the provided script. This is recommended for comparing models fairly.

1.  **Run the split script:**
    The script will scan a directory of HDF5 files, shuffle them, and create a `.pkl` file containing lists of files for training and validation.

    ```bash
    python scripts/create_split.py \
        --data_dir /path/to/your/data_root \
        --split_ratio 0.9 \
        --output_file ./data_splits/my_split.pkl
    ```
    * `--data_dir`: Should point to a directory containing subfolders for each neurotransmitter class (e.g., `/path/to/data_root/gaba/`, `/path/to/data_root/acetylcholine/`, etc.).

## Step 2: Training

Training is launched via the `trainer.py` script, which uses `config/config.py` for its settings.

1.  **Configure Training:**
    Open `config/config.py` and adjust the settings. Key parameters include:
    -   `DATA.USE_SPLIT_FILE`: Set to `True` to use the `.pkl` file from Step 1, or `False` to use all data found in the `DATA.DATA_DIR_PATH`.
    -   `DATA.SPLIT_FILE`: Path to your `.pkl` split file.
    -   `DATA.HOME`, `DATA.DATA_DIR_PATH`, `DATA.BRAIN_VOL`: Paths to locate your data.
    -   `TRAIN.MODEL_TYPE`: Choose between `RESNET` or `VGG`.
    -   `LOGGING`: Set paths for logs, snapshots, and checkpoints.

2.  **Launch Training:**
    ```bash
    python trainer.py -c config/config.py
    ```

## Step 3: Prediction

Prediction is launched via `predicter.py` and configured using `config/config_predict.py`. It is highly flexible and can source synapse locations from multiple backends.

1.  **Configure Prediction:**
    Open `config_predict.py` and set the parameters for your prediction run.

    -   **Set the checkpoint:**
        ```python
        _C.PREDICT.CHECKPOINT = "/path/to/your/checkpoints/model_checkpoint_100000.pt"
        ```
    -   **Set the raw data container:**
        ```python
        _C.RAW_DATA.CONTAINER = "/path/to/raw_data.zarr"
        _C.RAW_DATA.DATASET = "volumes/raw/s0"
        ```
    -   **Choose a Data Source Method:** Set `_C.DATA_SOURCE.METHOD` to one of the following:
        -   `'pkl'`: To predict on the validation set from a split file.
            ```python
            _C.DATA_SOURCE.PKL_FILE_PATH = "./data_splits/my_split.pkl"
            ```
        -   `'directory'`: To predict on all HDF5 files in a specific directory (for unlabeled data).
            ```python
            _C.DATA_SOURCE.DIRECTORY_PATH = "/path/to/unlabeled_synapses/"
            ```
        -   `'mongo'`: To predict on synapse locations stored in a MongoDB database (for very large scale inference).
            ```python
            _C.DATA_SOURCE.MONGO.DB_NAME = "synapse_database"
            _C.DATA_SOURCE.MONGO.DB_HOST = "mongodb://user:pass@host:27017/"
            _C.DATA_SOURCE.MONGO.COLLECTION = "synapses_to_predict"
            ```

2.  **Launch Prediction:**
    Run the `predicter.py` script, pointing it to your configuration file.

    ```bash
    python predicter.py config/config_predict.py
    ```
    The predictions will be saved as a CSV file in the directory specified by `_C.PREDICT.OUTPUT_DIR`.

## Step 4: Evaluation

If your prediction run included ground truth labels (e.g., when using a `.pkl` split file), you can evaluate the model's performance.

1.  **Run the evaluation script:**
    Point the script to the CSV file generated during the prediction step.

    ```bash
    python scripts/evaluate.py predictions/predictions.csv \
        --output_file predictions/evaluation_report.txt
    ```
    This will print an evaluation report including overall accuracy, per-class metrics (precision, recall, F1-score), and a confusion matrix to the console and save it to the specified output file.

