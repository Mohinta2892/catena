# Welcome to Pytorch Synful: Automatic detection of synaptic partners in a whole-brain Drosophila electron microscopy dataset

## What is Synful?
Synful leverages supervised machine learning to automate the detection of synaptic partners from volume Electron Microscopy (EM) datasets of insect brains. It is U-Net based network that learns to predict post-synaptic masks and pre-synaptic direction vectors either simultaneously or independently. The predicted synatic partnerships facilitate extraction neural connectivity maps at scale. Synful has been tested on adult fly EM. Here, we refactor the original codebase from a) TensorFlow 1.x to Pytorch b) test performance reproducibility on public datasets (e.g., CREMI and FAFB) and c) apply to new local larval EM data.

- Read the paper here: [Buhmann et al., Nature Methods, 2021 ](https://www.nature.com/articles/s41592-021-01183-7)
- Access the original codebase [here](https://github.com/funkelab/synful)
- Additional tools from Funkelab for Synful: [SynfulCircuit - A neural circuit querying engine](https://github.com/funkelab/synfulcircuit), [Synful_FAFB - Entry point for particularly querying the FAFB dataset](https://github.com/funkelab/synful_fafb)

> [!NOTE]
> Given the sparsity of synapses in manually annotated EM regions of interest (ROIs), these models train need long training times (e.g., 400000 t0 1M epochs)

## Getting Started
Read these:
- [System Requirements](https://github.com/Mohinta2892/catena/blob/dev/local_shape_descriptors/docs/source/systemrequirements.rst)
- Installation instructions: Docker, Conda
- Dataset preparation

## Usage instructions
Synful operates in broadly 2 modes: Single task mode, Multi-task Mode.
In the Single task mode, you train the network to either to predict the post-synaptic masks (semantic segmentation) or pre-synaptic direction vectors.
In the Multi-task mode, you train the network to simultaneously learn both objectives. Additionally, Multi-task mode can be implemented with either double-head U-Net architecture or a double-pathway (separate decoder/upsampling path) to correspond to each task.

<details close>
 <summary>Understand and modify as needed the <a href="config/config.py">config.py</a></summary>

<br>

<strong> For training models </strong> <br>
`config.py` contains `SYSTEM`, `DATA`, `PREPROCESS`, `TRAIN`, `MODEL_ISO` (for isotropic datesets) and `MODEL_ANISO` (for anisotropic datasets).
Most of these configurations and hyper-parameters have been populated with default used during experiments.
You may want to modify them to suit your needs. Please look at the commented text adjacent to the hyper-params set to get an idea of what they are.

Separate `config.py` files for public datasets like CREMI, SNEMI, ZEBRAFINCH are provided.

<strong> For running inference with trained models </strong> <br>
`config_predict.py` should be used to run affinity prediction. All configurations set in the file should be automatically picked up by `predicter.py` or `super_predicter_daisy.py`.
Ensure you set the same architectural hyper-parameters under `MODEL_ISO` OR `MODEL_ANISO` for pytorch to load the weights correctly.
Also, ensure you put the data in the correct path inside a `test` folder, and pass the correct `model checkpoint`.

</details>


