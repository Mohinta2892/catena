# Welcome to the PyTorch Local Shape Descriptors

## What are Local Shape Descriptors?

Local Shape Descriptors (LSDs) introduce an auxiliary learning task aimed at improving neuron segmentation within electron microscopy volumes. These descriptors are employed alongside conventional voxel-wise direct neighbor affinities to enhance neuron boundary detection. By capturing key local statistics of neurons, such as diameter, elongation, and direction, LSDs significantly refine segmentation accuracy. Comparative studies across a variety of specimens, imaging techniques, and resolutions reveal that incorporating LSDs consistently elevates the performance of affinity-based segmentation methods. This approach not only matches the effectiveness of current state-of-the-art neuron segmentation techniques but also offers a leap in computational efficiency, making it indispensable for processing the extensive datasets expected in future connectomics research.

- **Read the paper here: [Sheridan et al., Nature Methods, 2022](https://www.nature.com/articles/s41592-022-01711-z)**
- **Read the blogpost [here](https://localshapedescriptors.github.io/)**

> [!Note]
> These are supervised ML models, hence you need ground truth. Primary tests reveal: 40 microns of densely segmented volumes is good to begin with.
 
## Getting started

Read these:

Installation instructions
Dataset preparation
### Usage instructions

<details open>
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



<details close>
 <summary> Train models with <a href="trainer.py">trainer.py</a></summary>
</details>

<details close>
 <summary> Run affinity predictions as a single process with <a href="predicter.py">predicter.py</a></summary>
</details>

<details>
<summary> Run affinity predictions blockwise multiprocessing with <a href="super_predicter_daisy.py">super_predicter_daisy.py</a></summary>
</details>

## Where does Local Shape Descriptors perform well and where does it not perform?
