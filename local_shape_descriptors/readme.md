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
