# Welcome to TensorFlow Synful
## What is TensorFlow Synful?
Synful is the code implementation of Julia Buhmann's
` Automatic detection of synaptic partners in a whole-brain Drosophila electron microscopy data set `
paper that automatically detects synaptic partners from volumetric Electron Microscopy (EM) datasets using a UNet-based machine learning models following a single-task or mult-task training paradigm.
To that end, U-Net based network that learns to predict post-synaptic masks and pre-synaptic direction vectors either simultaneously or independently.
The predicted synaptic partnerships facilitate extraction neural connectivity maps at scale. Synful originally was tested on adult fly EM.
We extend that capability to larval *Drosophila* both isotropic (e.g., `8nm^3`)  and anisotropic (e.g., `4x4x50 nm` in xyz ) EM data.
We refactor this codebase only to the extent of enabling training and inference with it again within a docker environment.

**Please cite the authors and Funke lab if you happen you use this reimplementation of  Synful.**
- Read the paper here: [Buhmann et al., Nature Methods, 2021 ](https://www.nature.com/articles/s41592-021-01183-7)
- Access the original codebase [here](https://github.com/funkelab/synful)
- Additional tools from Funkelab for Synful: [SynfulCircuit - A neural circuit querying engine](https://github.com/funkelab/synfulcircuit), [Synful_FAFB - Entry point for particularly querying the FAFB dataset](https://github.com/funkelab/synful_fafb)

## Getting Started
Technology Pre-requisites:
- [System Requirements](https://github.com/Mohinta2892/catena/blob/dev/local_shape_descriptors/docs/source/systemrequirements.rst)
- Installation instructions: [Docker](https://github.com/Mohinta2892/catena/blob/dev/local_shape_descriptors/docker/readme.md)
- Dataset preparation




