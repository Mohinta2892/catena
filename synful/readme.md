# Synful PyTorch and TensorFlow Usage

Synful implements the partner-detection approach from Julia Buhmann et al.’s paper, `Automatic detection of synaptic partners in a whole-brain Drosophila electron microscopy dataset`. It uses a U-Net–style encoder–decoder to jointly (1) localize post-synaptic sites (potentially annotations are on top of PSDs) and (2) predict a direction vector field from each post-synaptic site toward its pre-synaptic partner. This multi-task setup lets the network learn where synapses are and who they connect to—within a single model.

- Read the paper here: [Buhmann et al., Nature Methods, 2021](https://tinyurl.com/ywahwmwj)

### Synful's Architecture at a glance (multi-task U-Net)

<br>
<div>
<p align="center">
<img src='https://github.com/Mohinta2892/catena/blob/dev/synful/assets/sup_fig_synful-1.png' align="center" width=800px>
</p>
</div>

We have refactored Synful's TensorFlow code and re-implemented it in PyTorch for more flexibility on newer CUDA machines and for eager execution of model operators while training.

## TensorFlow 
TensorFlow scripts are subdivided into running only inference with Buhmann et. al's pretrained networks, which were released [here](https://github.com/funkelab/synful) and also training Synful models in `TensorFlow from Scratch` on your own datasets.

- Please go [here](https://github.com/Mohinta2892/catena/tree/dev/synful/tensorflow/pretrained/train) to run inference with `pretrained` models on your anisotropic datasets.
> [!WARNING]
> You can infer on `isotropic` datasets with these models, but beware that they are quite sensitive to the resolution and quality of the EM.
> Even with test data with matching resolutions with CREMI, the models may not be as accurate as expected.
> We will reveal our findings soon!

- Please go [here](https://github.com/Mohinta2892/catena/tree/dev/synful/tensorflow/train_from_scratch) to train the models in from scratch on your own datasets.
> [!Note]
> Input is expected in `.hdf` in the [CREMI](https://cremi.org/data/) format. If you do not know how to convert to CREMI format, please follow the [example script](https://github.com/Mohinta2892/catena/blob/dev/synful/pytorch/data_utils/download_data/meta_analysis/scripts/convert_wasp_to_CREMI.py). 
> Synful can run with and without neuron segmentation.

## PyTorch
The pytorch re-implementation can be found [here](https://github.com/Mohinta2892/catena/tree/dev/synful/pytorch).

**Please note that this sub-folder is under active development.**

Currently we are testing the reimplemented models on both isotropic FIBSEM and anisotropic TEM datasets.
We will soon release complete usage instructions.



