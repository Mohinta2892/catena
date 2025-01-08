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
### Technology Pre-requisites:
- [System Requirements](https://github.com/Mohinta2892/catena/blob/dev/local_shape_descriptors/docs/source/systemrequirements.rst)
- Installation instructions: [Docker](https://github.com/Mohinta2892/catena/blob/dev/local_shape_descriptors/docker/readme.md), [MongoDB](https://www.mongodb.com/docs/manual/installation/)
- [Dataset preparation](https://github.com/Mohinta2892/catena/tree/dev/synful/pytorch/data_utils/download_data/meta_analysis)

*Note*: MongoDB is required only during inference.



### Pull the docker image:

```bash
docker pull mohinta2892/synful_tf1_py3:latest
```


### Check docker image exists
  ```
  docker images
  ```

## Train
- Run the loaded docker image:
  ```bash
  nvidia-docker run --shm-size 128gb --pids-limit -1 -it -u `id -u`:`id -g` -v `pwd`:`pwd` -w `pwd` -v {/path/to}/synful/tensorflow/:/home --network=host {nvcr.io/nvidia/tensorflow:21.12-tf1-py3}
  ```
- Change directory inside docker
```bash
cd /home/train_from_scratch/scripts
```
- Edit `parameters.json`:
```json
{
    "input_size": [128,128,128],  # A 12GB GPU should fit 256^3
    "downsample_factors": [[2, 2, 2], [2,2,2], [2,2,2]], # Adjust downsampling based on input size 
    "fmap_num": 12,
    "fmap_inc_factor": 5,
    "unet_model": "dh_unet", # Double-headed UNet will be trained.
    "learning_rate": 0.5e-4,
    "loss_comb_type": "sum",
    "m_loss_scale": 1.0,
    "d_loss_scale": 1.0, # Scale the loss if needed, unscaled works too!
    "reject_probability": 0.95, # 95% of batches with at least 1 post-syn to be processed, 5% of times an empty batch may be passed. 
    "blob_radius": 20, # Lower radius. Min should be 10
    "max_iteration": 300000, # Increase number of iterations. 1 batch is processed per iteation.
    "blob_mode": "ball", # Options - ball, sphere
    "d_scale": 1,
    "d_blob_radius": 150, # Lower radius for direction vectors scope. Min should be 100.
    "cliprange": [7e-4, 0.9993],
    "voxel_size": [8,8,8], # Edit resolution in nm
}
```

- Run `generate_network.py`
You may adjust the `test_input_size` in the script by editing the last two lines in the script.
```python
# Bigger network used for large datasets, make it as big as gpu memory allows.
parameter['input_size'] = (860, 860, 860)
mknet(parameter, name='test_net')
```
Then run:
```python
python generate_network.py
```

Assuming data is organised like (more elaborate example under [local_shape_descriptors](https://github.com/Mohinta2892/catena/tree/dev/local_shape_descriptors/data_utils/download_data)):
```bash
/home/catena/data
  - BRAIN_VOL_NAME
    - data_3d
        - train
        - test
```

- Edit `train.py` to point to data.
```python
data_dir = '/home/catena/data/{BRAIN_VOL_NAME}/data_3d/train'
data_dir_syn = data_dir
samples = [
    'train_vol1_x7827_9021_y5622_6798_z4441_5575', # do not include the ext hdf, the loader handles it
    ....
]
# calculate the roi in nm. If no offset then roi x starts at 0 and ends at |7827-9021|*8 as an example.
roi_1 = gp.Roi(np.array((0, 0, 0)), np.array((9072, 9408, 9552))) # this is in ZYX, place values carefully!
# make a list of all rois there are more than 1 input vols
rois = [roi_1, ... roi_n]
```

- Run `train.py`. It will automatically read `parameter.json`.
```python
CUDA_VISIBLE_DEVICES=0 python train.py
```

## Predict

## Visualization of results

## Evaluation


