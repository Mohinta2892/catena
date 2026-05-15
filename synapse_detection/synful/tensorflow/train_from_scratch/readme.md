# Welcome to TensorFlow Synful
## What is TensorFlow Synful?
Synful is the code implementation of Julia Buhmann's
` Automatic detection of synaptic partners in a whole-brain Drosophila electron microscopy data set `
paper that automatically detects synaptic partners from volumetric Electron Microscopy (EM) datasets using a UNet-based machine learning models following a single-task or mult-task training paradigm.
To that end, U-Net based network learns to predict post-synaptic masks and pre-synaptic direction vectors either simultaneously or independently.
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
  nvidia-docker run --shm-size 128gb --pids-limit -1 -it -u `id -u`:`id -g` -v `pwd`:`pwd` -w `pwd` -v {/path/to}/synful/tensorflow/:/home --network=host mohinta2892/synful_tf1_py3:latest
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
If 'funlib' package missing errors are thrown, check [#ISSUE44](https://github.com/Mohinta2892/catena/issues/44) to troubleshoot.

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

In the latest scripts under the [example experiments ](https://github.com/Mohinta2892/catena/tree/dev/synapse_detection/synful/tensorflow/train_from_scratch/scripts/experiment_examples/exp_tem_cross4), you **NO LONGER** need to set `roi_1 = gp.Roi(np.array((0, 0, 0)), np.array((9072, 9408, 9552)))`. The script handles it automatically. This main branch of training script will be updated v soon.

- Run `train.py`. It will automatically read `parameter.json`.
```python
CUDA_VISIBLE_DEVICES=0 python train.py
```

## Predict

>[!IMPORTANT]
>Inference must be called from a folder named `predict`. Especially when calling from within a docker.
> You must copy the checkpoints into this predict directory alongside their `.meta` and `train_net.json` and `test_net.json` files. See example below.

- Typical contents of a predict folder:
  ```bash
    04_predict_extract_blockwise.py
    extract_cremi.json
    extract_parameters_setup32_score10.json
    extract_parameters_setup32_score100.json
    extract_parameters_setup32_score100_nms.json
    output_predict_on_train
    predict_and_extract.py
    predict_blockwise.py
    predict_extract_parameters.json
    predict_extract_parameters_manc.json
    predict_extract_parameters_manc_score100.json
    predict_extract_parameters_octo_cube1.json
    predict_extract_parameters_octo_cube1_score100.json
    predict_extract_parameters_octo_cube2.json
    predict_extract_parameters_octo_cube2_score100.json
    predict_extract_parameters_octo_cube2_score557.json
    predict_extract_parameters_octo_cube3.json
    predict_extract_parameters_octo_cube3_score100 .json
    predict_extract_parameters_octo_train.json
    predict_extract_parameters_score100.json
    predict_extract_parameters_wasp.json
    predict_extract_parameters_wasp_score100.json
  
    test_net.meta # These must be copied from the train folder to the predict folder
    test_net_config.json
    train_net.meta
    train_net_checkpoint_300000.data-00000-of-00001
    train_net_checkpoint_300000.index
    train_net_checkpoint_300000.meta
    train_net_config.json
  ```

- Go into the predict folder. Please edit the below parameters in `predict_extract_parameters_*.json`.
  ```json
    {
    "experiment": "nips", # This can be anything. Choose something that makes sense to you for your set of experiments.
    "setup": "setup_03_neurips_wasp_preid_256", # Change this to your train setup's folder name.
    "iteration": 300000,  # Choose a checkpoint number.
    "raw_file" : "/zstore/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/test/wasp_train_vol0_syns_zyx_2217-2617_4038-4448_6335-6735_cremi_same_preid.hdf", # Provide the path to your RAW EM file
    "raw_dataset" : "volumes/raw", # The dataset in the .hdf file that contains the RAW EM
    "num_workers": 1, # If running from within a docker, leave this as 1. 
    "db_host": "localhost:27017", # Your mongo DB host. In general, if mongo is not being served via a network, MongoDB is served in localhost:27017
    "db_name": "wasp_score100", # Give a DB name, you should give different names if you run multiple inferences on the same dataset. Else Daisy will say that the DB is already exists.
    "out_basedir": "output_predict_on_train/", # Choose a folder name where your outputs will be saved.
    "overwrite":false, # Leave this as false if you do not want to overwrite the DB and predictions
    "configname": "train", # We use the train config which uses a 256^3 sliding window on the RAW data. You should use `test` to run prediction on very large datasets on bigger GPUs. OPTIONS: train, test
    "extraction_parameters": "./extract_parameters_setup32_score100.json", # Leave this as it is if you make no changes to the extract parameters.
    "synapse_context": [0,0,0], # Increase this to use context like [36,36,36] if you have a larger dataset.
    "mask" : "", # You can provide a mask to skip regions in the input data. Useful when skipping resin can speed up inference during whole brain inference 
    "mask_ds" : "", # Dataset which contains the ds in the mask
    "max_retries": 3 # It will retty to make an entry to MongoDB at least 3 times
    }
  ```
- Make edits to the `extract_parameters.json`. This is optional.
  ```json
  {
    "extract_type": "cc", # Uses connected components
    "cc_threshold": 0.95, # Uses a CC threshold of 0.95 to find the post-synaptic masks. Anything above this threshold becomes a detected post-site mask
    "loc_type": "edt", # Distance transform
    "score_thr": 100, # Use a score value to join pre-to-post. All sites that are >= this score will be saved as synapses.
    "score_type": "sum", # Scores are summed up to find the post sites. For more details you should see the paper's Connection Prediction section.
    "nms_radius": null # You can choose to pass a radius to suppress False positives. It should be large enough to make a difference. Make a calculated guess based on your training parameters.
  }
  ```
- Run inference to save only the synapses
  ```bash
  python 04_predict_extract_blockwise.py predict_extract_parameters.json
  ```
- To save the predictions of the post masks and pre direction vectors from the post-sites, run:
  ```bash
    python predict_blockwise.py predict_extract_parameters.json
  ```

>[!NOTE]
>You can use [run_predict_jobs](https://github.com/Mohinta2892/catena/blob/dev/synful/tensorflow/train_from_scratch/scripts/predict/run_predict_jobs.sh) to run prediction on multiple datasets and using multiple `parameters.json` files.
>It is bash script which will call the model inference on the specified json files sequentially. Please edit the GPU parameter. Currently set to 3.

## Visualization of results

- To visualize the predicted synapses, you should use [visualize_synful_inference](https://github.com/Mohinta2892/catena/blob/dev/visualize/visualize_synful_inference.py).
Please edit the following paths to point to your data:
```python
    # trainingfile = '/groups/flyem/home/huangg/cln/exp/cx_smallcubes/synful/5_bf350.h5'
    trainingfile = '/media/samia/DATA/mounts/zstore1/catena/data/preprocessed_3d/PARKER_s_vsize_8_8_8/parker_cube1_16852_17620_y7286_8054_z1506_2274_clahed.hdf'
    # neuron_ds = '/volumes/labels/neuron_ids' # optional
    # mask_ds = 'volumes/masks/groundtruth' #optional
    raw_ds = 'volumes/raw'
    # neuron = daisy.open_ds(trainingfile, neuron_ds) # optional
    # mask = daisy.open_ds(trainingfile, mask_ds)  # optional
    raw = open_ds(trainingfile, raw_ds)

    # inferencefile = '/media/samia/DATA/mounts/zstore1/synful/scripts/predict_dec_cube2/output_predict_on_train/octo/setup_03_octo_cube2/300000/octo_cube2_12485_13164_y6231_6901_z3971_4640.zarr'
    # pred_post_syn = 'volumes/pred_syn_indicator'
    # pred_post_dir = 'volumes/pred_partner_vectors'
    # pred_post_syn = open_ds(inferencefile, pred_post_syn)
    # pred_post_dir = open_ds(inferencefile, pred_post_dir)

    # Path to your synapses output directory, should be within the  `predict` folder
    synapsedir = '/media/samia/DATA/mounts/zstore1/synful/scripts/predict/output_predict_on_train/parker_cube1_8_setup_03_octo_cube_all3_same_preid_256_300000/syn_cc_thr095000_sum'
    gt_synfile = trainingfile # this is the file that contains raw or raw + gt
```

This visualize script is actively changes, hence slightly unclean. We will release a cleaner version soon.

## Evaluation

- To run eval you will need to save the predictions into 3 csvs, namely, pre-site locations csv, post-site locations csv and a pre-post mapping csv. Both GT and the predictions should follow the same format. Please check sample files shared to get an idea to check what they look like.
  We can generate the csvs by running `find_matches_n_vizualize.py`. Please edit the datapaths in the file, they follow the same structure as the aforementioned visualize script.

  ```bash
      python https://github.com/Mohinta2892/catena/blob/dev/synful/eval/predictions/find_matches_n_vizualize.py
  ```

- We repurpose Synful's original evaluation script, which is based on the CREMI eval standards. If you have held-out test sets, you run eval like:
  
```bash
python https://github.com/Mohinta2892/catena/blob/dev/synful/eval/predictions/synapse_partners_pairwise.py \
--gt-pre /media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/test/octo_cube1_8083_8765_y5878_6542_z4697_5319_gt_pre_locations.csv \
--gt-post /media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/test/octo_cube1_8083_8765_y5878_6542_z4697_5319_gt_post_locations.csv \
--pred-pre /media/samia/DATA/mounts/zstore1/synful/scripts/predict/output_predict_on_train/octo_cube1_sc100_setup_03_neurips_octo_labels_300000/pred_pre_locations_score770.csv \ 
--pred-post /media/samia/DATA/mounts/zstore1/synful/scripts/predict/output_predict_on_train/octo_cube1_sc100_setup_03_neurips_octo_labels_300000/pred_post_locations_score770.csv \
--pred-mapping-csv /media/samia/DATA/mounts/zstore1/synful/scripts/predict/output_predict_on_train/octo_cube1_sc100_setup_03_neurips_octo_labels_300000/pre_post_mapping_score770.csv \
--resolution-x 1 \ # change the resolution if your outputs in pixel space and will need to be converted nm space
--resolution-y 1 \ 
--resolution-z 1 \
--output-dir /media/samia/DATA/mounts/zstore1/synful/scripts/predict/output_predict_on_train/octo_cube1_sc100_setup_03_neurips_octo_labels_300000 \
--matching-threshold 550 # this is default and is in nm
```

This script will save F1-scores, Precision and Recall for every pair of synapse. The `matching-threshold` is an important parameter that tells you how far to look away from GT locations to find a corresponding predicted match.

- To calculate F1,Precision and Recall only at pre or post site predictions, you can run [calculate_detection_metrics_samia](https://github.com/Mohinta2892/catena/blob/dev/synful/eval/predictions/calculate_detection_metrics_samia.py):
  ```bash
  python python -u /data/dfranco/datasets/synapses/scripts/calculate_detection_metrics_samia.py \
  --input_pred_dir /data/dfranco/datasets/synapses/samia_results/new_run_256_patch_samia \
  --input_gt_dir /data/dfranco/datasets/synapses/OCTO/test/raw.original \ 
  --output_dir "/data/dfranco/datasets/synapses/samia_results/OUT/new_run_256_patch_samia" \
  --BiaPy_dir /data/dfranco/BiaPy \
  --tolerance 120 # this is default and is in pixels 
  ```

  
