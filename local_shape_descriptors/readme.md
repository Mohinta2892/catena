# Welcome to the PyTorch Local Shape Descriptors

## What are Local Shape Descriptors?

Local Shape Descriptors (LSDs) introduce an auxiliary learning task aimed at improving neuron segmentation within electron microscopy volumes. These descriptors are employed alongside conventional voxel-wise direct neighbor affinities to enhance neuron boundary detection. By capturing key local statistics of neurons, such as diameter, elongation, and direction, LSDs significantly refine segmentation accuracy. Comparative studies across a variety of specimens, imaging techniques, and resolutions reveal that incorporating LSDs consistently elevates the performance of affinity-based segmentation methods. This approach not only matches the effectiveness of current state-of-the-art neuron segmentation techniques but also offers a leap in computational efficiency, making it indispensable for processing the extensive datasets expected in future connectomics research.

- **Read the paper here: [Sheridan et al., Nature Methods, 2022](https://www.nature.com/articles/s41592-022-01711-z)**
- **Read the blogpost [here](https://localshapedescriptors.github.io/)**

> [!Note]
> These are supervised ML models, hence you need ground truth. Primary tests reveal 40 microns of densely segmented volumes is good to begin with.
 
## Getting started

Read these:

- [System Requirements](docs/source/systemrequirements.rst)
- Installation instructions: [Docker](docker/install_docker.md), [Conda](conda_env/install_conda.md)
- [Dataset preparation](data_utils/download_data)

### Usage instructions

#### Semantic Segmentation to get the affinity maps

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

<details close>
 <summary> Train models with <a href="trainer.py">trainer.py</a></summary>
<br>
<strong> For training models </strong> <br>

Set the hyper-params in the `config.py` file and then run:

```
python trainer.py -c config_cremi.py
```

Note: When a config file is not passed, the default is `config.py`.

</details>

<details close>
 <summary> Run affinity predictions as a single process with <a href="predicter.py">predicter.py</a></summary>

<br>

You can place as many datasets in the `test` folder of your `BRAIN_VOLUME` as you want. Each will be processed but sequentially.

Download **pretrained** models from [here](https://www.dropbox.com/scl/fo/uxmoj3v6i8mos6lwjjvio/h?rlkey=w10iia8rd8alkx3i67u88w0er&dl=0). These models have mostly been trained with default architectural params. We will share more details sooner.

Please modify `config_predict.py` to match your `config.py` used during training. Check **above** for details.

<strong> Run prediction </strong> <br>

```
python predicter.py
```

Note: `predicter.py` does not accept a `config.py` args yet! Hence, all changes must be made in `config_predict.py` as this is default.

</details>

<details>
<summary> Run affinity predictions blockwise multiprocessing with <a href="super_predicter_daisy.py">super_predicter_daisy.py</a></summary>

> **WARNING** <br>
> THIS HAS ONLY BEEN TESTED WITH 3D VOLUMES.

You can place as many datasets in the `test` folder of your `BRAIN_VOLUME` as you want. Each will be processed but sequentially BUT WILL USE MULTIPLE-WORKERS, which makes the predictions faster.

Download **pretrained** models from [here](https://www.dropbox.com/scl/fo/uxmoj3v6i8mos6lwjjvio/h?rlkey=w10iia8rd8alkx3i67u88w0er&dl=0). These models have mostly been trained with default architectural params. We will share more details sooner.

Please modify `config_predict.py` to match your `config.py` used during training. Check **above** for details.

<strong> Run prediction parallely with Daisy task scheduling </strong> <br>

```
python super_predicter_daisy.py
```

Note: `super_predicter_daisy.py` does not accept a `config.py` args yet! Hence, all changes must be made in `config_predict.py` as this is default.

</details>

#### Instance Segmentation from predicted affinities
>[!IMPORTANT]
> Output segmentations are saved in the same output zarr under `lsd_outputs` containing `volumes/pred_affs`.
> Agglomeration thresholds are appended to dataset names: `volumes/segmentation_055`


<details>
<summary> Extract supervoxels and agglomerate for small ROIs with <a href="instance_segmenter.py">instance_segmenter.py</a></summary>

> **WARNING** <br>
> This script should be used with volumes that fit into memory. Predicted affinities are cast as before watershedding float32, so you should have enough RAM.

You must keep the output affinities under `lsd_outputs` for `instance_segmenter.py` to pick them up.
Edit data paths in `config_predict.py`. Watershed and agglomeration will be run sequentially on all output *zarr* files that contain `volumes/pred_affs`.
<br>


<strong> Run watershed and agglomeration </strong> <br>

```
python instance_segmenter.py
```

</details>

<details>
<summary> Extract supervoxels chunk-wise from large volumes with <a href="02_extract_fragments_blockwise.py">02_extract_fragments_blockwise.py</a></summary>

> **IMPORTANT** <br>
> Install [MongoDB](https://www.mongodb.com/docs/manual/installation/) before you begin. <br>
> Ensure you have `pymongo~=4.3.3` and `daisy~=1.0`

> **WARNING** <br>
> `db_host = "localhost:27017"` and `db_name = "lsd_parallel_fragments"` are hardcoded as these in the script. Yet to be supported via `config_predict.py`. `collection_name` would be auto set to the name of your zarr file.

<strong> Run watershed with daisy chunk-wise</strong> <br>

```
python 02_extract_fragments_blockwise.py
```
**NB: 02_extract_fragments_blockwise.py calls [02_extract_fragments_worker.py](engine/post/02_extract_fragments_worker.py)**

</details>

<details>
<summary> Agglomerate supervoxels of large volumes chunk-wise with <a href="03_agglomerate_blockwise.py">03_agglomerate_blockwise.py</a></summary>

> **WARNING** <br>
> This cannot be run if `02_extract_fragments_blockwise.py` has not been run.

<strong> Run agglomeration with daisy chunk-wise</strong> <br>

```
python 03_agglomerate_blockwise.py
```
**NB: 03_agglomerate_blockwise.py calls [03_agglomerate_worker.py](engine/post/03_agglomerate_worker.py)**
</details>

##### Final steps to extract final segmentation for LARGE volumes

<details>
<summary> Finding all segments and saving them as Look-Up-Tables (LUTs) <a href="04_find_segments_full.py">04_find_segments_full.py</a></summary>

> **WARNING** <br>
> This cannot be run if `03_agglomerate_blockwise.py` has not been run. <br>
> **Don't forget to pass `daisy_logs/config_0.yml` from your daisy_logs folder auto-created under `catena/local_shape_descriptors`.** <br>
> Output LUTs are saved under `lsd_outputs`

<strong> Create a LUT file </strong> <br>

```
python 04_find_segments_full.py daisy_logs/config_0.yml
```
</details>

<details>
<summary> Extracting a final segmentation <a href="05_extract_segmentation_from_lut.py">05_extract_segmentation_from_lut.py</a></summary>

> **WARNING** <br>
> This cannot be run if `04_find_segments_full.py` has not been run. <br>
> **Don't forget to pass `daisy_logs/config_0.yml` from your daisy_logs folder auto-created under `catena/local_shape_descriptors`.** <br>
> Final segmentations are saved in the zarr under `lsd_outputs`.

<strong> Extract Segments from LUT </strong> <br>

```
python 05_extract_segmentation_from_lut.py daisy_logs/config_0.yml
```
</details>

## Where does Local Shape Descriptors perform well and where does it fail?
