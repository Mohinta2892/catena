## Downloading or making your own ground-truth

We train this implementation of Synful on public and local datasets. Our local datasets are mostly isotropic imaged using Focused Ion Beam Scanning Electron Microscope (FIBSEM). Hence, our focus mainly has been on curating isotropic public datasets.

#### Build a conda env
- Check you have conda installed in your base machine.

*Note*: [syn_environment.yml](https://github.com/Mohinta2892/catena/blob/dev/synful/pytorch/data_utils/download_data/meta_analysis/syn_environment.yml) env file is generated from a Linux Ubuntu machine. Hence, may not work seamlessly when you try to build on a different compute machine. We will release a slim pip install `requirements.txt` soon!
```bash
conda env create --name syn --file=syn_environment.yml
```

<details close>
 <summary><strong>Curating Hemi-brain for training</strong></summary>

<br>

Extensive documentation and code [here for hemi-brain](https://github.com/shiyanlee/synapse_neuprint). You should be able find a EM region of interest, filter and find all synpases of interest based on a confidence that comes with this dataset with the code.
Once you have the csv files for:
-  bounding boxes
-  synapse point coordinates for the pre and the post synapse

*Optional: Organise the csvs*
```bash
hemibrain
  cluster_files
    - syn_points_bbox1.csv
    - syn_points_bbox2.csv
  bboxes
    - bbox1.csv
    - bbox2.csv

```
- Edit and run [download_syns_from_hemibrain.py](https://github.com/Mohinta2892/catena/blob/dev/synful/pytorch/data_utils/download_data/meta_analysis/download_syns_from_hemibrain.py). This script will save EM and annotations in the CREMI format in `.HDF` files.

  >[!Important] This script does not yet support command-line argument passing. Please edit `cluster_files` to point your synapse_points.csv, `top_roi_bboxes` to point to your bbox.csv and `outputpath` to where you would like save your volumes. 

Currently the script assumes that your EM resolution is 8nm in zyx and is hardcoded as below. Edit if you are working with a downsampled hemibrain.

```python
    synapse_data = {
        'bodyId_pre': syn_inputs['bodyId_pre'],
        'bodyId_post': syn_inputs['bodyId_post'],
        'x_pre_roi': (syn_inputs['x_pre'] - x_min).astype(int),
        'x_post_roi': (syn_inputs['x_post'] - x_min).astype(int),
        'y_pre_roi': (syn_inputs['y_pre'] - y_min).astype(int),
        'y_post_roi': (syn_inputs['y_post'] - y_min).astype(int),
        'z_pre_roi': (syn_inputs['z_pre'] - z_min).astype(int),
        'z_post_roi': (syn_inputs['z_post'] - z_min).astype(int)
        'x_pre_roi': (syn_inputs['x_pre'] - x_min).astype(int) * 8, <-- Edit here
        'x_post_roi': (syn_inputs['x_post'] - x_min).astype(int) * 8,
        'y_pre_roi': (syn_inputs['y_pre'] - y_min).astype(int) * 8,
        'y_post_roi': (syn_inputs['y_post'] - y_min).astype(int) * 8,
        'z_pre_roi': (syn_inputs['z_pre'] - z_min).astype(int) * 8,
        'z_post_roi': (syn_inputs['z_post'] - z_min).astype(int) * 8
    }
```

Hemi-brain allows you to download both clahed (contrast enhanced) and unclahed raw EM. You can choose which to download by (un)commenting one or the other.
```python

dataset_em = ts.open({
    'driver': 'neuroglancer_precomputed',
    # 'kvstore': 'gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg', <--Choose clahe
    'kvstore': 'gs://neuroglancer-janelia-flyem-hemibrain/emdata/raw/jpeg',   <--Choose non-clahe 
    'context': {'cache_pool': {'total_bytes_limit': 100_000_000}},
    'recheck_cached_data': 'open',
}).result()[ts.d['channel'][0]]

```
##### RUN PYTHON SCRIPT:
```bash
conda activate syn
```
```python
python download_syns_from_hemibrain.py
```

</details>

<details close>
 <summary><strong>Curating Local data in Catmaid</strong></summary>

<br>

Extensive documentation and code [here for Catmaid-based volumes](https://github.com/shiyanlee/synapse_CATMAID). The code overall follows the same strategy in getting the synapse locations from catmaid. However, this code would require you to have preset CNS from where you want to pull the synapses.
Once you have the csv files for:
-  bounding boxes
-  synapse point coordinates for the pre and the post synapse


*Optional: Organise the csvs*
```bash
localvol1
  cluster_files
    - syn_points_bbox1.csv
    - syn_points_bbox2.csv
  bboxes
    - bbox1.csv
    - bbox2.csv

```
- Edit and run [download_local_synapses.py](https://github.com/Mohinta2892/catena/blob/dev/synful/pytorch/data_utils/download_data/meta_analysis/download_local_synapses.py). This script will save EM and annotations in the CREMI format in `.HDF` files.

  >[!Important] This script does not yet support command-line argument passing. Please edit `cluster_files` to point your synapse_points.csv, `top_roi_bboxes` to point to your bbox.csv and `outputpath` to where you would like save your volumes. 

Currently the script assumes that your EM resolution is 8nm in zyx and is hardcoded as below. Edit if you are working with a downsampled hemibrain.
It also assumes that the volumes are stored on disk in the ZYX orientation. So check below if you have to adjust the orientation.

```python
      synapse_data = {
        'bodyId_pre': syn_inputs['pre_neuron'],
        'bodyId_post': syn_inputs['post_neuron'],
        'connectorId': syn_inputs['connector_id'],
        # because x-axis is reversed, we have to reverse the coords of synapses for correct visualisation
        # but x != z in the coord space. Keep the axis as they are in the csv
        # 'x_pre_roi': (x_max - (syn_inputs['pre_x'] )).astype(int),
        'x_pre_roi': ((x_max - (syn_inputs['connector_x'] // 8)).astype(int)) * 8, <--Edit resolution and orientation here
        'x_post_roi': ((x_max - (syn_inputs['post_x'] // 8)).astype(int)) * 8,
        # 'y_pre_roi': (syn_inputs['pre_y'] - y_min).astype(int),
        'y_pre_roi': ((syn_inputs['connector_y'] // 8 - y_min).astype(int)) * 8,
        'y_post_roi': ((syn_inputs['post_y'] // 8 - y_min).astype(int)) * 8,
        # 'z_pre_roi': (syn_inputs['pre_z']  - z_min).astype(int),
        'z_pre_roi': ((syn_inputs['connector_z'] // 8 - z_min).astype(int)) * 8,
        'z_post_roi': ((syn_inputs['post_z'] // 8 - z_min).astype(int)) * 8,
        'x_connector': ((x_max - (syn_inputs['connector_x'] // 8)).astype(int)) * 8,
        'y_connector': ((syn_inputs['connector_y'] // 8 - y_min).astype(int)) * 8,
        'z_connector': ((syn_inputs['connector_z'] // 8 - z_min).astype(int)) * 8,
    }
```
*Note*: The EM downloaded will have the same contrast level as seen in your CATMAID volume. If you wish to enhance the contrast you can run [clahe_gconn.py](https://github.com/Mohinta2892/catena/blob/dev/local_shape_descriptors/data_utils/preprocess_volumes/clahe_gconn.py).


</details>
