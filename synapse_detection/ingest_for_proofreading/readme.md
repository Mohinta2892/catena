### Importing predictions to CATMAID and CAVE (FlyWIRE) for proofreading

Ultimately all predictions must be either uploaded to FlyWire (rebranded as CAVE) or to CATMAID for collaborative proofreading.

This folder contains (or points to) the scripts that allows us to upload the predictions to either platform.

There are a bunch of post-processing involved to ensure the predicted data is as clean as possible. These are under the post_scripts_for_upload.

Please follow instructions if you wish to upload synapses to CAVE or CATMAID.
#### CAVE

- Run:
```
(ngl) samia@samia-PC-XB10250:/media/samia/DATA/mounts/gpu2/synapse_detection$ python upload_synapses_cave.py --db /media/samia/DATA/mounts/gpu2/synapse_detection/predictions/octo_cns/octo_cns_setup_03_octo_cube_all3_same_preid_256_300000_filt/final_upload_ready.db \
zlatic_octo_8x8x8_full_200525_datastack --reset
```
>[!Note]
> **--reset**: will start to upload from the very first ID; 
> **--verify**: will only upload the first 10K synapses and stop 

#### CATMAID:
- Please follow the jupyter notebook [here](https://github.com/shiyanlee/synapse_CATMAID/blob/main/code%20/import_to_catmaid/push_all_synspred_pymaid.ipynb) for selective loading.
- For parallel loading of synapses, [use this concurrent upload script.](https://github.com/shiyanlee/synapse_CATMAID/blob/main/code%20/import_to_catmaid/concurrent_synapse_importer_v2.py)

>[!WARNING]
>Both version of these scripts have been tested with the db that is generated via this [script](https://github.com/Mohinta2892/catena/blob/dev/synapse_detection/synful/pytorch/data_utils/post_process/synapse_to_sqlite.py).
