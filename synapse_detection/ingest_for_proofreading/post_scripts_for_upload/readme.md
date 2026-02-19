# Post processing scripts to filter the DB containing synapses prior to uploading in CAVE or CATMAID

- In `filter_autotapse_n_neuropil_mask_synapse_predictions.py`, the configuration is not `argparse` driven. Please edit these information prior to running.
  ```python
  # ================= CONFIGURATION =================
  SOURCE_DB_PATH = "/mnt/graid/synapse_detection/predictions/octo_cns/octo_cns_setup_03_octo_cube_all3_same_preid_256_300000/synapse_predictions.db"
  OUTPUT_DB_PATH = "/mnt/graid/synapse_detection/predictions/octo_cns/octo_cns_setup_03_octo_cube_all3_same_preid_256_300000_filt/final_upload_ready.db"
  
  # SEGMENTATION & MASK
  SEG_VOL_PATH = 'file:///mnt/graid/biomedparse/octo/seg_241224_250131b_rsg8_spl'
  NEUROPIL_TIF_PATH = '/mnt/graid/synapse_neuron_associations/octo_s6_neurophil_corrected_dilated.tif' 
  
  # RESOLUTION & SCALING
  RESOLUTION_NM = np.array([8, 8, 8]) 
  MASK_SCALE_FACTOR = 64
  MIP_LEVEL = 0
  FORCE_ZYX_TO_XYZ = True # THIS is when the mask tiff is ZYX orientation.
  
  # WORKER SETTINGS
  NUM_WORKERS = 16 
  BATCH_SIZE = 100 
```

