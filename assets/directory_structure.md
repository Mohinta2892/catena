# Catena's current directory structure
Code corresponds to about ~2M tokens in its present state.
```
Directory structure:
└── mohinta2892-catena/
    ├── README.md
    ├── CITATION.cff
    ├── LICENSE
    ├── assets/
    │   └── readme.md
    ├── em_mask_generation/
    │   ├── readme.md
    │   ├── conventional_cv/
    │   │   ├── readme.md
    │   │   ├── DOG_thresholding.ipynb
    │   │   ├── shanon_entropy_parallel.py
    │   │   └── example_outputs/
    │   ├── data_download_preprocess/
    │   │   └── convert_cloudvol_2_tif.py
    │   ├── ml_cv/
    │   │   ├── readme.md
    │   │   ├── em_mask_generation.ipynb
    │   │   ├── syn_environment.yaml
    │   │   └── assets/
    │   ├── post_process/
    │   │   └── post_process_mask.py
    │   └── utils/
    │       └── em_mask_gif_creator.py
    ├── local_shape_descriptors/
    │   ├── readme.md
    │   ├── 02_extract_fragments_blockwise.py
    │   ├── 03_agglomerate_blockwise.py
    │   ├── __init__.py
    │   ├── instance_segmenter.py
    │   ├── predicter.py
    │   ├── preprocess_data.py
    │   ├── super_predicter.py
    │   ├── super_predicter_daisy.py
    │   ├── super_predicter_daisy_chunkskipping.py
    │   ├── trainer.py
    │   ├── add_ons/
    │   │   ├── __init__.py
    │   │   ├── __pycache__/
    │   │   ├── funlib_persistence/
    │   │   │   ├── __init__.py
    │   │   │   ├── persistence_utils.py
    │   │   │   └── __pycache__/
    │   │   └── gp/
    │   │       ├── __init__.py
    │   │       ├── batch_zarr_write.py
    │   │       ├── gp_utils.py
    │   │       ├── print_profiling_stats.py
    │   │       ├── reject_if_empty.py
    │   │       └── __pycache__/
    │   ├── analysis/
    │   │   ├── readme.md
    │   │   ├── 06_evaluate_volumes.py
    │   │   ├── compare_segmentations.py
    │   │   ├── eval_predictions.py
    │   │   ├── napari_environment.yml
    │   │   └── popeye_analyse_preds.py
    │   ├── assets/
    │   │   └── readme.md
    │   ├── conda_env/
    │   │   ├── readme.md
    │   │   ├── complete_py310_environment.yml
    │   │   └── complete_py38_environment.yml
    │   ├── config/
    │   │   ├── __init__.py
    │   │   ├── config.py
    │   │   ├── config_cremi.py
    │   │   ├── config_predict.py
    │   │   ├── config_snemi.py
    │   │   ├── config_utils.py
    │   │   ├── config_zebra.py
    │   │   ├── generate_config_predict.py
    │   │   └── predict_options.py
    │   ├── data_utils/
    │   │   ├── download_data/
    │   │   │   ├── readme.md
    │   │   │   ├── __init__.py
    │   │   │   ├── batch_task_daisy.py
    │   │   │   ├── create_dir_organisation.py
    │   │   │   ├── datasets.json
    │   │   │   ├── download_gcloud_datasets.py
    │   │   │   ├── download_volumes.py
    │   │   │   ├── hdf_to_zarr.py
    │   │   │   ├── hdf_to_zarr_daisy.py
    │   │   │   ├── read_sharded_data.py
    │   │   │   └── read_sharded_data_parallel.py
    │   │   ├── helpers/
    │   │   │   ├── readme.md
    │   │   │   ├── find_maximum_input_size_at_inference.py
    │   │   │   ├── restore_mongodb_from_logs.py
    │   │   │   └── upload_skels_catmaid.py
    │   │   ├── post_process/
    │   │   │   └── binarise_aff_maps.py
    │   │   └── preprocess_volumes/
    │   │       ├── __init__.py
    │   │       ├── clahe_gconn.py
    │   │       ├── create_labels_mask.py
    │   │       ├── crop_gt.py
    │   │       ├── find_cuda_devices.py
    │   │       ├── histogram_match.py
    │   │       ├── iso_voxel.py
    │   │       ├── make_gt_from_seg2link_vols.py
    │   │       ├── merge_multiclass_labels.py
    │   │       ├── model_utils.py
    │   │       ├── pad_input.py
    │   │       ├── preprocess_data.py
    │   │       ├── resample.py
    │   │       ├── utils.py
    │   │       └── __pycache__/
    │   ├── docker/
    │   │   ├── readme.md
    │   │   ├── Dockerfile
    │   │   ├── lsd_environment.yml
    │   │   └── requirements.txt
    │   ├── docs/
    │   │   ├── make.bat
    │   │   ├── Makefile
    │   │   ├── .DS_Store
    │   │   ├── .nojekyll
    │   │   ├── build/
    │   │   │   ├── .DS_Store
    │   │   │   ├── doctrees/
    │   │   │   │   ├── environment.pickle
    │   │   │   │   ├── index.doctree
    │   │   │   │   ├── systemrequirements.doctree
    │   │   │   │   └── neuron_segmentation/
    │   │   │   │       ├── design_notes.doctree
    │   │   │   │       ├── index.doctree
    │   │   │   │       ├── install.doctree
    │   │   │   │       └── usage.doctree
    │   │   │   └── html/
    │   │   │       ├── genindex.html
    │   │   │       ├── index.html
    │   │   │       ├── objects.inv
    │   │   │       ├── search.html
    │   │   │       ├── searchindex.js
    │   │   │       ├── systemrequirements.html
    │   │   │       ├── .buildinfo
    │   │   │       ├── .DS_Store
    │   │   │       ├── _sources/
    │   │   │       │   ├── index.rst.txt
    │   │   │       │   ├── systemrequirements.rst.txt
    │   │   │       │   └── neuron_segmentation/
    │   │   │       │       ├── design_notes.rst.txt
    │   │   │       │       ├── index.rst.txt
    │   │   │       │       ├── install.rst.txt
    │   │   │       │       └── usage.rst.txt
    │   │   │       ├── _static/
    │   │   │       │   ├── _sphinx_javascript_frameworks_compat.js
    │   │   │       │   ├── alabaster.css
    │   │   │       │   ├── basic.css
    │   │   │       │   ├── copybutton.css
    │   │   │       │   ├── copybutton.js
    │   │   │       │   ├── copybutton_funcs.js
    │   │   │       │   ├── custom.css
    │   │   │       │   ├── debug.css
    │   │   │       │   ├── doctools.js
    │   │   │       │   ├── documentation_options.js
    │   │   │       │   ├── jquery.js
    │   │   │       │   ├── language_data.js
    │   │   │       │   ├── pygments.css
    │   │   │       │   ├── searchtools.js
    │   │   │       │   ├── skeleton.css
    │   │   │       │   ├── sphinx_highlight.js
    │   │   │       │   ├── webpack-macros.html
    │   │   │       │   ├── css/
    │   │   │       │   │   ├── badge_only.css
    │   │   │       │   │   ├── theme.css
    │   │   │       │   │   └── fonts/
    │   │   │       │   │       ├── fontawesome-webfont.eot
    │   │   │       │   │       ├── fontawesome-webfont.ttf
    │   │   │       │   │       ├── fontawesome-webfont.woff
    │   │   │       │   │       ├── fontawesome-webfont.woff2
    │   │   │       │   │       ├── lato-bold-italic.woff
    │   │   │       │   │       ├── lato-bold-italic.woff2
    │   │   │       │   │       ├── lato-bold.woff
    │   │   │       │   │       ├── lato-bold.woff2
    │   │   │       │   │       ├── lato-normal-italic.woff
    │   │   │       │   │       ├── lato-normal-italic.woff2
    │   │   │       │   │       ├── lato-normal.woff
    │   │   │       │   │       ├── lato-normal.woff2
    │   │   │       │   │       ├── Roboto-Slab-Bold.woff
    │   │   │       │   │       ├── Roboto-Slab-Bold.woff2
    │   │   │       │   │       ├── Roboto-Slab-Regular.woff
    │   │   │       │   │       └── Roboto-Slab-Regular.woff2
    │   │   │       │   ├── js/
    │   │   │       │   │   ├── badge_only.js
    │   │   │       │   │   └── theme.js
    │   │   │       │   ├── scripts/
    │   │   │       │   │   ├── bootstrap.js
    │   │   │       │   │   ├── bootstrap.js.LICENSE.txt
    │   │   │       │   │   ├── furo-extensions.js
    │   │   │       │   │   ├── furo.js
    │   │   │       │   │   ├── furo.js.LICENSE.txt
    │   │   │       │   │   └── pydata-sphinx-theme.js
    │   │   │       │   ├── styles/
    │   │   │       │   │   ├── bootstrap.css
    │   │   │       │   │   ├── furo-extensions.css
    │   │   │       │   │   ├── furo.css
    │   │   │       │   │   ├── pydata-sphinx-theme.css
    │   │   │       │   │   └── theme.css
    │   │   │       │   └── vendor/
    │   │   │       │       └── fontawesome/
    │   │   │       │           ├── 6.1.2/
    │   │   │       │           │   ├── LICENSE.txt
    │   │   │       │           │   ├── css/
    │   │   │       │           │   ├── js/
    │   │   │       │           │   │   └── all.min.js.LICENSE.txt
    │   │   │       │           │   └── webfonts/
    │   │   │       │           │       ├── fa-brands-400.ttf
    │   │   │       │           │       ├── fa-brands-400.woff2
    │   │   │       │           │       ├── fa-regular-400.ttf
    │   │   │       │           │       ├── fa-regular-400.woff2
    │   │   │       │           │       ├── fa-solid-900.ttf
    │   │   │       │           │       ├── fa-solid-900.woff2
    │   │   │       │           │       ├── fa-v4compatibility.ttf
    │   │   │       │           │       └── fa-v4compatibility.woff2
    │   │   │       │           └── 6.5.1/
    │   │   │       │               ├── LICENSE.txt
    │   │   │       │               ├── css/
    │   │   │       │               ├── js/
    │   │   │       │               │   └── all.min.js.LICENSE.txt
    │   │   │       │               └── webfonts/
    │   │   │       │                   ├── fa-brands-400.ttf
    │   │   │       │                   ├── fa-brands-400.woff2
    │   │   │       │                   ├── fa-regular-400.ttf
    │   │   │       │                   ├── fa-regular-400.woff2
    │   │   │       │                   ├── fa-solid-900.ttf
    │   │   │       │                   ├── fa-solid-900.woff2
    │   │   │       │                   ├── fa-v4compatibility.ttf
    │   │   │       │                   └── fa-v4compatibility.woff2
    │   │   │       └── neuron_segmentation/
    │   │   │           ├── design_notes.html
    │   │   │           ├── index.html
    │   │   │           ├── install.html
    │   │   │           └── usage.html
    │   │   └── source/
    │   │       ├── conf.py
    │   │       ├── index.rst
    │   │       ├── systemrequirements.rst
    │   │       ├── .DS_Store
    │   │       ├── _static/
    │   │       │   └── custom.css
    │   │       ├── _templates/
    │   │       │   ├── custom-class-template.rst.txt
    │   │       │   └── custom-module-template.rst.txt
    │   │       └── neuron_segmentation/
    │   │           ├── design_notes.rst
    │   │           ├── index.rst
    │   │           ├── inference_w_pretrained.rst
    │   │           ├── install.rst
    │   │           ├── usage.rst
    │   │           └── watershed_agglomeration.rst
    │   ├── engine/
    │   │   ├── post/
    │   │   │   ├── 02_extract_fragments_worker.py
    │   │   │   ├── 03_agglomerate_worker.py
    │   │   │   ├── 04_find_segments_full.py
    │   │   │   ├── 05_extract_segmentation_from_lut.py
    │   │   │   ├── parallel_aff_agglomerate.py
    │   │   │   ├── parallel_fragments.py
    │   │   │   ├── rag.py
    │   │   │   ├── run_waterz.py
    │   │   │   ├── run_waterz_parallel.py
    │   │   │   ├── shared_rag_provider.py
    │   │   │   ├── skeletonise.py
    │   │   │   └── watershed_helpers.py
    │   │   ├── predict/
    │   │   │   ├── predict_2d.py
    │   │   │   ├── predict_3d.py
    │   │   │   ├── predict_3d_daisy_worker.py
    │   │   │   └── predict_worker_daisy.py
    │   │   └── training/
    │   │       ├── train_2d.py
    │   │       ├── train_2d_mito.py
    │   │       ├── train_3d.py
    │   │       └── __pycache__/
    │   ├── install_src/
    │   │   ├── readme.md
    │   │   ├── install_waterz.sh
    │   │   ├── waterz-0.9.5.dist-info.zip
    │   │   └── waterz.zip
    │   ├── logs/
    │   │   ├── readme.md
    │   │   └── train_logs.txt
    │   ├── metrics/
    │   │   ├── ap.py
    │   │   └── ssim_np.py
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── losses.py
    │   │   ├── models.py
    │   │   └── __pycache__/
    │   └── tests/
    │       ├── test_cfg_import.py
    │       ├── test_log_parse_for_mongo.py
    │       ├── test_log_parse_for_mongo_v2.py
    │       ├── test_padding.py
    │       └── test_preprocess_hist_match.py
    ├── proofreading/
    │   ├── readme.md
    │   ├── auto_merge_seg_w_tracings.py
    │   ├── cave_client_to_load_stacks.py
    │   └── assets/
    │       ├── plot_sim_auto_merge.zip
    │       └── segmentdots_auto_merge_pdf.zip
    ├── synful/
    │   ├── readme.md
    │   ├── assets/
    │   │   └── readme.md
    │   ├── eval/
    │   │   ├── gt/
    │   │   │   ├── hemibrain_synapse_analysis.ipynb
    │   │   │   ├── octo_synapse_analysis.ipynb
    │   │   │   └── seymour_synapse_analysis.ipynb
    │   │   ├── predictions/
    │   │   │   ├── convert_pred_in_px_to_nm.py
    │   │   │   ├── synapse_partners_pairwise.py
    │   │   │   ├── synapse_val_mito.py
    │   │   │   └── synaptic_partners_kdtree.py
    │   │   └── synapse_consensus/
    │   │       └── fetch_data_per_user.ipynb
    │   ├── pytorch/
    │   │   ├── readme.md
    │   │   ├── predicter.py
    │   │   ├── super_predicter_daisy.py
    │   │   ├── trainer.py
    │   │   ├── add_ons/
    │   │   │   ├── __init__.py
    │   │   │   ├── database.py
    │   │   │   ├── detection.py
    │   │   │   ├── evaluate_annotations.py
    │   │   │   ├── evaluation.py
    │   │   │   ├── googlebrainmaps.py
    │   │   │   ├── nms.py
    │   │   │   ├── synapse.py
    │   │   │   ├── synapse_cleftcluster.py
    │   │   │   ├── synapse_mapping.py
    │   │   │   ├── funlib_persistence/
    │   │   │   │   ├── persistence_utils.py
    │   │   │   │   └── __pycache__/
    │   │   │   └── gp/
    │   │   │       ├── __init__.py
    │   │   │       ├── add_partner_vector_map.py
    │   │   │       ├── batch_request.py
    │   │   │       ├── cloud_volume_source.py
    │   │   │       ├── extract_synapses.py
    │   │   │       ├── hdf5_points_source.py
    │   │   │       ├── intensity_scale_shift_clip.py
    │   │   │       ├── points.py
    │   │   │       ├── points_spec.py
    │   │   │       ├── predict.py
    │   │   │       ├── prepost_points_graphkey.py
    │   │   │       ├── train.py
    │   │   │       ├── unsqueeze.py
    │   │   │       ├── upsample.py
    │   │   │       └── zarr_points_source.py
    │   │   ├── config/
    │   │   │   ├── __init__.py
    │   │   │   ├── config.py
    │   │   │   ├── config_cremi.py
    │   │   │   ├── config_hemi.py
    │   │   │   ├── config_octo.py
    │   │   │   ├── config_predict.py
    │   │   │   └── config_predict_octo.py
    │   │   ├── data_utils/
    │   │   │   ├── download_data/
    │   │   │   │   └── meta_analysis/
    │   │   │   │       ├── readme.md
    │   │   │   │       ├── syn_environment.yml
    │   │   │   │       └── scripts/
    │   │   │   │           ├── readme.md
    │   │   │   │           ├── andre_synapse_meta_analysis.zip
    │   │   │   │           ├── convert_wasp_to_CREMI.py
    │   │   │   │           ├── download_local_synapses.py
    │   │   │   │           ├── download_local_synapses_same_preid.py
    │   │   │   │           ├── download_syns_from_hemibrain.py
    │   │   │   │           ├── download_syns_from_manc.py
    │   │   │   │           └── hemibrain_synapse_meta_analysis.zip
    │   │   │   └── preprocess_volumes/
    │   │   │       ├── pad_input.py
    │   │   │       └── __pycache__/
    │   │   ├── engine/
    │   │   │   ├── __init__.py
    │   │   │   ├── predict/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── predict_3d.py
    │   │   │   │   ├── predict_extract_3d.py
    │   │   │   │   └── predict_worker_daisy.py
    │   │   │   ├── single_task/
    │   │   │   │   └── PostSynapticMask/
    │   │   │   │       ├── parameter.json
    │   │   │   │       ├── predict.py
    │   │   │   │       ├── predict_scan.py
    │   │   │   │       ├── predict_template.json
    │   │   │   │       ├── torch_models_loss.py
    │   │   │   │       ├── train_st_mask.py
    │   │   │   │       └── utils.py
    │   │   │   └── training/
    │   │   │       ├── __init__.py
    │   │   │       └── train_3d.py
    │   │   ├── models/
    │   │   │   ├── __init__.py
    │   │   │   ├── losses.py
    │   │   │   └── models.py
    │   │   └── tests/
    │   │       ├── predict_extract_worker_daisy.py
    │   │       ├── predict_worker_daisy_test.py
    │   │       ├── provider_test.py
    │   │       ├── rasterize_points.py
    │   │       ├── super_predicter_daisy_test.py
    │   │       ├── super_predicter_extract_daisy_test.py
    │   │       ├── torch_train.py
    │   │       └── vector_map.py
    │   └── tensorflow/
    │       ├── pretrained/
    │       │   └── train/
    │       │       ├── readme.md
    │       │       ├── CHANGELOG.md
    │       │       ├── p_setup05/
    │       │       │   ├── example_log.txt
    │       │       │   ├── p_setup05_config.json
    │       │       │   ├── predict.py
    │       │       │   ├── predict_blockwise.py
    │       │       │   └── train_net_config.json
    │       │       ├── p_setup10/
    │       │       │   ├── p_setup10_config.json
    │       │       │   ├── predict.py
    │       │       │   ├── predict_blockwise.py
    │       │       │   └── train_net_config.json
    │       │       ├── p_setup45/
    │       │       │   ├── p_setup45_config.json
    │       │       │   ├── predict.py
    │       │       │   ├── predict_blockwise.py
    │       │       │   └── train_net_config.json
    │       │       ├── p_setup51/
    │       │       │   ├── p_setup51_config.json
    │       │       │   ├── predict.py
    │       │       │   ├── predict_blockwise.py
    │       │       │   └── train_net_config.json
    │       │       ├── p_setup52/
    │       │       │   ├── p_setup52_config.json
    │       │       │   ├── predict.py
    │       │       │   ├── predict_blockwise.py
    │       │       │   └── train_net_config.json
    │       │       └── p_setup54/
    │       │           ├── 04_predict_extract_blockwise.py
    │       │           ├── extract_parameters_setup32.json
    │       │           ├── p_setup54_config.json
    │       │           ├── predict.py
    │       │           ├── predict_and_extract.py
    │       │           ├── predict_blockwise.py
    │       │           ├── predict_extract_parameters.json
    │       │           ├── train_net_config.json
    │       │           └── worker_config.json
    │       └── train_from_scratch/
    │           ├── readme.md
    │           ├── scripts/
    │           │   ├── predict/
    │           │   │   ├── 04_predict_extract_blockwise.py
    │           │   │   ├── extract_cremi.json
    │           │   │   ├── extract_parameters_setup32.json
    │           │   │   ├── predict_and_extract.py
    │           │   │   ├── predict_blockwise.py
    │           │   │   ├── predict_extract_parameters.json
    │           │   │   └── predict_template.json
    │           │   └── train/
    │           │       └── setup03/
    │           │           ├── generate_network.py
    │           │           ├── parameter.json
    │           │           ├── predict.py
    │           │           └── train.py
    │           └── synful/
    │               ├── __init__.py
    │               ├── database.py
    │               ├── detection.py
    │               ├── evaluate_annotations.py
    │               ├── evaluation.py
    │               ├── googlebrainmaps.py
    │               ├── nms.py
    │               ├── synapse.py
    │               ├── synapse_cleftcluster.py
    │               ├── synapse_mapping.py
    │               ├── __pycache__/
    │               ├── gunpowder/
    │               │   ├── __init__.py
    │               │   ├── add_partner_vector_map.py
    │               │   ├── cloud_volume_source.py
    │               │   ├── extract_synapses.py
    │               │   ├── hdf5_points_source.py
    │               │   ├── intensity_scale_shift_clip.py
    │               │   ├── predict.py
    │               │   ├── train.py
    │               │   ├── unsqueeze.py
    │               │   └── upsample.py
    │               └── tests/
    │                   ├── test_add_partner_vector_map.py
    │                   ├── test_extract_synapses.py
    │                   └── test_synapse.py
    └── visualize/
        ├── readme.md
        ├── environment_napari.yml
        ├── environment_neuroglancer.yml
        ├── environment_neuroglancer_mac.yml
        ├── nglancer_pyconnectomics_example.py
        ├── visualize_napari.py
        ├── visualize_napari_synapses.py
        ├── visualize_nglancer_synapses_hdf.py
        ├── visualize_nglancer_synapses_mito_hdf.py
        └── visualize_synful_inference.py
```
