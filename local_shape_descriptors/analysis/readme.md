# Additional package dependencies to run these files

- Install [Napari](https://napari.org/stable/tutorials/fundamentals/installation)
- Install [Kimimaro](https://github.com/seung-lab/kimimaro/tree/master)
- Install [NAVis](https://navis.readthedocs.io/en/latest/source/install.html)
- Install [Skicit-Image](https://scikit-image.org/docs/stable/user_guide/install.html)
- Install [Skicit-Learn](https://scikit-learn.org/stable/install.html)
- Install Funlib Dependencies like below:
  ```bash
  pip install cython
  pip install git+https://github.com/funkelab/funlib.evaluate.git
  pip install git+https://github.com/funkelab/funlib.segment.git
  ```
- Install Graph-Tool. It is a requirement in funlib.evaluate, but will not be installed automatically.
```
conda install -c conda-forge graph-tool
```

**Troubleshooting issues**: [Issue #37](https://github.com/Mohinta2892/catena/issues/37), [Issue #36](https://github.com/Mohinta2892/catena/issues/36)
  

Please install all of the above into the same conda env that you perhaps create for Napari.
Take a look at [napari_environment.yml](analysis/conda_env-requirements/lsd_analysis_py311.yml). This `env.yml` has not been tested for reproducibility of env yet! 
>[!TIP]
> Creating a separate env can help in preventing pyqt issues in the local_shape_descriptors `funkelsd` env.

#### Downloading Data from CAVE or GCLOUD

- If segmentations are hosted in CAVE or in GCLOUD buckets, the scripts under [download_and_preprocess_data](https://github.com/Mohinta2892/catena/tree/dev/local_shape_descriptors/analysis/download_and_preprocess_data) can be used to download and relabel segmentations.
>[!Warning]
>Currently the segmentations are transposed to ZYX and flipped along `axes=2` in a 3D volume to match our raw EM. But you should use `napari or neuroglancer` and verify that the downloaded segmentation matches the underlying EM.
