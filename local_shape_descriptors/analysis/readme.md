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

Troubleshooting issues: [Issue #37](https://github.com/Mohinta2892/catena/issues/37), [Issue #36](https://github.com/Mohinta2892/catena/issues/36)
  

Please install all of the above into the same conda env that you perhaps create for Napari.
Take a look at [napari_environment.yml](analysis/conda_env-requirements/lsd_analysis_py311.yml). This `env.yml` has not been tested for reproducibility of env yet! 
>[!TIP]
> Creating a separate env can help in preventing pyqt issues in the local_shape_descriptors `funkelsd` env.

