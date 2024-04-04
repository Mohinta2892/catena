# Additional package dependencies to run these files

- Install [Napari](https://napari.org/stable/tutorials/fundamentals/installation)
- Install [Kimimaro](https://github.com/seung-lab/kimimaro/tree/master)
- Install [NAVis](https://navis.readthedocs.io/en/latest/source/install.html)
- Install [Skicit-Image](https://scikit-image.org/docs/stable/user_guide/install.html)

Please install all of the above into the same conda env that you perhaps create for Napari.
Take a look at [napari_environment.yml](analysis/napari_environment.yml). This `env.yml` has not been tested for reproducibility of env yet! 
>[!TIP]
> Creating a separate env can help in preventing pyqt issues in the local_shape_descriptors `funkelsd` env.

