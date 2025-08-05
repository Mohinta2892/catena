# Mitochondria Segmentation

Knowledge of the spread of mitochondria in an EM dataset is of great importance. Mitochondria segmentation can be used to not only derive biological insights but also proofread spurious synapse detections.
Hence, for mitochondria segmentation, we currently support two models: 1) a naive UNet from MONAI and 2) a custom Residual UNet inspired from [Xie et al.](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf021/8104107).
Both models and data loading has been adapted for FIBSEM isotropic datasets. However, they are not limited to it.

>[!IMPORTANT]
> You can use Local Shape Descriptors (LSDs) for mitochondria segmentation. However, the resulting predictions will generate labels for both mito and neuron segmentation jointly and in an entangled fashion.
> LSDs with when trained with mito + neuron labels, generally result in better neuron segmentation, since the model is less confused about what constitutes neuron and mito boundaries. To be clear, the model no longer considers the mito as neurons.
> You must have dense ground-truth spanning mito and neuron instances to validate the metrics.

# Getting Started

- Conda (for managing the Python environment)

## Setup
- Clone the git and go to the dev branch.
  ```bash
    git clone https://github.com/Mohinta2892/catena.git
    cd catena
    git checkout dev
    cd mitochondria_segmentation
   ```
- Create and activate the Conda environment:

```bash
  conda env create -n mito -f conda_env/torch_environment.yml
  conda activate mito
```

>[!WARNING]
> The package has been developed and tested with `python=3.8.13`. However, it should work on newer versions of `python` too. 
> The dependencies for `monai`, `torch`, `torch_metrics` and `zarr/h5py` will need to be adjusted accordingly.

# Train Models


# Infer with trained models

# Results overview



