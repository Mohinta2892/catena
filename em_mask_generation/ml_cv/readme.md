## EM MASK GENERATION

Raw EM data often is surrounding by resin and unnecessary bits of information (e.g., trachea in the **Drosophila**) which can be easily masked out before running inference on the whole CNS volume. This generally speeds up the inference, as "not to be segmented" regions in the volume are automatically
skipped. We have developed a machine learning model that enables us to generate these masks very efficiently on a lower resolution EM volume, which can then be used during inference with LSDs.

### Features

### System Requirements
- GPU: Min. 12GB Nvidia CUDA (Developed on 24GB RTX3090)
- RAM: 64GB Minimum
- OS: Linux (Rhel, developed on Ubuntu)
- Conda: For virtual python environments

  Please make a new conda env based on the `environment.yaml` shared. Note that you may have installation issues due to the `lib` files, which are specific to the machine it has been developed on. Get rid of them and try again.
  
### Training 
Please run the cells [em_mask_generation.ipynb](https://github.com/Mohinta2892/catena/blob/dev/em_mask_generation/ml_cv/em_mask_generation.ipynb) to train and test on your data. *Admittedly, this notebook is not super clean at the moment. We will work on fixing that soon.*

Please note we use MONAI for data-loading and model training.

**QUICK RECAP IMP FEATURES OF THE CODE:**
- **Input EM and MASK**: Both `.tiff` and `.zarr` accepted
- **Output**: `.tif`
- Built in Train-val-test splitting 
- Built in Early Stopping based on validation loss
- **LOSS**: Trained with `DICELOSS`
- Built in `Eval` metric analysis
- Built in inference with or without transforms



### Inference

Please run the inference cells if you wish to check how our models perform on your dataset with [checkpoints](https://www.dropbox.com/scl/fo/zr2hjfbh0iseioieubtyh/AChtLHBAtf_jWzm1ok_uzk0?rlkey=0l72wva1fjsopqhc2cmo6sq49&st=3yyxy8u5&dl=0).


### Post-Processing of output mask 
>[!IMPORTANT]
> This is an important step given that even when the model generalises well to unseen data, it is **NOT PERFECT**!

Post-process masks with this [script](https://github.com/Mohinta2892/catena/blob/dev/em_mask_generation/post_process/post_process_mask.py).
The post-processing is not perfect either, an improved version is expected soon!

