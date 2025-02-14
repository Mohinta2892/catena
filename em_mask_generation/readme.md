# EM Mask Generation (Tissue labelling: delineates brain and non-brain regions)
EM: *Drosophila* Larva FIBSEM downsampled

Classes: 
- Brain = Any tissue that forms the brain and ventral nerve cord
- Non-brain = Resin, tracheal membranes or any other sections of the volume that do not need to be segmented

## Conventional CV
We apply a `Diffusion of Gaussion (DOG)` and a `local texture and structure analysis`-based maskings.
The DOG does not lead to good masks. However., the texture-based masking does.
See example below.

<img src="https://github.com/Mohinta2892/catena/blob/dev/em_mask_generation/conventional_cv/example_outputs/TEXTURE_mask_filled-ezgif.com-speed.gif" width="400" height="400" />


Even though, it can generate a good starting mask that separates background from biological, 
there is no way to mask the other tissues (e.g. tracheal membranes). Hence, we train a ML model to do this.

## Machine learning
We use a super simple `UNET` to train with downsampled version of an EM volume with its corresponding semantic labels for brain and non-brain.
We use MONAI to train this network, with minimum data augmentation to allow generalisation to other EM volumes.
This model is trained in pixel space (voxel resolution is not taken into account).
The model at inference generalises well (though not perfect as can be expected) to completely unseen EM volumes. 
A post processing step must follow after model inference to fill in gaps or grow/erode the masks to properly overlay it on the EM.

A qualitative comparison of the predicted and post-processed masks overlayed on the unseen EM dataset can be found [here.](https://www.dropbox.com/scl/fi/a7c8wqm4xtanwewxrobs0/pred_post_octo-ezgif.com-optimize.gif?rlkey=f9qg0slbpbr96lg20m91waz60&st=agpb6b9h&dl=0)
