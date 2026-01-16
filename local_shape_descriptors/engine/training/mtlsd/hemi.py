import os
import numpy as np

import torch_em
from torch_em.data.datasets import util

HEMI_PATHS = {
    "original": {
        "EB-INNER": "/media/samia/DATA/ark/dan-samia/lsd/funke/hemi/training/zarr/eb-inner-groundtruth-with-context-x20172-y2322-z14332.zarr",
        "EB-OUTER": "/media/samia/DATA/ark/dan-samia/lsd/funke/hemi/training/zarr/eb-outer-groundtruth-with-context-x20532-y3512-z14332.zarr",
        "FB-INNER": "/media/samia/DATA/ark/dan-samia/lsd/funke/hemi/training/zarr/fb-inner-groundtruth-with-context-x17342-y4052-z14332.zarr",
        "FB-OUTER": "/media/samia/DATA/ark/dan-samia/lsd/funke/hemi/training/zarr/fb-outer-groundtruth-with-context-x13542-y2462-z14332.zarr",
        "LH": "/media/samia/DATA/ark/dan-samia/lsd/funke/hemi/training/zarr/lh-groundtruth-with-context-x7737-y20781-z12444.zarr",
        "LOBULA": "/media/samia/DATA/ark/dan-samia/lsd/funke/hemi/training/zarr/lobula-groundtruth-with-context-x3648-y12800-z29056.zarr",
        "PB1": "/media/samia/DATA/ark/dan-samia/lsd/funke/hemi/training/zarr/pb-groundtruth-with-context-x8472-y2372-z9372.zarr",
        "PB2": "/media/samia/DATA/ark/dan-samia/lsd/funke/hemi/training/zarr/pb-groundtruth-with-context-x8472-y2892-z9372.zarr",
    },
    "realigned": {},
    # CREMI DEFECTS
    "defects": "/media/samia/DATA/ark/dan-samia/lsd/funke/cremi/defects/sample_ABC_padded_defects.h5"
}


# TODO add support for realigned volumes
def get_hemi_dataset(
        patch_shape,
        samples=("EB-INNER", "EB-OUTER", "FB-INNER", "FB-OUTER", "LH", "LOBULA", "PB1", "PB2"),
        use_realigned=False,
        offsets=None,
        boundaries=False,
        rois={},
        defect_augmentation_kwargs={
            "p_drop_slice": 0.025,
            "p_low_contrast": 0.025,
            "p_deform_slice": 0.0,
            "deformation_mode": "compress",
        },
        **kwargs,
):
    """Dataset for the segmentation of neurons in EM.

    This dataset is from the CREMI challenge: https://cremi.org/.
    """
    assert len(patch_shape) == 3
    if rois is not None:
        assert isinstance(rois, dict)

    # if use_realigned:
    #     # we need to sample batches in this case
    #     # sampler = torch_em.data.MinForegroundSampler(min_fraction=0.05, p_reject=.75)
    #     raise NotImplementedError
    # else:
    #     urls = CREMI_URLS["original"]
    #     checksums = CHECKSUMS["original"]

    data_paths = []
    data_rois = []
    for name in samples:
        data_paths.append(data_path)
        # WHAT IS THIS?? : this does train val split
        data_rois.append(rois.get(name, np.s_[:, :, :]))

    # can use cremi artefacts to augment hemi?!
    if defect_augmentation_kwargs is not None and "artifact_source" not in defect_augmentation_kwargs:
        # READ THE DOWNLOADED CREMI DEFECT VOLUME
        defect_path = HEMI_PATHS["defects"]
        defect_patch_shape = (1,) + tuple(patch_shape[1:])
        artifact_source = torch_em.transform.get_artifact_source(defect_path, defect_patch_shape,
                                                                 min_mask_fraction=0.75,
                                                                 raw_key="defect_sections/raw",
                                                                 mask_key="defect_sections/mask")
        defect_augmentation_kwargs.update({"artifact_source": artifact_source})

    raw_key = "volumes/raw"
    label_key = "volumes/labels/neuron_ids"

    # defect augmentations
    if defect_augmentation_kwargs is not None:
        raw_transform = torch_em.transform.get_raw_transform(
            augmentation1=torch_em.transform.EMDefectAugmentation(**defect_augmentation_kwargs)
        )
        kwargs = util.update_kwargs(kwargs, "raw_transform", raw_transform)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=False, boundaries=boundaries, offsets=offsets
    )

    return torch_em.default_segmentation_dataset(
        data_paths, raw_key, data_paths, label_key, patch_shape, rois=data_rois, **kwargs
    )


def get_hemi_loader(
        patch_shape,
        batch_size,
        samples=("EB-INNER", "EB-OUTER", "FB-INNER", "FB-OUTER", "LH", "LOBULA", "PB1", "PB2"),
        use_realigned=False,
        offsets=None,
        boundaries=False,
        rois={},
        defect_augmentation_kwargs={
            "p_drop_slice": 0.025,
            "p_low_contrast": 0.025,
            "p_deform_slice": 0.0,
            "deformation_mode": "compress",
        },
        **kwargs,
):
    """Dataset for the segmentation of neurons in EM. See 'get_hemi_dataset' for details.
    """
    dataset_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_hemi_dataset(
        patch_shape=patch_shape,
        samples=samples,
        use_realigned=use_realigned,
        offsets=offsets,
        boundaries=boundaries,
        rois=rois,
        defect_augmentation_kwargs=defect_augmentation_kwargs,
        **dataset_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
