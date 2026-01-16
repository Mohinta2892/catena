import zarr
import tifffile as t
import matplotlib.pyplot as plt
import copy

neuron_seg_mask = zarr.open("/media/samia/DATA/mounts/zstore1/lsd_outputs/MTLSD/3d/run-aclsd-together/model_checkpoint_300000/wasp_train_vol0_syns_zyx_2217-2617_4038-4448_6335-6735_cremi_same_preid.zarr")["volumes/binary_pred_affs"]
mito_seg = zarr.open("/media/samia/DATA/mounts/zstore1/lsd_outputs/MTLSD/3d/clahed_hemi_mito_unproof/model_checkpoint_300000/wasp_train_vol0_syns_zyx_2217-2617_4038-4448_6335-6735_cremi_same_preid.zarr")["volumes/binary_pred_affs"]

# add_mito_to_seg = zarr.open("/media/samia/DATA/mounts/zstore1/lsd_outputs/MTLSD/3d/run-aclsd-together/model_checkpoint_300000/wasp_train_vol0_syns_zyx_2217-2617_4038-4448_6335-6735_cremi_same_preid.zarr", "a")

# mito_only = copy.deepcopy(mito_seg[...])
# mito_only[neuron_seg_mask] = 0

# plt.imshow(mito_only[mito_only.shape[0]//2, ...])
# plt.show()
#
# add_mito_to_seg["volumes/mito_ids"] = mito_only

t.imwrite("/media/samia/DATA/mounts/zstore1/lsd_outputs/MTLSD/3d/run-aclsd-together/model_checkpoint_300000/wasp_train_vol0_syns_zyx_2217-2617_4038-4448_6335-6735_neuron_seg_mask.tiff", neuron_seg_mask)
t.imwrite("/media/samia/DATA/mounts/zstore1/lsd_outputs/MTLSD/3d/run-aclsd-together/model_checkpoint_300000/wasp_train_vol0_syns_zyx_2217-2617_4038-4448_6335-6735_mito_seg.tiff", mito_seg)
