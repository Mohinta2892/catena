import h5py
import zarr
import numpy as np

syn = h5py.File(
    "/media/samia/DATA/mounts/zstore1/catena/data/OCTO_CORRECTED_SAME_PREID/data_3d/train/octo_cube3_calyx_5603_6267_y3254_3890_z7464_8163.hdf",
    "a")

seg = zarr.open(
    "/media/samia/DATA/mounts/fibserver1/smohinta_data/lsd_outputs/MTLSD/3d/run-aclsd-together/model_checkpoint_latest/octo_cube3_calyx_5347_6523_y2998_4146_z7208_8419/octo_cube3_calyx_5347_6523_y2998_4146_z7208_8419.zarr")

seg_v = seg["volumes/final_segmentation_hist_quant_60_10"]
print(seg_v.shape)
print(syn["volumes/raw"].shape)

seg_v_c = seg_v[256:-256, 256:-256, 256:-256]
print(seg_v_c.shape)

syn["volumes/labels/neuron_ids"] = seg_v_c

syn["volumes/labels/neuron_ids"].attrs["resolution"] = (8, 8, 8)
syn["volumes/labels/neuron_ids"].attrs["offset"] = (0, 0, 0)
