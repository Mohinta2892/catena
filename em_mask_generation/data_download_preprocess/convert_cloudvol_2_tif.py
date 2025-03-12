import cloudvolume
import tifffile as t

mip = 2
vol = cloudvolume.CloudVolume("file:///ceph.groups/mzlatic.grp/code/connectomic-tools/data/SAM_3G/processed_volumes/brain_crop_1", mip=2, parallel=True, progress=True, fill_missing=True)
print(f"volume shape {vol.shape}")

# convert to a ndarray. Only do this when it fits in memory (RAM)
data = vol[..., 0]

# save it to a tiff file
output_file = "/cephfs/smohinta/nblast_mclayton/sam_vol_for_mask.tiff"
t.imwrite(output_file, data)
