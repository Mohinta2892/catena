"""
This is a scratch file to read `neuroglancer multiscale data`/ sharded datasets via cloud volume and access any scale resolution.
To be cleaned.
"""

import numpy as np
import napari
import zarr
f = zarr.open("/media/samia/DATA/mounts/fibserver1/Leonardo-processed/leonardo_s0_one_hemisphere_clahe.zarr")
f["volume"].shape
v = napari.Viewer()
v.add_image(f["volume"][512:1024, 512:1024, 512:1024])
s = zarr.open("/media/samia/DATA/mounts/fibserver1/Leonardo-processed/leonardo_s0_one_hemisphere_seg.zarr")
s = zarr.open("/media/samia/DATA/mounts/fibserver1/Leonardo-processed/leonardo_s0_one_hemisphere_seg/seg_240318b.zarr")
s["volume"].shape
f["volume"].attrs["resolution"]
res = (30, 12, 12)
source_voxel_size = res
source_voxel_size_dims = len(source_voxel_size)
scales = np.array(source_voxel_size) / np.array(30, 24, 24)
target_voxel_size = (30,24,24)
scales = np.array(source_voxel_size) / np.array(target_voxel_size)
souce_data = f
scales = np.array(
    (1,) * (len(source_data.shape) - source_voxel_size_dims) + tuple(scales))
source_data = f
scales = np.array(
    (1,) * (len(source_data.shape) - source_voxel_size_dims) + tuple(scales))
source_data = f["volume"]
scales = np.array(
    (1,) * (len(source_data.shape) - source_voxel_size_dims) + tuple(scales))
scales
resampled_data = rescale(source_data.astype(np.float32), scales, order=interp_order,
                         anti_aliasing=False).astype(source_data.dtype)
from skimage.transform import rescale, resize
source_data = f["volume"][512:1024, 512:1024, 512:1024]
scales = np.array(
    (1,) * (len(source_data.shape) - source_voxel_size_dims) + tuple(scales))
sclae
scales
source_data.shape
interp_order=0
resampled_data = rescale(source_data.astype(np.float32), scales, order=interp_order,
                         anti_aliasing=False).astype(source_data.dtype)
resampled_data.shape
v.add_images(resampled_data)
v.add_image(resampled_data)
s.shape
seg
s
s["volume"].shape
st = np.transpose(s, (2, 1, 0))
st = np.transpose(s["volume"], (2, 1, 0))
source_data = f["volume"]
source_voxel_size = (12, 12, 30)
target_voxel_size = (24, 24, 30)
scales = np.array(
    (1,) * (len(source_data.shape) - source_voxel_size_dims) + tuple(scales))
scales
resampled_data = rescale(source_data.astype(np.float32), scales, order=interp_order,
                         anti_aliasing=False).astype(source_data.dtype)
s["volume"].shape
v.add_label(s["volume"][512:1024, 512:512+256, 512:512+256])
v.add_labels(s["volume"][512:1024, 512:512+256, 512:512+256])
v.add_labels(s["volume"][168:337, 168:337, 512:1024])
import cloudvolume
from cloudvolume import CloudVolume
vol = CloudVolume("precomputed://file:/media/samia/DATA/mounts/fibserver1/Leonardo-processed/leonardo_s0_one_hemisphere_clahe/12.0x12.0x30.0")
vol = CloudVolume("precomputed://file://media/samia/DATA/mounts/fibserver1/Leonardo-processed/leonardo_s0_one_hemisphere_clahe/12.0x12.0x30.0")
vol = CloudVolume("precomputed://file://media/samia/DATA/mounts/fibserver1/Leonardo-processed/leonardo_s0_one_hemisphere_clahe")
vol = CloudVolume("precomputed://file:///media/samia/DATA/mounts/fibserver1/Leonardo-processed/leonardo_s0_one_hemisphere_clahe")
vol
vol.info
vol.available_resolutions
vol.dataset_name
vol.image
vol.image.grid_size
vol.image.grid_size()
vol.shape
vol.available_mips
vol.image.has_data(mip=0)
vol.image.has_data(mip=1)
files = vol.download_files([...], mip=1, decompress=True)
files = vol.download_files([2264, 2597,2859], mip=1, decompress=True)
files = vol.download_files([(0,0,0),(2264, 2597,2859)], mip=1, decompress=True)
files = vol.download_files([(0,0,0), (2264, 2597,2859)], mip=1, decompress=False)
files = vol.download_files([[0,0,0], [2264, 2597,2859]], mip=1, decompress=False)
files = vol.download_files((0:2264, 0:2597,0:2859), mip=1, decompress=False)
files = vol.download_files((0:2264, 0:2597,0:2859), mip=1, decompress=False)
cloudvolume.Bbox
bbox = cloudvolume.Bbox([(0, 0,0), (2264, 2597,2859)])
bbox = cloudvolume.Bbox((0, 0,0), (2264, 2597,2859))
bbox
files = vol.download_files(bbox, mip=1, decompress=False)
bbox = cloudvolume.Bbox((0, 0,0), (2264, 2597,1859))
files = vol.download_files(bbox, mip=1, decompress=False)
files
type(fiels)
type(files)
files.keys()
files = vol.download(bbox, mip=1, decompress=True)
files = vol.download(bbox, mip=1)
type(files)
files.keys()
files.resolution
files.data
files.data.shape
files.data.dtype
data = files.data
data = np.squeeze(files.data)
data
import zar
import zarr
of = zarr.open("/media/samia/DATA/ark/dan-samia/lsd/funke/leornado/zarr/leonardo_preprocessed_gconn.zarr", "a")
of["volumes/raw"] = data
of["volumes/raw"].attrs["resolution"] = (24, 24, 30)
data.shape
data_t = np.transpose(data, (2, 1, 0))
data_t.shape
of["volumes/raw_transpose"] = data_t
of["volumes/raw_transpose"].attrs["resolution"] = (30, 24, 24)
of["volumes/raw_transpose"].attrs["offset"] = (0,0,0)
of["volumes/raw"].attrs["offset"]= (0,0,0)
