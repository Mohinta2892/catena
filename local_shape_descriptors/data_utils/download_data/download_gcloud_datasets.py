# !pip install tensorstore
# !pip install neuprint-python
# !pip install navis
# !pip install ipykernel
#
# # also run this in the conda env
# !conda install -c conda-forge ipywidgets
# !jupyter nbextension enable --py widgetsnbextension
# !jupyter labextension install @jupyter-widgets/jupyterlab-manager

# creds: Run this first
NEUPRINT_APPLICATION_CREDENTIALS = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InNtMjY2N0BjYW0uYWMudWsiLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0p2WTBzQURieG1OMW11OFAySldsb2Q4U296alpaajc3c1NJbG5RdXk1bD1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg3NTc0Mzc3N30.oyY1HURafdq3mZAr1TU82M1Lr2TpG4q2HRB42LcnZ90"
from neuprint import Client, fetch_skeleton, fetch_mitochondria, skeleton_segments

import tensorstore as ts
import numpy as np
import zarr

client = Client('neuprint.janelia.org', 'hemibrain:v1.2.1', token=NEUPRINT_APPLICATION_CREDENTIALS)

# HEMI 10202
# x1 = 15035
# x2 = 15635
# y1 = 28559
# y2 = 29159
# z1 = 9602
# z2 = 10202

# MANC 15000
x1 = 23400
x2 = 24000
y1 = 24000
y2 = 24600
z1 = 14400
z2 = 15000

# dif_x = (x2 - x1) // 2
# dif_y = (y2 - y1) // 2
# dif_z = (z2 - z1) // 2
# c_x = dif_x + x1
# c_y = dif_y + y1
# c_z = dif_z + z1
#
# # calculate the Roi bounding box
# c = np.array([c_x, c_y, c_z])
# box = np.array([c[0] - dif_x, c[0] + dif_x, c[1] - dif_y, c[1] + dif_y, c[2] - dif_z, c[2] + dif_z])
# # box = np.array([c - 256, c + 256]) # shape 512 x 512 x 512
# # print(box)
#
# # [(x,y,z), (X,Y,Z)] = box
#
# x, y, z = box[0], box[2], box[4]
# X, Y, Z = box[1], box[3], box[5]

x, y, z = x1, y1, z1
X, Y, Z = x2, y2, z2

print([(x, y, z), (X, Y, Z)])

# Adapted from the TensorStore tutorial:
# https://google.github.io/tensorstore/python/tutorial.html#reading-the-janelia-flyem-hemibrain-dataset
# soma_dataset_future = ts.open({
#     'driver':
#         'neuroglancer_precomputed',
#     'kvstore':
#         'gs://neuroglancer-janelia-flyem-hemibrain/v1.1/segmentation/',
#     # Use 100MB in-memory cache.
#     'context': {
#         'cache_pool': {
#             'total_bytes_limit': 100_000_000
#         }
#     },
#     'recheck_cached_data':
#         'open',
# })

## MANC

soma_dataset_future = ts.open({
    'driver':
        'neuroglancer_precomputed',
    'kvstore':
        'gs://manc-seg-v1p2/manc-seg-v1.2',
    # Use 100MB in-memory cache.
    'context': {
        'cache_pool': {
            'total_bytes_limit': 100_000_000
        }
    },
    'recheck_cached_data':
        'open',
})

# raw EM non-CLAHE?

# em_dataset_future = ts.open({
#     'driver':
#         'neuroglancer_precomputed',
#     'kvstore':
#         'gs://neuroglancer-janelia-flyem-hemibrain/emdata/raw/jpeg',
#     # Use 100MB in-memory cache.
#     'context': {
#         'cache_pool': {
#             'total_bytes_limit': 100_000_000
#         }
#     },
#     'recheck_cached_data':
#         'open',
# })

## MANC Non clahe
em_dataset_future = ts.open({
    'driver':
        'neuroglancer_precomputed',
    'kvstore':
        'gs://flyem-vnc-2-26-213dba213ef26e094c16c860ae7f4be0/emdata/jpeg',
    # Use 100MB in-memory cache.
    'context': {
        'cache_pool': {
            'total_bytes_limit': 100_000_000
        }
    },
    'recheck_cached_data':
        'open',
})

# CLAHE in YZ
#
# em_clahe_dataset_future = ts.open({
#     'driver':
#         'neuroglancer_precomputed',
#     'kvstore':
#         'gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg',
#     # Use 100MB in-memory cache.
#     'context': {
#         'cache_pool': {
#             'total_bytes_limit': 100_000_000
#         }
#     },
#     'recheck_cached_data':
#         'open',
# })


em_clahe_dataset_future = ts.open({
    'driver':
        'neuroglancer_precomputed',
    'kvstore':
        'gs://flyem-vnc-2-26-213dba213ef26e094c16c860ae7f4be0/v3_emdata_clahe_xy/jpeg',
    # Use 100MB in-memory cache.
    'context': {
        'cache_pool': {
            'total_bytes_limit': 100_000_000
        }
    },
    'recheck_cached_data':
        'open',
})

# # mitochondria -nndividual
# mito_future = ts.open({
#     'driver':
#         'neuroglancer_precomputed',
#     'kvstore':
#         'gs://neuroglancer-janelia-flyem-hemibrain/v1.2/mito-objects',  # individual mito
#     # 'gs://neuroglancer-janelia-flyem-hemibrain/v1.2/mito-objects-grouped',  # grouped mito, where they all match their parent neuron
#     # Use 100MB in-memory cache.
#     'context': {
#         'cache_pool': {
#             'total_bytes_limit': 100_000_000
#         }
#     },
#     'recheck_cached_data':
#         'open',
# })
# # mitochondria -grouped, matched to parent neuron - donno what this means exactly??
#
# mito_grp_future = ts.open({
#     'driver':
#         'neuroglancer_precomputed',
#     'kvstore':
#     #         'gs://neuroglancer-janelia-flyem-hemibrain/v1.2/mito-objects',  # individual mito
#         'gs://neuroglancer-janelia-flyem-hemibrain/v1.2/mito-objects-grouped',
#     # grouped mito, where they all match their parent neuron
#     # Use 100MB in-memory cache.
#     'context': {
#         'cache_pool': {
#             'total_bytes_limit': 100_000_000
#         }
#     },
#     'recheck_cached_data':
#         'open',
# })

# all arrays should be in XYZ orientation, however we want them to have ZYX for compatibility with existing code

soma_dset = soma_dataset_future.result()[ts.d['channel'][0]]  # strip the channel dim as it is "1": grayscale
soma_vol = np.transpose(soma_dset[x:X, y:Y, z:Z].read().result(), (2, 1, 0))

em_dset = em_dataset_future.result()[ts.d['channel'][0]]
em_vol = np.transpose(em_dset[x:X, y:Y, z:Z].read().result(), (2, 1, 0))

em_clahe_dset = em_clahe_dataset_future.result()[ts.d['channel'][0]]
em_clahe_vol = np.transpose(em_clahe_dset[x:X, y:Y, z:Z].read().result(), (2, 1, 0))
#
# mito_dset = mito_future.result()[ts.d['channel'][0]]
# mito_vol = np.transpose(mito_dset[x:X, y:Y, z:Z].read().result(), (2, 1, 0))
#
# mito_grp_dset = mito_grp_future.result()[ts.d['channel'][0]]
# mito_grp_vol = np.transpose(mito_dset[x:X, y:Y, z:Z].read().result(), (2, 1, 0))

# FIX ME: We do not do transform the data dtypes before saving as zarr datasets, but good to know
print("Checking dtypes to see if conversion is need or not!")
print(f"EM vol {em_vol.dtype}, EM vol CLAHE {em_clahe_vol.dtype},\
      Soma labels {soma_vol.dtype}, ")
      # f"Mito labels {mito_vol.dtype}")

# we check that the Roi has tissue data is not empty
countzero = not np.all(em_vol)
assert countzero, "Raw is empty"

f = zarr.open(
    f"/media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/mito_test/manc_x{x1}-{x2}_y{y1}-{y2}_z{z1}-{z2}.zarr",
    "a")
# these are somas but we call them neuronids since they are essentially that. Also, helps maintain consistency!
f["volumes/labels/neuron_ids"] = soma_vol  # potentially uint64
f["volumes/labels/neuron_ids"].attrs["offset"] = (0, 0, 0)
f["volumes/labels/neuron_ids"].attrs["resolution"] = (8, 8, 8)
# f["volumes/labels/neuron_ids"].attrs["crop_central_coords_xyz"] = (c_x, c_y, c_z)

f["volumes/raw"] = em_vol  # potentially uint8
f["volumes/raw"].attrs["offset"] = (0, 0, 0)
f["volumes/raw"].attrs["resolution"] = (8, 8, 8)
# f["volumes/raw"].attrs["crop_central_coords_xyz"] = (c_x, c_y, c_z)

f["volumes/raw_clahe"] = em_clahe_vol  # potentially uint8
f["volumes/raw_clahe"].attrs["offset"] = (0, 0, 0)
f["volumes/raw_clahe"].attrs["resolution"] = (8, 8, 8)
# f["volumes/raw_clahe"].attrs["crop_central_coords_xyz"] = (c_x, c_y, c_z)

f["volumes/raw_clahe"] = em_clahe_vol  # potentially uint8
f["volumes/raw_clahe"].attrs["offset"] = (0, 0, 0)
f["volumes/raw_clahe"].attrs["resolution"] = (8, 8, 8)
# f["volumes/raw_clahe"].attrs["crop_central_coords_xyz"] = (c_x, c_y, c_z)

# f["volumes/labels/mito_ids"] = mito_vol  # potentially uint64
# f["volumes/labels/mito_ids"].attrs["offset"] = (0, 0, 0)
# f["volumes/labels/mito_ids"].attrs["resolution"] = (8, 8, 8)
# # f["volumes/labels/mito_ids"].attrs["crop_central_coords_xyz"] = (c_x, c_y, c_z)
#
# f["volumes/labels/mito_grp_ids"] = mito_grp_vol  # potentially uint64
# f["volumes/labels/mito_grp_ids"].attrs["offset"] = (0, 0, 0)
# f["volumes/labels/mito_grp_ids"].attrs["resolution"] = (8, 8, 8)
# f["volumes/labels/mito_grp_ids"].attrs["crop_central_coords_xyz"] = (c_x, c_y, c_z)
