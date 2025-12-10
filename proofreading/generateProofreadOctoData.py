"""

The following imported libraries are your dependencies.

!pip install cloud-volume
!pip install dask
!pip install caveclient
!pip install igneous-pipeline
!pip install tqdm
!pip install numpy
!pip install tensorstore

The current script is currently customised for our needs and cannot be used as off-the-shelf.
However, it contains the generic pipeline on how to extract the segmentations out of CAVE.

We are working on cleaning this up and releasing a plug-n-play script.

Authors: Michael Clayton, Samia Mohinta
Affiliations: MRC LMB and University of Cambridge

"""

import os
import sys
import numpy as np
import dask
import tensorstore as ts
from tqdm import tqdm
import igneous.task_creation as tc
from taskqueue import TaskQueue
from cloudvolume import CloudVolume
from caveclient import CAVEclient
from cloudvolume.lib import Bbox

connectomic_tools_path = os.getenv("connectomic_tools_path")  # custom dependency
sys.path.append(connectomic_tools_path)
from classes.EMAlignment import setupDaskClient # Example of classes are under SLURMClient.py script.
from classes.NeuroglancerViewer import NeuroglancerViewer

# Import Graphene segmentation
flywire_token = "x"
client = CAVEclient(server_address='https://global.connectomics.braininbrain.org', datastack_name='ds_name',
                    auth_token=flywire_token)
timestamp = 1750665448  # this is CRUCIAL
src_vol = client.info.segmentation_cloudvolume(agglomerate=True, mip=0)
src_vol_down = client.info.segmentation_cloudvolume(agglomerate=True, mip=2)
whole_bbox = Bbox([0, 0, 0], list(src_vol.shape)[:3])
whole_bbox_down = Bbox([0, 0, 0], list(src_vol_down.shape)[:3])

# Calculate difference in resolution between mip=0 and mip=2
resolution_difference = np.round(np.divide(src_vol.shape, src_vol_down.shape)).astype(int)

# Define proof-read IDs
proofread_ids = [
    648518346360884739,
    648518346370540737,
    648518346364422034,
    648518346360360026,
    # 648518346362705549,
    648518346364413735,
    648518346363644529,
    648518346362749310,
    648518346362801892,
    648518346367787881,
    648518346362429739,
    648518346363695060,
]

translation_dictionary = {
    0: 0,
    648518346360884739: 1,
    648518346370540737: 2,
    648518346364422034: 3,
    648518346360360026: 4,
    # 648518346362705549: 5,
    648518346362705549: 0,  # remove neuron 5 due to issues
    648518346364413735: 6,
    648518346363644529: 7,
    648518346362749310: 7,
    648518346362801892: 8,
    648518346367787881: 8,
    648518346362429739: 9,
    648518346363695060: 10,
}

# Setup SLURM client
worker_memory = 50
walltime = "32:00:00"
client, cluster = setupDaskClient(workers=30, cores=15, memory=f"{worker_memory}GB", walltime=walltime)
print(client.dashboard_link)

# Set to stop dask from loading too many jobs at once and running out of memory
dask.config.set({'distributed.scheduler.worker-saturation': 0.1})
dask.config.set({'distributed.scheduler.worker-ttl': None})


@dask.delayed
def set_data(source_vol, target_vol, bbox, seg_ids, timestamp):
    x_start, y_start, z_start, x_end, y_end, z_end = bbox
    bbox = Bbox([x_start, y_start, z_start], [x_end, y_end, z_end])
    r = source_vol.download(bbox=bbox, agglomerate=False, timestamp=timestamp, segids=seg_ids)
    if r.max() != 0:
        # Vectorized approach using np.vectorize
        replace_func = np.vectorize(lambda x: translation_dictionary.get(x, x))  # Keep original if not in dict
        r_replaced = replace_func(r)
        r_replaced = r_replaced.astype(np.uint64)
        # Add to store
        target_vol[x_start:x_end, y_start:y_end, z_start:z_end] = r_replaced


# Initialise downsampled segmentation
chunk_size = 320
segmentation_down_path = f"{connectomic_tools_path}/analyses/OctoProofreading/segmentation_down_2"
if not (os.path.exists(segmentation_down_path)):
    #  Initialise downsampled segmentation
    segmentation_down = ts.open({
        'driver': 'neuroglancer_precomputed',
        'kvstore': {'driver': 'file', 'path': segmentation_down_path},
        'scale_index': 0,
        'scale_metadata': {
            'chunk_size': [chunk_size, chunk_size, chunk_size],
            'encoding': 'compressed_segmentation',
            'key': '32.0x32.0x32.0',
            'resolution': [32, 32, 32],
        },
    },
        create=True,
        dtype=ts.uint64,
        shape=list(src_vol_down.shape),
        delete_existing=True,
    ).result()
    # Get jobs
    jobs = []
    for x in tqdm(range(0, src_vol_down.shape[0], chunk_size)):
        for y in range(0, src_vol_down.shape[1], chunk_size):
            for z in range(0, src_vol_down.shape[2], chunk_size):
                bbox = [x, y, z, x + chunk_size, y + chunk_size, z + chunk_size]
                r = set_data(src_vol_down, segmentation_down, bbox, proofread_ids, timestamp)
                jobs.append(r)
    # Run jobs
    r = client.compute(jobs)

else:

    # Load downsampled segmentation
    segmentation_down = CloudVolume(f"precomputed://file://{segmentation_down_path}", mip=0, fill_missing=True)

# ---------------------------------------
# Generate full resolution image
# ---------------------------------------

# Get indices of filled chunks
full_res_bbox = []
for x in tqdm(range(0, src_vol_down.shape[0], chunk_size)):
    for y in range(0, src_vol_down.shape[1], chunk_size):
        for z in range(0, src_vol_down.shape[2], chunk_size):
            bbox = [x, y, z, x + chunk_size, y + chunk_size, z + chunk_size]
            bbox_string = f"{x}-{x + chunk_size}_{y}-{y + chunk_size}_{z}-{z + chunk_size}"
            if os.path.exists(f"{segmentation_down_path}/32.0x32.0x32.0/{bbox_string}"):
                # Get chunk range in high-res quality
                bbox_res = [
                    x * resolution_difference[0],
                    y * resolution_difference[1],
                    z * resolution_difference[2],
                    (x + chunk_size) * resolution_difference[0],
                    (y + chunk_size) * resolution_difference[1],
                    (z + chunk_size) * resolution_difference[2],
                ]
                full_res_bbox.append(bbox_res)

# Initialise full segmentation
chunk_size = 320
segmentation_path = f"{connectomic_tools_path}/analyses/OctoProofreading/segmentation_full"
segmentation_full = ts.open({
    'driver': 'neuroglancer_precomputed',
    'kvstore': {'driver': 'file', 'path': segmentation_path},
    'scale_index': 0,
    'scale_metadata': {
        'chunk_size': [chunk_size, chunk_size, chunk_size],
        'encoding': 'compressed_segmentation',
        'key': '8.0x8.0x8.0',
        'resolution': [8, 8, 8],
    },
},
    create=True,
    dtype=ts.uint64,
    shape=list(src_vol.shape),
    delete_existing=True,
).result()

# Get jobs
jobs = []
for bbox in tqdm(full_res_bbox[1:]):
    x_start, y_start, z_start, x_end, y_end, z_end = bbox
    for x in range(x_start, x_end, chunk_size):
        for y in range(y_start, y_end, chunk_size):
            for z in range(z_start, z_end, chunk_size):
                bbox = [x, y, z, x + chunk_size, y + chunk_size, z + chunk_size]
                r = set_data(src_vol, segmentation_full, bbox, proofread_ids, timestamp)
                jobs.append(r)

# Run jobs
r = client.compute(jobs)
print(client.dashboard_link)

# ---------------------------------------
# Run meshing and skeletonisation
# ---------------------------------------

from taskqueue import LocalTaskQueue
import igneous.task_creation as tc

# Mesh on 8 cores, use True to use all cores
cloudpath = f"precomputed://file://{segmentation_path}"
tq = LocalTaskQueue(parallel=8)
tasks = tc.create_meshing_tasks(cloudpath, mip=0, shape=(256, 256, 256))
tq.insert(tasks)
tq.execute()
tasks = tc.create_mesh_manifest_tasks(cloudpath)
tq.insert(tasks)
tq.execute()
print("Done!")

# First Pass: Generate Skeletons
tasks = tc.create_skeletonizing_tasks(
    cloudpath,
    mip=0,  # Which resolution to skeletionize at (near isotropic is often good)
    shape=np.array([512, 512, 512]),  # size of individual skeletonizing tasks (not necessary to be chunk aligned)
    sharded=False,  # Generate (true) concatenated .frag files (False) single skeleton fragments
    spatial_index=False,  # Generate a spatial index so skeletons can be queried by bounding box
    info=None,  # provide a cloudvolume info file if necessary (usually not)
    fill_missing=False,  # Use zeros if part of the image is missing instead of raising an error
    # see Kimimaro's documentation for the below parameters
    teasar_params={'scale': 10, 'const': 10},
    object_ids=None,  # Only skeletonize these ids
    mask_ids=None,  # Mask out these ids
    fix_branching=True,  # (True) higher quality branches at speed cost
    fix_borders=True,  # (True) Enable easy stitching of 1 voxel overlapping tasks
    dust_threshold=1000,  # Don't skeletonize below this physical distance
    progress=False,  # Show a progress bar
    parallel=1,  # Number of parallel processes to use (more useful locally)
    cross_sectional_area=False,  # Compute the cross sectional area for each vertex.
    cross_sectional_area_smoothing_window=5,  # Rolling average of vertices.
)

tq.insert(tasks)
tq.execute()

# Second Pass: Fuse Skeletons (unsharded version)
tasks = tc.create_unsharded_skeleton_merge_tasks(
    f"precomputed://file://{segmentation_path}",
    crop=0,  # in voxels
    magnitude=3,  # same as mesh manifests
    dust_threshold=4000,  # in nm
    tick_threshold=6000,  # in nm
    delete_fragments=False  # Delete scratch files from first stage
)
tq.insert(tasks)
tq.execute()

# ---------------------------------------
# View skeletons
# ---------------------------------------
import matplotlib.pylab as plt

skeletons = []
for i in range(1, 11):
    try:
        skeleton = nv.segmentation.skeleton.get(i)
    except:
        print(f"Skipping {i}!")
        continue
    skeletons.append(skeleton.vertices)

plt.close("all")
fig = plt.figure(figsize=(20, 20))
for i in range(len(skeletons)):
    skel = skeletons[i]
    plt.scatter(skel[:, 1], skel[:, 2])

plt.tight_layout()
plt.savefig("tmp.png", dpi=300)

# ---------------------------------------
# View segmentation
# ---------------------------------------
import os
import sys
import neuroglancer
from cloudvolume import CloudVolume

connectomic_tools_path = os.getenv("connectomic_tools_path")
sys.path.append(connectomic_tools_path)
from classes.NeuroglancerViewer import NeuroglancerViewer

#  Load volume
nv = NeuroglancerViewer(downsample_order=0)
segmentation_path = f"{connectomic_tools_path}/analyses/OctoProofreading/segmentation_full"
nv.segmentation = CloudVolume(f"precomputed://file://{segmentation_path}", mip=0, fill_missing=True,
                              mesh_dir="mesh_mip_0_err_40", skel_dir="skeletons_mip_0")
nv.scales = [8, 8, 8, 1]
nv.dimensions = neuroglancer.CoordinateSpace(
    names=["x", "y", "z", "c^"],
    units=["nm", "nm", "nm", ""],
    scales=[8, 8, 8, 1]
)

# Set up skeleton source
skeleton_source = nv.create_skeleton_source()
segmentation_with_skeletons = nv.add_segmentation_layer_with_skeletons(nv.segmentation, skeleton_source, nv.dimensions)
with nv.viewer.txn() as s:
    # s.layers["seg"] = neuroglancer.ImageLayer(
    #     source="precomputed://gs://fly-larva-sf/octo/seg_241224_250131b_rsg8_spl"
    # )
    s.layers["clahe"] = neuroglancer.ImageLayer(
        source="precomputed://gs://fly-larva-sf/octo/clahe"
    )
    s.layers.append(name="Segmentation", layer=segmentation_with_skeletons)
    s.layers["Segmentation"].skeleton_rendering.mode2d = "lines"
    s.layers["Segmentation"].skeleton_rendering.line_width2d = 2
    s.layers["Segmentation"].skeleton_rendering.mode3d = "lines"
    s.layers["Segmentation"].skeleton_rendering.line_width3d = 2

nv.viewer
nv.go_to_location([12012, 8211, 7481])
