"""
This script takes a precomputed Google volume with segmentation and skeletonises the unique instances and saves them
as .swc files in independent folders.
These independent folders are based on how you divide the original data into chunks.
Note: you should look at the hard-coded lines below!!!
Author: Samia Mohinta
Affiliation: Cardona-lab, University of Cambridge, UK
"""
import cloudvolume
from cloudvolume import CloudVolume
import argparse
import dask.array as da
import numpy as np
import os
import json
import kimimaro
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster
import concurrent.futures


def get_mesh(vol):
    """Request the finest-resolution mesh for a single vol segment id.
    Adapted from: https://gist.github.com/jbms/1ec1192c34ec816c2c517a3b51a8ed6c
    """
    mesh = vol.mesh.get(100)

    # Mesh vertices are in nanometers (not voxels)
    print(list(mesh.values())[0].vertices)


def get_skeletons(vol):
    """ Request the skeleton for a single vol segment id. A skeletons folder must exist in the precomputed vol.
    Adapted from: https://gist.github.com/jbms/1ec1192c34ec816c2c517a3b51a8ed6c
    """
    skel = vol.skeleton.get(100)
    # # Skeleton vertices are in nanometers (not voxels)
    print(skel.vertices)


def read_precomp(filename):
    vol = CloudVolume(f"precomputed://file://{filename}", fill_missing=True)
    # volume statistics
    print(f"volume info {vol.info}")
    print(f"volume shape {vol.shape}")
    print(f"volume grid size {vol.image.grid_size()}")
    print(f"volume chunk size {vol.chunk_size}")

    return vol, vol.shape


def create_chunk_folders(base_path, total_chunks):
    for i in range(total_chunks):
        folder_path = os.path.join(base_path, f"chunk_{i}")
        os.makedirs(folder_path, exist_ok=True)


def save_skeleton(skeleton, folder_path, chunk_id):
    # filename = os.path.join(folder_path, f"skeleton_{chunk_id}.swc")
    # kimimaro.save_swc(skeleton, filename)
    print(f"Saving chunk id {chunk_id}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for label, skel in skeleton.items():
        fname = os.path.join(folder_path, f"{label}.swc")
        with open(fname, "wt") as f:
            f.write(skel.to_swc())


def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {"processed_chunks": []}


def save_progress(progress_file, processed_chunks):
    with open(progress_file, 'w') as f:
        json.dump({"processed_chunks": processed_chunks}, f)


def save_chunk_data_as_npz(chunk_data, folder_path, chunk_id):
    """Save the chunk data as a compressed .npz file under its respective chunk folder."""
    filename = os.path.join(folder_path, f"chunk_data_{chunk_id}.npz")
    np.savez_compressed(filename, chunk_data=chunk_data)
    print(f"Chunk data saved at {filename}")


def load_chunk_data_from_npz(folder_path, chunk_id):
    """Load the chunk data from the .npz file."""
    filename = os.path.join(folder_path, f"chunk_data_{chunk_id}.npz")
    with np.load(filename) as data:
        chunk_data = data['chunk_data']
    print(f"Chunk data loaded from {filename}")
    return chunk_data


def process_chunks(dask_array, output_base_path, progress_file, anisotropy=False, resolution=(8, 8, 8)):
    total_chunks = np.prod(np.array(dask_array.numblocks))

    print(f"total chunks: {total_chunks}")
    create_chunk_folders(output_base_path, total_chunks)
    progress = load_progress(progress_file)
    processed_chunks = set(progress["processed_chunks"])

    for chunk_id, chunk_slice in enumerate(da.core.slices_from_chunks(dask_array.chunks)):
        if chunk_id in processed_chunks:
            print(f"Skipping already processed chunk {chunk_id}")
            continue

        print(f"Processing chunk_id {chunk_id}, at chunk_slice {chunk_slice}")
        try:
            chunk_data = dask_array[chunk_slice].compute()
            chunk_data = da.squeeze(chunk_data)
            chunk_data = da.from_array(chunk_data)

            """ Debugging - uncomment if necessary
            # Ensure the chunk data is converted to a NumPy array
            # if isinstance(chunk_data, cloudvolume.VolumeCutout):
            #     chunk_data = np.array(chunk_data)
            #
            # # Check if chunk_data is valid and not scalar (0-dimensional)
            # if chunk_data.ndim == 0:
            #     print(f"Skipping chunk {chunk_id}: Data is 0-dimensional after computing.")
            #     continue
            #
            # # Debug information
            # print(f"Chunk {chunk_id} shape after squeeze: {chunk_data.shape}")
            # print(f"Chunk {chunk_id} dtype: {chunk_data.dtype}")
            # print(f"Chunk {chunk_id} min: {chunk_data.min()}, max: {chunk_data.max()}")

            # Ensure the chunk is not empty
            # if chunk_data.size == 0:
            #     print(f"Skipping empty chunk {chunk_id}")
            #     continue


            # Save the chunk data as .npy
            # folder_path = os.path.join(output_base_path, f"chunk_{chunk_id}")
            # save_chunk_data_as_npz(chunk_data, folder_path, chunk_id)
            #
            # # Read the chunk data back from the .npz file
            # chunk_data = load_chunk_data_from_npz(folder_path, chunk_id)
            """

            # # keep kimimaro defaults
            skeletons = kimimaro.skeletonize(
                chunk_data,
                teasar_params={
                    "scale": 1.5,
                    "const": 300,  # physical units
                    "pdrf_scale": 100000,
                    "pdrf_exponent": 4,
                    "soma_acceptance_threshold": 3500,  # physical units
                    "soma_detection_threshold": 750,  # physical units
                    "soma_invalidation_const": 300,  # physical units
                    "soma_invalidation_scale": 2,
                    "max_paths": 300,  # default None
                },
                # object_ids=[ ... ], # process only the specified labels
                # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
                # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
                dust_threshold=1000,  # skip connected components with fewer than this many voxels
                anisotropy=resolution,  # default True, but must pass a tuple
                fix_branching=True,  # default True
                fix_borders=True,  # default True
                fill_holes=False,  # default False
                fix_avocados=False,  # default False
                progress=True,  # default False, show progress bar
                parallel=5,  # <= 0 all cpu, 1 single process, 2+ multiprocess
                parallel_chunk_size=10,  # how many skeletons to process before updating progress bar
            )
            # Save the skeletons
            folder_path = os.path.join(output_base_path, f"chunk_{chunk_id}")
            # for i, skeleton in enumerate(skeletons.values()):
            save_skeleton(skeletons, folder_path, f"{chunk_id}")

            processed_chunks.add(chunk_id)
            save_progress(progress_file, list(processed_chunks))
            print(f"Processed chunk {chunk_id}")
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {str(e)}")

    print("All chunks processed.")


class DaskClient:
    def __init__(self):
        self.client = None
        self.cluster = None

        # Setup Dask client

    def setupDaskClientCluster(workers, cores, memory, walltime, nanny=False, death_timeout_minutes=60,
                               local_directory=None):
        from dask.distributed import Client
        from dask_jobqueue import SLURMCluster
        # Setup cluster and client
        print("Setting up SLURM cluster...")
        # deleteSlurmOutputFiles()
        username = os.getenv("USER")
        cluster = SLURMCluster(
            memory=memory,
            processes=1,
            cores=cores,
            nanny=nanny,
            walltime=walltime,
            log_directory=f"/cephfs/{username}/catena/kimimaro/logs",
            # local_directory= f"/lmb/home/{username}" if local_directory==None else local_directory,
            local_directory=local_directory,
        )
        client = Client(cluster)
        # Create workers (i.e. individual jobs) NOTE that this step is essential!
        cluster.scale(workers)
        return client, cluster

    def setupDaskClient(self, n_workers=None, cores_per_worker=100, gb_per_worker=100, walltime="10:00:00", nanny=False,
                        death_timeout_minutes=60, local_directory=None):
        if os.path.exists("/lmb/"):
            if self.client == None:
                # Get size of volumes
                if n_workers == None:
                    volume_size_gb = self.volumes[0].nbytes / 1e9
                    memory_requirement = 1.2 * volume_size_gb
                    n_workers = int(memory_requirement / gb_per_worker)
                    n_workers = 10 if n_workers < 10 else n_workers  # take at least 10 workers
                self.client, self.cluster = self.setupDaskClientCluster(n_workers, cores_per_worker,
                                                                        f"{gb_per_worker}GB",
                                                                        walltime, nanny, death_timeout_minutes,
                                                                        local_directory if local_directory is not None else 'output/')
                print(f"Dask client running at {self.client.dashboard_link}")
        # Else, if you are working on a private machine
        else:
            print("WARNING: You are not working on the LMB cluster. Dask client will not be setup.")
        # print("Waiting for workers...")
        # self.cluster.wait_for_workers(1) # wait for at least one worker

    # Run jobs in batches
    def runJobsInBatches(self, jobs, client=None, max_jobs=250):
        from dask.distributed import progress, wait

        if client is not None:
            self.client = client

        self.futures = []
        for i in range(0, len(jobs), max_jobs):
            print(f"...running jobs in batch {i} to {i + max_jobs} (/ {len(jobs)})")
            batch = jobs[i:i + max_jobs]
            cur_futures = self.client.compute(batch)
            r = progress(cur_futures)
            r = wait(cur_futures)
            self.futures += cur_futures
        return self.futures


def process_chunk_dask_v2(chunk_id, chunk_slice, dask_array, output_base_path, progress_file, anisotropy, resolution):
    try:
        print(f"Processing chunk_id {chunk_id} at chunk_slice {chunk_slice}")
        chunk_data = dask_array[chunk_slice]

        # Ensure the chunk is not empty
        if chunk_data.size == 0:
            print(f"Skipping empty chunk {chunk_id}")
            return chunk_id, False

        # Call Kimimaro skeletonization here
        skeletons = kimimaro.skeletonize(
            chunk_data,
            teasar_params={
                "scale": 1.5,
                "const": 300,
                "pdrf_scale": 100000,
                "pdrf_exponent": 4,
                "soma_acceptance_threshold": 3500,
                "soma_detection_threshold": 750,
                "soma_invalidation_const": 300,
                "soma_invalidation_scale": 2,
                "max_paths": 300,
            },
            dust_threshold=1000,
            anisotropy=resolution,
            fix_branching=True,
            fix_borders=True,
            fill_holes=False,
            fix_avocados=False,
            progress=True,
            parallel=1,
            parallel_chunk_size=10,
        )

        # Save the skeletons
        folder_path = os.path.join(output_base_path, f"chunk_{chunk_id}")
        # for i, skeleton in enumerate(skeletons.values()):
        save_skeleton(skeletons, folder_path, f"{chunk_id}")

        print(f"Processed chunk {chunk_id}")
        return chunk_id, True

    except Exception as e:
        print(f"Error processing chunk {chunk_id}: {str(e)}")
        return chunk_id, False


def process_chunks_parallel(dask_array, output_base_path, progress_file, anisotropy=False, resolution=(8, 8, 8),
                            num_workers=4):
    progress = load_progress(progress_file)
    processed_chunks = set(progress["processed_chunks"])

    tasks = []
    # jobs = []

    vol_chunk_size = 64
    total_chunks = 5520
    n_chunks = 8  # anything above 512 crashes the code, so be conservative...
    chunk_step = vol_chunk_size * n_chunks
    chunk_id = 0
    create_chunk_folders(output_base_path, total_chunks)  # make chunk folders

    vol_shape = dask_array.shape
    max_chunks = vol_shape[0] * vol_shape[1] * vol_shape[2]

    print(max_chunks, max_chunks / chunk_step)
    for z in range(0, vol_shape[0], chunk_step):
        for y in range(0, vol_shape[1], chunk_step):
            for x in range(0, vol_shape[2], chunk_step):
                chunk_slice = (slice(z, min(z + chunk_step, vol_shape[0])),
                               slice(y, min(y + chunk_step, vol_shape[1])),
                               slice(x, min(x + chunk_step, vol_shape[2])))
                if chunk_id in processed_chunks:
                    print(f"Skipping already processed chunk {chunk_id}")
                    continue

                print(f"Scheduling chunk_id {chunk_id} and roi {chunk_slice} for processing")

                # Create a delayed task for each chunk
                task = delayed(process_chunk_dask_v2)(chunk_id, chunk_slice, dask_array, output_base_path,
                                                      progress_file,
                                                      anisotropy, resolution)
                tasks.append(task)

                chunk_id += 1  # increment chunk id

    print(f"num tasks {len(tasks)}")
    print(f"Total number of chunks processed: {chunk_id}")

    dask_client = DaskClient()
    # the params you pass matter. `max_jobs` should be low. Do not spawn too many workers, try to give each worker as
    # much memory as possible (at least >= the chunk_size above).
    # The local directory must have enough space and you should have unrestricted r/w access.
    # `nanny=False` is recommended.
    dask_client.setupDaskClient(n_workers=10, cores_per_worker=10, gb_per_worker=512, walltime="10:00:00", nanny=False,
                                death_timeout_minutes=60, local_directory=output_base_path)
    futures = dask_client.runJobsInBatches(jobs=tasks[20:], max_jobs=5)

    # # Update the progress after all tasks have completed
    # for chunk_id, success in results:
    #     if success:
    #         processed_chunks.add(chunk_id)
    # don't know what this is?
    # save_progress(progress_file, list(futures))
    print("All chunks processed.")
    return dask_client


def main(args):
    pass


@dask.delayed
def load_volume_chunk(vol, xStart, yStart, zStart, n_chunks, chunk_size=64):
    chunk_range = n_chunks * chunk_size
    cur_chunk = vol[xStart:xStart + chunk_range, yStart:yStart + chunk_range, zStart:zStart + chunk_range]
    cur_chunk *= 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Pull Skeletons from a Google Precomputed array with Cloud-Volume")
    # Octo path:/ceph.groups/mzlatic.grp/code/em_code/data/octo/8x8x8_cropped_seg/seg_240318a/
    parser.add_argument('--pre_file',
                        default='/ceph.groups/mzlatic.grp/code/em_code/data/octo/8x8x8_cropped_seg/seg_240318a/',
                        help="/Path/to/your/precomputed/segmentation")

    args = parser.parse_args()

    """
    Debugging purposes..
    # main(args)

    # dask_client = DaskClient()
    # dask_client.setupDaskClient(n_workers=10, cores_per_worker=10, gb_per_worker=256, walltime="10:00:00", nanny=False,
    #                             death_timeout_minutes=60,
    #                             local_directory="/net/fibserver1/data/raw/smohinta_data/kimimaro_out_octo")
    # jobs = []
    # n_chunks = 20
    # chunk_range = 64 * n_chunks
    # for i in range(0, chunk_range * 10, chunk_range):
    #     jobs.append(load_volume_chunk(vol, i, 0, 0, n_chunks=20))
    #
    # print(dask_client.client.dashboard_link)
    #
    # dask_client.client.compute(jobs)
    """
    vol, vol_shape = read_precomp(filename=args.pre_file)
    d_arr = vol  # do not convert into dask.array
    print(f"dask array shape in XYZ {d_arr.shape}")

    dask_client = process_chunks_parallel(dask_array=d_arr,
                                          output_base_path="/net/fibserver1/data/raw/smohinta_data/kimimaro_out_octo",
                                          progress_file="/net/fibserver1/data/raw/smohinta_data/kimimaro_logs/progress_octo.json",
                                          resolution=(8, 8, 8,),
                                          )
