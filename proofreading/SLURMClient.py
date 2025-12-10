"""
Includes functions that can be used for setting up DASK clients.

Developer: Michael Clayton
Affiliation: MRC LMB

"""

import os
import gc
from time import sleep
LS_analysis_path = os.getenv("LS_analysis_path")

# Setup Dask client
def setupDaskClient(workers, cores, memory, walltime, nanny=False, death_timeout_minutes=60, local_directory=None):
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
        log_directory=f"/lmb/home/{username}/logs",
        # local_directory= f"/lmb/home/{username}" if local_directory==None else local_directory,
        local_directory = LS_analysis_path if local_directory==None else local_directory,
    )
    client = Client(cluster)
    # Create workers (i.e. individual jobs) NOTE that this step is essential!
    cluster.scale(workers)
    return client, cluster

def getWorkerMemoryUsage(client):
    # Get the scheduler information
    scheduler_info = client.scheduler_info()
    # Print the memory usage for each worker
    memory_usage = []
    for worker, info in scheduler_info['workers'].items():
        cur_memory = info["metrics"]["memory"] / 1024 ** 3
        memory_usage.append(cur_memory)
    return memory_usage

def waitForWorkers(client, min_workers):
    """Wait for minimum numbers of workers"""
    print("Waiting for workers...")
    while len(client.scheduler_info()["workers"]) < min_workers:
        sleep(1.0)

def clear_worker_memory(dask_worker):
    gc.collect()