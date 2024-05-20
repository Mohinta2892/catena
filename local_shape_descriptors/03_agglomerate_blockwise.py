"""
We are going to try to do this as in the old scripts, without calling parallel fragments.
Todo: Switch to this later since we do not have to make the task then!
This script performs watershed and an initial agglomeration (as set in `EPSILON_THRESHOLD`
in `config_predict.py` on a chunk by chunk basis.
NB: Hardcoded chunk values appear in `main`. TODO: control them via config_predict.py.
NB 2: Re. visualising fragments (chunk segmentations), napari may not be the best visualiser unless passed
an explicit color map in the range of the segmentation values. This is because it displays really large
values as a gradient colormap (in shades of green). TODO: Try neuroglancer visualisation.
"""

import json
import hashlib
import logging
import lsd
import numpy as np
import os
import daisy
import sys
import time
import pymongo
from funlib.geometry import Roi, Coordinate
from funlib.persistence import open_ds
import subprocess
from re import sub

# add current directory to path and allow absolute imports
sys.path.insert(0, '.')
from config.config_predict import *
from data_utils.preprocess_volumes.utils import calculate_min_2d_samples
from glob import glob
from add_ons.funlib_persistence.persistence_utils import *

logging.basicConfig(level=logging.INFO)
logging.getLogger('lsd.parallel_aff_agglomerate').setLevel(logging.DEBUG)
module_logger = logging.getLogger(__name__)


def agglomerate(
        cfg,
        block_size,
        context,
        sample_name,
        fragments_in_xy=False,
        initial_epsilon_agglomerate=0.0,
        agglomerate_until=0.0,
        mask_file=None,
        mask_dataset=None,
        filter_fragments=0,
        drop=False,
        **kwargs):
    """Run agglomeration in parallel blocks. Requires that affinities have been
    predicted before.

    Args:
        TODO

    """

    # Add dbname and dbhost to cfg
    cfg.DATA.DB_NAME = db_name
    cfg.DATA.DB_HOST = db_host

    # network_dir = os.path.join(experiment, setup, str(iteration))

    # copied from WPatton's mutex_agglomerate git repo
    client = pymongo.MongoClient(db_host)
    db = client[db_name]

    completed_collection_name = f"{sample_name}_agglom_blocks_completed_{str(agglomerate_until).replace('.', '')}"
    cfg.DATA.DB_COLLECTION_NAME = completed_collection_name
    completed_collection = None

    if completed_collection_name in db.list_collection_names():
        completed_collection = db[completed_collection_name]
        if drop:
            print(f"dropping {completed_collection}")
            db.drop_collection(completed_collection)
    if f"{sample_name}_edges" in db.list_collection_names():
        edges_collection = db[f"{sample_name}_edges"]
        if drop:
            print(f"dropping {edges_collection}")
            db.drop_collection(edges_collection)

    if completed_collection_name not in db.list_collection_names():
        completed_collection = db[completed_collection_name]
        completed_collection.create_index(
            [("block_id", pymongo.ASCENDING)], name="block_id"
        )
    # NB: Caching completed `block_ids` is removed; not sure why!!
    # Check later.

    logging.info("Reading affs from %s", cfg.DATA.SAMPLE)
    affs_dataset = "volumes/pred_affs"  # is always this
    cfg.INS_SEGMENT.AFFS_DS = affs_dataset  # add here to be reused in worker
    try:
        affs = open_ds(cfg.DATA.SAMPLE, affs_dataset, mode='r')
    except Exception as e:
        # perhaps for a n5
        affs_dataset = affs_dataset + '/s0'
        affs = open_ds(cfg.DATA.SAMPLE, affs_dataset)
    logging.info('Affs dataset has shape %s, ROI %s, voxel size %s' % (affs.shape, affs.roi, affs.voxel_size))

    # this is the starting level
    fragments_dataset = f"volumes/segmentation_{str(initial_epsilon_agglomerate).replace('.', '')}"
    cfg.INS_SEGMENT.OUT_FRAGS_DS = fragments_dataset
    logging.info("Reading fragments from %s", cfg.DATA.SAMPLE)
    try:
        fragments = open_ds(cfg.DATA.SAMPLE, fragments_dataset, mode="r")
    except Exception as e:
        raise Exception("Check if your file contains segmentations!!")

    # must be cast as gunpowder Coordinates
    voxel_size = Coordinate(cfg.MODEL.VOXEL_SIZE)

    # Fragments is grown with context
    total_input_roi = fragments.roi.grow(context, context)

    # For chunking, affs dimensions are used; ideally aff.roi == fragments.roi
    block_read_roi = Roi((0,) * affs.roi.dims, block_size).grow(context, context)
    block_write_roi = Roi((0,) * affs.roi.dims, block_size)

    fragments_task = daisy.Task(
        f"{sample_name}_fragments",
        total_roi=total_input_roi,
        read_roi=block_read_roi,
        write_roi=block_write_roi,
        process_function=lambda: start_worker(cfg,
                                              fragments_in_xy=fragments_in_xy,
                                              epsilon_agglomerate=initial_epsilon_agglomerate,
                                              mask_file=mask_file,
                                              mask_dataset=mask_dataset,
                                              filter_fragments=filter_fragments
                                              ),
        check_function=lambda b: check_block(completed_collection, b),
        num_workers=cfg.SYSTEM.NUM_WORKERS,
        read_write_conflict=False,
        fit="shrink",
    )

    daisy.run_blockwise([fragments_task])


def start_worker(
        cfg,
        fragments_in_xy,
        epsilon_agglomerate,
        mask_file,
        mask_dataset,
        filter_fragments,
        **kwargs):
    worker_id = daisy.Context.from_env()["worker_id"]
    task_id = daisy.Context.from_env()["task_id"]

    logging.info("worker %s started...", worker_id)

    output_basename = daisy.get_worker_log_basename(worker_id, task_id)

    logging.info('epsilon_agglomerate: %s', epsilon_agglomerate)
    logging.info('mask_file: %s', mask_file)
    logging.info('mask_dataset: %s', mask_dataset)
    logging.info('filter_fragments: %s', filter_fragments)

    log_out = output_basename.parent / f"worker_{worker_id}.out"
    log_err = output_basename.parent / f"worker_{worker_id}.err"

    config_file = os.path.join(f"{output_basename.parent}", f"config_{worker_id}.yaml")

    # update the config file
    cfg.INS_SEGMENT.FRAGMENTS_IN_XY = fragments_in_xy
    cfg.INS_SEGMENT.EPSILON_AGGLOMERATE = epsilon_agglomerate
    cfg.INS_SEGMENT.MASK_FILE = mask_file
    cfg.INS_SEGMENT.MASK_DATASET = mask_dataset
    cfg.INS_SEGMENT.FILTER_FRAGMENTS = filter_fragments
    waterz_merge_function = {0.10: 'hist_quant_10', 0.25: 'hist_quant_25', 0.35: 'hist_quant_35',
                             0.40: 'hist_quant_40', 0.45: 'hist_quant_45', 0.50: 'hist_quant_50',
                             0.55: 'hist_quant_55', 0.60: 'hist_quant_60', 0.65: 'hist_quant_65',
                             0.70: 'hist_quant_70', 0.75: 'hist_quant_75', 0.80: 'hist_quant_80',
                             0.85: 'hist_quant_85', 0.90: 'hist_quant_90', 0.95: 'hist_quant_95'}

    # we want to run agglomerate multiple times based on range of values
    agglomerate_until = np.array([round(x, 2) for x in cfg.INS_SEGMENT.THRESHOLDS])
    agglomerate_until = agglomerate_until[np.argwhere(agglomerate_until > round(epsilon_agglomerate, 2))].flatten()

    # for agglom_next in agglomerate_until:
    agglom_next = agglomerate_until[0]
    cfg.INS_SEGMENT.MERGE_FUNCTION = waterz_merge_function[agglom_next]
    logging.debug(f"Agglom merge function {cfg.INS_SEGMENT.MERGE_FUNCTION}")
    cfg.INS_SEGMENT.THRESHOLD = float(agglom_next)
    logging.info(f"cfg.INS_SEGMENT.THRESHOLD {cfg.INS_SEGMENT.THRESHOLD}")
    logging.info('Running block with config %s...' % config_file)

    with open(config_file, "w", encoding="utf-8") as f:
        f.write(cfg.dump())

    worker = "./engine/post/03_agglomerate_worker.py"

    subprocess.run(["python", worker, config_file])


def check_block(completed_collection, block):
    done = len(list(completed_collection.find({"block_id": block.block_id}))) >= 1

    return done


def rename_keys(original_config, key_mapping):
    for new_key, old_key in key_mapping.items():
        if hasattr(original_config, old_key):
            original_config[new_key] = getattr(original_config, old_key)

    return original_config


if __name__ == "__main__":

    cfg = get_cfg_defaults()
    # can be used to override pre-defined settings
    if os.path.exists("./experiment.yaml"):
        cfg.merge_from_file("experiment.yaml")

    # adding a copy of global model params to avoid if-else in train based on input data
    if cfg.DATA.FIB:
        key_mapping = {
            'MODEL': 'MODEL_ISO'
        }
    else:
        key_mapping = {
            'MODEL': 'MODEL_ANISO'
        }

    cfg = rename_keys(cfg, key_mapping)

    voxel_size = Coordinate(cfg.MODEL.VOXEL_SIZE)

    block_size = Coordinate(256, 256, 256) * voxel_size  # hardcoded
    context = Coordinate(16, 16, 16) * voxel_size
    # add the context to cfg:
    cfg.DATA.CONTEXT = tuple(context)
    # we want to run agglomerate multiple times based on range of values
    agglomerate_until = np.array([round(x, 2) for x in cfg.INS_SEGMENT.THRESHOLDS])
    agglomerate_until = agglomerate_until[
        np.argwhere(agglomerate_until > round(cfg.INS_SEGMENT.EPSILON_AGGLOMERATE, 2))].flatten()

    # for agglom_next in agglomerate_until:
    agglom_next = agglomerate_until[0]
    cfg.INS_SEGMENT.THRESHOLD = float(agglom_next)

    # do not freeze this because we want to add other options in the predict script
    # cfg.freeze()
    print(cfg)

    # make the outfile path here - /basepath/modeltype/2d/checkpoint_name
    # this should exist already given that instance segmentation will run on affinity predictions
    data_dir = os.path.join(cfg.DATA.OUTFILE, cfg.TRAIN.MODEL_TYPE,
                            '2d' if cfg.DATA.DIM_2D else '3d',
                            "/".join(cfg.TRAIN.CHECKPOINT.split("/")[-2:]))

    if not os.path.exists(data_dir):
        os.makedirs(os.path.dirname(data_dir), exist_ok=True)

    # TODO: add the logger here and import the same logging file
    # Follow: https://stackoverflow.com/questions/43947206/automatically-delete-old-python-log-files
    # logger.debug(f"data_dir {data_dir}")

    samples = glob(f"{data_dir}/*.zarr")

    assert len(samples), \
        "No data to run prediction on found. Check if data is placed under `{brain_vol}/data_{2/3d}/test`"
    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    # # we expect data going in at this point to be sequentially traversed one at a time.
    # # TODO: batch inference could make it faster
    if cfg.TRAIN.BATCH_SIZE > 1:
        cfg.TRAIN.BATCH_SIZE = 1

    if cfg.DATA.DIM_2D:
        # sample == .zarr
        for sample in samples:
            # .zarr --> volumes/raw/{0}.. volumes/raw/{n}
            num_samples = calculate_min_2d_samples([sample])
            assert num_samples, "Something went wrong.. num_samples cannot be zero," \
                                " it represents num of z-slices in the 2D zarrs."
            cfg.DATA.SAMPLE = sample
            # overwrite in the loop - otherwise will create zarr within zarr
            cfg.DATA.OUTFILE = data_dir
            cfg.DATA.OUTFILE = os.path.join(cfg.DATA.OUTFILE, os.path.basename(cfg.DATA.SAMPLE))

            """ This is a really bad implementation `Friday night blues`
             because the model is reloaded for each sample!!"""
            for n in range(num_samples):
                cfg.DATA.SAMPLE_SLICE = n

    else:
        # we loop through the datasets here and call inference on them.
        # Todo: add daisy support for spawning async for every dataset
        for sample in samples:
            cfg.DATA.SAMPLE = sample
            # overwrite in the loop - otherwise will create zarr within zarr
            cfg.DATA.OUTFILE = data_dir
            cfg.DATA.OUTFILE = os.path.join(cfg.DATA.OUTFILE, os.path.basename(cfg.DATA.SAMPLE))

            start = time.time()

            sample_name = os.path.basename(sample).split('.')[0]
            sample_name = sub(r"(_|-)+", " ", sample_name).title().replace(" ", "")
            sample_name = ''.join([sample_name[0].lower(), sample_name[1:]])
            db_host = "localhost:27017" if cfg.DATA.DB_HOST == '' else cfg.DATA.DB_HOST  # default
            db_name = "lsd_parallel_fragments" if cfg.DATA.DB_NAME == '' else cfg.DATA.DB_NAME  # default
            cfg.DATA.SAMPLE_NAME = sample_name

            agglomerate(cfg,
                        block_size=block_size,
                        context=context,
                        sample_name=sample_name,
                        fragments_in_xy=cfg.INS_SEGMENT.FRAGMENTS_IN_XY,
                        # set these as argparse/config file params
                        initial_epsilon_agglomerate=cfg.INS_SEGMENT.EPSILON_AGGLOMERATE,
                        agglomerate_until=cfg.INS_SEGMENT.THRESHOLD,
                        mask_file=None if cfg.INS_SEGMENT.MASK_FILE == '' else cfg.INS_SEGMENT.MASK_FILE,
                        mask_dataset=None if cfg.INS_SEGMENT.MASK_DATASET == '' else cfg.INS_SEGMENT.MASK_DATASET,
                        filter_fragments=cfg.INS_SEGMENT.FILTER_FRAGMENTS,
                        drop=False)

            end = time.time()

            seconds = end - start
            minutes = seconds / 60
            hours = minutes / 60
            days = hours / 24

            print('Total time to extract fragments: %f seconds / %f minutes / %f hours / %f days' % (
                seconds, minutes, hours, days))
