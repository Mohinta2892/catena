"""
We are going to try to do this as in the old scripts, without calling parallel fragments.
Todo: Switch to this later since we do not have to make the task then!
This script performs watershed and an initial agglomeration (as set in `EPSILON_THRESHOLD`
in `config_predict.py`) on a chunk by chunk basis.
NB: Hardcoded `block_size` appear in `main()`. TODO: control them via config_predict.py.
NB 2: Re. visualising fragments (chunk segmentations), napari may not be the best visualiser unless passed
an explicit color map in the range of the segmentation values. This is because it displays really large
values as a gradient colormap (e.g., in shades of green). TODO: Try neuroglancer visualisation.
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
logging.getLogger('lsd.parallel_fragments').setLevel(logging.DEBUG)
module_logger = logging.getLogger(__name__)


def extract_fragments(
        cfg,
        block_size,
        context,
        sample_name,
        fragments_in_xy=False,
        epsilon_agglomerate=0,
        mask_file=None,
        mask_dataset=None,
        filter_fragments=0,
        drop=False,
        **kwargs):
    """
    Extract fragments from a given sample based on the provided configuration.

    Args:
    cfg (YACS config): Configuration dictionary containing parameters for fragment extraction.

    block_size (`funlib.geometry.Coordinate`): Size of the blocks to be extracted.

    context (int): Context parameter.
    sample_name (str): Name of the sample from which fragments are to be extracted.
    fragments_in_xy (bool, optional): Whether the fragments are in xy plane. Defaults to False.
    epsilon_agglomerate (int, optional): Epsilon parameter for agglomerating fragments. Defaults to 0.
    mask_file (str, optional): Path to the mask file. Defaults to None.
    mask_dataset (str, optional): Dataset to be used for masking. Defaults to None.
    filter_fragments (int, optional): Filter parameter for fragments. Defaults to 0.
    drop (bool, optional): Whether to drop fragments. Defaults to False.
    **kwargs: Additional keyword arguments.

    Returns:
    Extracted fragments based on the provided parameters.

"""

    # Add dbname and dbhost to cfg
    cfg.DATA.DB_NAME = db_name
    cfg.DATA.DB_HOST = db_host

    # network_dir = os.path.join(experiment, setup, str(iteration))

    # copied from WPatton's mutex_agglomerate git repo
    client = pymongo.MongoClient(db_host)
    db = client[db_name]

    completed_collection_name = f"{sample_name}_fragment_blocks_extracted_{str(epsilon_agglomerate).replace('.', '')}"
    cfg.DATA.DB_COLLECTION_NAME = completed_collection_name
    completed_collection = None

    if completed_collection_name in db.list_collection_names():
        completed_collection = db[completed_collection_name]
        if drop:
            print(f"dropping {completed_collection}")
            db.drop_collection(completed_collection)
    if f"{sample_name}_nodes" in db.list_collection_names():
        nodes_collection = db[f"{sample_name}_nodes"]
        if drop:
            print(f"dropping {nodes_collection}")
            db.drop_collection(nodes_collection)
    if f"{sample_name}_meta" in db.list_collection_names():
        meta_collection = db[f"{sample_name}_meta"]
        if drop:
            print(f"dropping {meta_collection}")
            db.drop_collection(meta_collection)
    for collection_name in db.list_collection_names():
        if collection_name.startswith(f"{sample_name}_edges"):
            edges_collection = db[collection_name]
            if drop:
                print(f"dropping {edges_collection}")
                db.drop_collection(edges_collection)

    if completed_collection_name not in db.list_collection_names():
        completed_collection = db[completed_collection_name]
        completed_collection.create_index(
            [("block_id", pymongo.ASCENDING)], name="block_id"
        )

    complete_cache = set(
        [tuple(doc["block_id"]) for doc in completed_collection.find()]
    )
    # total_roi = daisy.Roi((0, 0, 67200), (900000, 285600, 403200))
    # total_roi = daisy.Roi((459960, 92120, 217952), (80040, 75880, 62048))
    # total_roi = daisy.Roi((200580, 62440, 222824), (58260, 35056, 43512))

    logging.info("Reading affs from %s", cfg.DATA.SAMPLE)
    affs_dataset = "volumes/pred_affs"  # is always this
    cfg.INS_SEGMENT.AFFS_DS = affs_dataset  # add here to be reused in worker
    try:
        affs = open_ds(cfg.DATA.SAMPLE, affs_dataset, mode='r')
    except Exception as e:
        # perhaps for a n5
        # TODO: readjust later
        print(e)
        affs_dataset = affs_dataset + '/s0'
        affs = open_ds(cfg.DATA.SAMPLE, affs_dataset)
    logging.info('Source dataset has shape %s, ROI %s, voxel size %s' % (affs.shape, affs.roi, affs.voxel_size))

    # must be cast as gunpowder Coordinates
    voxel_size = Coordinate(cfg.MODEL.VOXEL_SIZE)

    # THIS IS WRONG!! MAKE A NOTE OF IT IN DOCS FOR TROUBLE SHOOTING/TIPS/DESIGN SPECS.
    # input_shape = Coordinate(cfg.MODEL.INPUT_SHAPE) + Coordinate(cfg.MODEL.GROW_INPUT)
    # output_shape = Coordinate(cfg.MODEL.OUTPUT_SHAPE) + Coordinate(cfg.MODEL.GROW_INPUT)
    # net_input_size = input_shape * voxel_size  # this was input_size in predict.py
    # net_output_size = output_shape * voxel_size
    # # context added/removed
    # context = (net_input_size - net_output_size) / 2
    # logging.debug(f"input_size: {net_input_size}; output_size: {net_output_size}")
    # cfg.DATA.CONTEXT = tuple(context)
    #
    # source_roi = affs.roi
    # total_output_roi = source_roi.grow(context, context)  # first change positive context
    # logging.debug(f"Total output ROI {total_output_roi} and context {context}")
    # create read and write ROI
    # block_read_roi = Roi((0, 0, 0), net_input_size)  # - context
    # block_write_roi = Roi((0, 0, 0), net_output_size)

    total_input_roi = affs.roi.grow(context, context)
    block_read_roi = Roi((0,) * affs.roi.dims, block_size).grow(context, context)
    block_write_roi = Roi((0,) * affs.roi.dims, block_size)

    num_voxels_in_block = (block_write_roi / affs.voxel_size).size

    logging.info('Preparing output dataset...')
    # saves the seg datasets as `volumes/segmentation_055`
    out_frags = f"volumes/segmentation_{str(epsilon_agglomerate).replace('.', '')}"
    cfg.INS_SEGMENT.OUT_FRAGS_DS = out_frags
    prepare_predict_datasets_daisy(
        cfg, dtype=np.uint64, ds_key=out_frags, source_roi=affs.roi, voxel_size=affs.voxel_size,
        write_roi=block_write_roi, num_channels=None,
        delete_ds=drop)

    fragments_task = daisy.Task(
        f"{sample_name}_fragments",
        total_roi=total_input_roi,
        read_roi=block_read_roi,
        write_roi=block_write_roi,
        process_function=lambda: start_worker(cfg,
                                              fragments_in_xy=fragments_in_xy,
                                              epsilon_agglomerate=epsilon_agglomerate,
                                              mask_file=mask_file,
                                              mask_dataset=mask_dataset,
                                              filter_fragments=filter_fragments,
                                              num_voxels_in_block=num_voxels_in_block),
        check_function=lambda b: check_block(completed_collection, complete_cache, b),
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
        num_voxels_in_block,
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
    cfg.INS_SEGMENT.NUM_VOXELS_IN_BLOCK = num_voxels_in_block

    logging.info('Running block with config %s...' % config_file)

    with open(config_file, "w", encoding="utf-8") as f:
        f.write(cfg.dump())

    worker = "./engine/post/02_extract_fragments_worker.py"

    # IF error daisy is not installed/ TypeError: 'type' object is not subscriptable due to  attr_filter: Optional[dict[str, Any]]
    # in mongodb graph provider, it might be due to suybprocess below pointing to a different python, not the python you have in your conda
    # env. A workaround -replace python with sys.executable: https://stackoverflow.com/questions/51819719/using-subprocess-in-anaconda-environment
    subprocess.run(["python", worker, config_file])


def check_block(completed_collection, complete_cache, block):
    done = (
            block.block_id in complete_cache
            or len(list(completed_collection.find({"block_id": block.block_id}))) >= 1
    )

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

    block_size = Coordinate(cfg.INS_SEGMENT.BLOCK_SIZE) * voxel_size  # Set in config
    context = Coordinate(cfg.INS_SEGMENT.CONTEXT) * voxel_size
    # add the context to cfg:
    cfg.DATA.CONTEXT = tuple(context)
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
            cfg.DATA.SAMPLE_NAME = sample_name
            db_host = "localhost:27017" if cfg.DATA.DB_HOST == '' else cfg.DATA.DB_HOST  # default
            db_name = "lsd_parallel_fragments" if cfg.DATA.DB_NAME == '' else cfg.DATA.DB_NAME  # default

            extract_fragments(cfg,
                              block_size=block_size,
                              context=context,
                              sample_name=sample_name,
                              fragments_in_xy=cfg.INS_SEGMENT.FRAGMENTS_IN_XY,
                              # set these as argparse/config file params
                              epsilon_agglomerate=cfg.INS_SEGMENT.EPSILON_AGGLOMERATE,
                              mask_file=None if cfg.INS_SEGMENT.MASK_FILE == '' else cfg.INS_SEGMENT.MASK_FILE,
                              mask_dataset=None if cfg.INS_SEGMENT.MASK_DATASET == '' else cfg.INS_SEGMENT.MASK_DATASET,
                              filter_fragments=cfg.INS_SEGMENT.FILTER_FRAGMENTS,
                              drop=cfg.DATA.DROP_DS_MONGOTABLE)  # default is False

            end = time.time()

            seconds = end - start
            minutes = seconds / 60
            hours = minutes / 60
            days = hours / 24

            print('Total time to extract fragments: %f seconds / %f minutes / %f hours / %f days' % (
                seconds, minutes, hours, days))
