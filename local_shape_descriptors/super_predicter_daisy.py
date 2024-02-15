from __future__ import annotations
# # Might have to use this since :
# # RuntimeError: Cannot re-initialize CUDA in forked subprocess.
# # To use CUDA with multiprocessing, you must use the 'spawn' start method
# import torch
# torch.multiprocessing.set_start_method('spawn')
# multiprocessing.set_start_method('spawn')
import hashlib
import json
import logging
import numpy as np
import os
import daisy
import sys
import time
import datetime
import pymongo
from funlib.persistence import open_ds
from funlib.geometry import Roi, Coordinate

# add current directory to path and allow absolute imports
sys.path.insert(0, '.')
from config.config_predict import *
from data_utils.preprocess_volumes.utils import calculate_min_2d_samples
from glob import glob
from add_ons.funlib_persistence.persistence_utils import *
from gunpowder import *
from contextlib import redirect_stdout  # a hack to dump cfg as yaml; https://github.com/rbgirshick/yacs/issues/31
import subprocess

logging.basicConfig(level=logging.INFO)

logging.getLogger('daisy').setLevel(logging.DEBUG)
module_logger = logging.getLogger(__name__)



def predict_blockwise(
        cfg,
        sample_name='sample',
        db_host="localhost:27017",
        db_name="try-lsd-parallel-pred",
        drop=False
):
    """
    All roi related code here!
    :return:
    """
    # Add dbname and dbhost to cfg
    cfg.DATA.DB_NAME = db_name
    cfg.DATA.DB_HOST = db_host

    # Mongo-related stuff
    client = pymongo.MongoClient(db_host)
    db = client[db_name]

    completed_collection_name = f"{sample_name}_predicted_affs"
    # Save to config for worker to use
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

    # get ROI of source
    raw_dataset = "volumes/raw"  # is always this
    try:
        source = open_ds(cfg.DATA.SAMPLE, raw_dataset)
    except:
        # perhaps for a n5
        # TODO: readjust later
        raw_dataset = raw_dataset + '/s0'
        source = daisy.open_ds(raw_file, raw_dataset)
    logging.info('Source dataset has shape %s, ROI %s, voxel size %s' % (source.shape, source.roi, source.voxel_size))

    # must be cast as gunpowder Coordinates
    voxel_size = Coordinate(cfg.MODEL.VOXEL_SIZE)
    input_shape = Coordinate(cfg.MODEL.INPUT_SHAPE) + Coordinate(cfg.MODEL.GROW_INPUT)
    output_shape = Coordinate(cfg.MODEL.OUTPUT_SHAPE) + Coordinate(cfg.MODEL.GROW_INPUT)
    net_input_size = input_shape * voxel_size  # this was input_size in predict.py
    net_output_size = output_shape * voxel_size
    # context added/removed
    context = (net_input_size - net_output_size) / 2
    module_logger.debug(f"input_size: {net_input_size}; output_size: {net_output_size}")

    source_roi = source.roi
    total_output_roi = source_roi.grow(-context, -context)
    module_logger.debug(f"Total output ROI {total_output_roi} and context {context}")

    # create read and write ROI
    block_read_roi = Roi((0, 0, 0), net_input_size) - context
    block_write_roi = Roi((0, 0, 0), net_output_size)

    logging.info('Preparing output dataset...')

    # Creating datasets in output zarr
    # Hard-code warning: the ds keys in the out-zarr are hardcoded for now, hence will ensure same output format
    # Todo: move to config to allow customisation of ds keys
    out_raw = "volumes/raw"
    prepare_predict_datasets_daisy(cfg, dtype=np.uint8, voxel_size=voxel_size, ds_key=out_raw, source_roi=source.roi,
                                   write_roi=block_write_roi,
                                   delete_ds=drop)
    print(out_raw)

    if cfg.TRAIN.MODEL_TYPE in ["MTLSD", "LSD"]:
        out_lsds = "volumes/pred_lsds"
        prepare_predict_datasets_daisy(cfg, dtype=np.float32, ds_key=out_lsds,
                                       source_roi=source.roi, num_channels=10,
                                       write_roi=block_write_roi,
                                       voxel_size=voxel_size,
                                       delete_ds=drop)  # shape C(10) x D x H x W

    if cfg.TRAIN.MODEL_TYPE in ["MTLSD", "AFF"]:
        out_affs = "volumes/pred_affs"
        prepare_predict_datasets_daisy(cfg, dtype=np.float32,
                                       ds_key=out_affs,
                                       source_roi=source.roi,
                                       num_channels=len(
                                           cfg.TRAIN.NEIGHBORHOOD),
                                       write_roi=block_write_roi,
                                       voxel_size=voxel_size, delete_ds=drop)

        if cfg.DATA.INVERT_PRED_AFFS and cfg.TRAIN.MODEL_TYPE in ["MTLSD", "AFF"]:
            # this will choose a max affinity channel, cast to uint8 and invert
            out_inv_affs = "volumes/inverted_pred_affs"
            prepare_predict_datasets_daisy(cfg, dtype=np.uint8, ds_key=out_inv_affs,
                                           source_roi=source.roi,
                                           num_channels=1,
                                           write_roi=block_write_roi,
                                           voxel_size=voxel_size, delete_ds=drop)

    predict_affs_task = daisy.Task(
        f"{sample_name}_pred_affs",
        total_roi=total_output_roi,
        read_roi=block_read_roi,
        write_roi=block_write_roi,
        process_function=lambda: start_worker(cfg),
        check_function=lambda b: check_block(completed_collection, complete_cache, b),
        num_workers=cfg.SYSTEM.NUM_WORKERS,
        read_write_conflict=False,
        fit="shrink",
    )

    daisy.run_blockwise([predict_affs_task])


def check_block(completed_collection, complete_cache, block):
    done = (
            block.block_id in complete_cache
            or len(list(completed_collection.find({"block_id": block.block_id}))) >= 1
    )

    return done


def start_worker(cfg):
    """
    This calls the python predict.py
    :return:
    """
    worker_id = daisy.Context.from_env()["worker_id"]
    task_id = daisy.Context.from_env()["task_id"]

    logging.info("worker %s started...", worker_id)
    output_basename = daisy.get_worker_log_basename(worker_id, task_id)

    log_out = output_basename.parent / f"worker_{worker_id}.out"
    log_err = output_basename.parent / f"worker_{worker_id}.err"
    # 
    # config_str = "".join(["%s" % (v,) for v in config.values()])
    # config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

    config_file = os.path.join(f"{output_basename.parent}", f"config_{worker_id}.yaml")
  
    # Based on worker id, reset the cuda device: experimental.
    # This works when number of workers = 1, throws 
    # `RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method`
    # when using > 1 num_workers!
    # cfg.TRAIN.DEVICE = f"cuda:{worker_id}"

    # print(cfg.dump())  # print formatted configs
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(cfg.dump())

    logging.info("Running block with config %s..." % config_file)

    # abs path to the worker - make it relative
    worker = "./engine/predict/predict_worker_daisy.py"

    subprocess.run(
        ["python", f"{worker}", f"{config_file}"]
    )


def rename_keys(original_config, key_mapping):
    for new_key, old_key in key_mapping.items():
        if hasattr(original_config, old_key):
            original_config[new_key] = getattr(original_config, old_key)

    return original_config


if __name__ == '__main__':



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
    # do not freeze this because we want to add other options in the predict script
    # cfg.freeze()
    print(cfg)

    data_dir = os.path.join(cfg.DATA.HOME, cfg.DATA.DATA_DIR_PATH, cfg.DATA.BRAIN_VOL)
    # data is expected to be here
    if cfg.DATA.DIM_2D:
        data_dir = os.path.join(data_dir, 'data_2d', 'test')
    else:
        data_dir = os.path.join(data_dir, 'data_3d', 'test')

    # TODO: add the logger here and import the same logging file
    # Follow: https://stackoverflow.com/questions/43947206/automatically-delete-old-python-log-files
    # logger.debug(f"data_dir {data_dir}")

    samples = glob(f"{data_dir}/*.zarr")

    assert len(samples), \
        "No data to run prediction on found. Check if data is placed under `{brain_vol}/data_{2/3d}/test`"
    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    # make the outfile path here - /basepath/modeltype/2d/checkpoint_name
    out_filepath = os.path.join(cfg.DATA.OUTFILE, cfg.TRAIN.MODEL_TYPE,
                                '2d' if cfg.DATA.DIM_2D else '3d',
                                "/".join(cfg.TRAIN.CHECKPOINT.split("/")[-2:]))
    if not os.path.exists(out_filepath):
        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

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
            cfg.DATA.OUTFILE = out_filepath
            cfg.DATA.OUTFILE = os.path.join(cfg.DATA.OUTFILE, os.path.basename(cfg.DATA.SAMPLE))

            """ This is a really bad implementation `Friday night blues`
             because the model is reloaded for each sample!!"""
            for n in range(num_samples):
                cfg.DATA.SAMPLE_SLICE = n

    else:
        # we loop through the datasets here and call inference on them.
        # Todo: add daisy support for spawning
        for sample in samples:
            cfg.DATA.SAMPLE = sample
            # overwrite in the loop - otherwise will create zarr within zarr
            cfg.DATA.OUTFILE = out_filepath
            cfg.DATA.OUTFILE = os.path.join(cfg.DATA.OUTFILE, os.path.basename(cfg.DATA.SAMPLE))

    start = time.time()

    sample_name = "test"
    db_host = "localhost:27017"
    db_name = "lsd_predictions_parallel"
    predict_blockwise(
        cfg, sample_name=sample_name, db_host=db_host, db_name=db_name
    )

    end = time.time()

    seconds = end - start
    minutes = seconds / 60
    hours = minutes / 60
    days = hours / 24

    print(
        "Total time to extract fragments: %f seconds / %f minutes / %f hours / %f days"
        % (seconds, minutes, hours, days)
    )
