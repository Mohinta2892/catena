import daisy
import logging
# from lsd.post.parallel_fragments import watershed_in_block # has incorrect imports
import json
import sys
import pymongo
import time
from funlib.persistence import open_ds, Array, graphs
from funlib.geometry import Roi, Coordinate
from yacs.config import CfgNode as CN

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from engine.post.parallel_fragments import watershed_in_block
from engine.post.run_waterz import run_waterz_parallel

logging.basicConfig(level=logging.INFO)


def extract_fragments_worker(cfg):
    affs_file = cfg.DATA.SAMPLE
    affs_dataset = cfg.INS_SEGMENT.AFFS_DS  # is always this
    fragments_file = cfg.DATA.SAMPLE
    fragments_dataset = cfg.INS_SEGMENT.OUT_FRAGS_DS
    db_name = cfg.DATA.DB_NAME
    db_host = cfg.DATA.DB_HOST
    num_voxels_in_block = cfg.INS_SEGMENT.NUM_VOXELS_IN_BLOCK
    fragments_in_xy = cfg.INS_SEGMENT.FRAGMENTS_IN_XY
    epsilon_agglomerate = cfg.INS_SEGMENT.EPSILON_AGGLOMERATE
    filter_fragments = cfg.INS_SEGMENT.FILTER_FRAGMENTS
    context = Coordinate(cfg.DATA.CONTEXT)
    sample_name = cfg.DATA.SAMPLE_NAME

    logging.info("Reading affs from %s", affs_file)
    affs = open_ds(affs_file, affs_dataset, mode='r')

    logging.info("Reading fragments from %s", fragments_file)
    fragments = open_ds(
        fragments_file,
        fragments_dataset,
        mode='r+')

    if cfg.INS_SEGMENT.MASK_FILE is not None:

        logging.info("Reading mask from %s", config['mask_file'])
        mask = open_ds(
            cfg.INS_SEGMENT.MASK_FILE,
            cfg.INS_SEGMENT.MASK_DATASET,
            mode='r')

    else:

        mask = None

    # open RAG DB
    logging.info("Opening RAG DB...")
    rag_provider = graphs.MongoDbGraphProvider(
        db_name,
        host=db_host,
        mode='r+',
        directed=False,
        position_attribute=['center_z', 'center_y', 'center_x'],
        edges_collection=f"{sample_name}_edges",
        nodes_collection=f"{sample_name}_nodes",
        meta_collection=f"{sample_name}_meta",
    )
    logging.info("RAG DB opened")

    # open block done DB
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    blocks_extracted = db[cfg.DATA.DB_COLLECTION_NAME]

    client = daisy.Client()

    while True:

        with client.acquire_block() as block:

            if block is None:
                break

            start = time.time()

            logging.info("block read roi begin: %s", block.read_roi.get_begin())
            logging.info("block read roi shape: %s", block.read_roi.get_shape())
            logging.info("block write roi begin: %s", block.write_roi.get_begin())
            logging.info("block write roi shape: %s", block.write_roi.get_shape())

            # This performs watershed and an initial agglomeration chunk-wise
            watershed_in_block(
                affs=affs,
                block=block,
                context=context,
                rag_provider=rag_provider,
                fragments_out=fragments,
                num_voxels_in_block=num_voxels_in_block,
                mask=mask,
                fragments_in_xy=fragments_in_xy,
                epsilon_agglomerate=epsilon_agglomerate,
                filter_fragments=filter_fragments)

            # This performs watershed_from_affinities and agglomerate sequentially.
            # Should ideally do the same thing as above. Kept for comparison though!
            # run_waterz_parallel(affs=affs,
            #                     block=block,
            #                     context=context,
            #                     rag_provider=rag_provider,
            #                     fragments_out=fragments,
            #                     num_voxels_in_block=num_voxels_in_block,
            #                     mask=mask,
            #                     fragments_in_xy=fragments_in_xy,
            #                     epsilon_agglomerate=epsilon_agglomerate,
            #                     filter_fragments=filter_fragments
            #                     )

            document = {
                'num_cpus': cfg.SYSTEM.NUM_WORKERS,
                'block_id': block.block_id,
                'read_roi': (
                    block.read_roi.get_begin(),
                    block.read_roi.get_shape()
                ),
                'write_roi': (
                    block.write_roi.get_begin(),
                    block.write_roi.get_shape()
                ),
                'start': start,
                'duration': time.time() - start
            }
            blocks_extracted.insert_one(document)

            logging.info(f"releasing block: {block}")


if __name__ == '__main__':
    config_file = sys.argv[1]

    # parse the args file to become cfg
    cfg = CN()
    # Allow creating new keys recursively.: https://github.com/rbgirshick/yacs/issues/25
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file)
    extract_fragments_worker(cfg)
