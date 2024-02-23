import daisy
import json
import logging
import numpy as np
import os
import sys
import time
from funlib.segment.arrays import replace_values
from funlib.persistence import prepare_ds, open_ds
from funlib.geometry import Roi, Coordinate
from yacs.config import CfgNode as CN

logging.basicConfig(level=logging.INFO)


def extract_segmentation(
        sample_name,
        fragments_file,
        fragments_dataset,
        edges_collection,
        threshold,
        block_size,
        out_file,
        out_dataset,
        num_workers,
        roi_offset=None,
        roi_shape=None,
        run_type=None,
        **kwargs):
    '''

    Args:

        fragments_file (``string``):

            Path to file (zarr/n5) containing fragments (supervoxels).

        fragments_dataset (``string``):

            Name of fragments dataset (e.g `volumes/fragments`)

        edges_collection (``string``):

            The name of the MongoDB database edges collection to use.

        threshold (``float``):

            The threshold to use for generating a segmentation.

        block_size (``tuple`` of ``int``):

            The size of one block in world units (must be multiple of voxel
            size).

        out_file (``string``):

            Path to file (zarr/n5) to write segmentation to.

        out_dataset (``string``):

            Name of segmentation dataset (e.g `volumes/segmentation`).

        num_workers (``int``):

            How many workers to use when reading the region adjacency graph
            blockwise.

        roi_offset (array-like of ``int``, optional):

            The starting point (inclusive) of the ROI. Entries can be ``None``
            to indicate unboundedness.

        roi_shape (array-like of ``int``, optional):

            The shape of the ROI. Entries can be ``None`` to indicate
            unboundedness.

        run_type (``string``, optional):

            Can be used to direct luts into directory (e.g testing, validation,
            etc).

    '''

    # open fragments
    fragments = open_ds(fragments_file, fragments_dataset)

    total_roi = fragments.roi
    if roi_offset is not None:
        assert roi_shape is not None, "If roi_offset is set, roi_shape " \
                                      "also needs to be provided"
        total_roi = daisy.Roi(offset=roi_offset, shape=roi_shape)

    read_roi = Roi((0,) * 3, daisy.Coordinate(block_size))
    write_roi = read_roi

    logging.info("Preparing segmentation dataset...")
    segmentation = prepare_ds(
        out_file,  # this should be the zarr!
        out_dataset,
        total_roi,
        voxel_size=fragments.voxel_size,
        dtype=np.uint64,
        write_roi=write_roi)

    lut_dir = os.path.join(os.path.dirname(fragments_file), "luts_full")
    logging.debug(f"Luts should have been saved here {lut_dir}")

    lookup = 'seg_%s_%d' % (edges_collection, int(threshold * 100))
    lut_file = os.path.join(lut_dir, f"{lookup}.npz")
    assert os.path.exists(lut_file), f"{lut_file} does not exist"
    logging.info(f"Picking lut_file: {lut_file}")

    logging.info("Reading fragment-segment LUT...")

    lut = np.load(lut_file)['fragment_segment_lut']

    logging.info(f"Found {len(lut[0])} fragments in LUT")

    num_segments = len(np.unique(lut[1]))
    logging.info(f"Relabelling fragments to {num_segments} segments")

    extract_segmentation = daisy.Task(
        f"{sample_name}_extract_seg",
        total_roi,
        read_roi,
        write_roi,
        lambda b: segment_in_block(
            b,
            fragments_file,
            segmentation,
            fragments,
            lut),
        fit='shrink',
        num_workers=num_workers)
    daisy.run_blockwise([extract_segmentation])


def segment_in_block(
        block,
        fragments_file,
        segmentation,
        fragments,
        lut):
    logging.info("Copying fragments to memory...")

    # load fragments
    fragments = fragments.to_ndarray(block.write_roi)

    # replace values, write to empty array
    relabelled = np.zeros_like(fragments)
    relabelled = replace_values(fragments, lut[0], lut[1], out_array=relabelled)

    segmentation[block.write_roi] = relabelled


if __name__ == "__main__":
    config_file = sys.argv[1]  # take one of the worker config files from daisy_logs??

    # parse the args file to become cfg
    cfg = CN()
    # Allow creating new keys recursively.: https://github.com/rbgirshick/yacs/issues/25
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file)

    start = time.time()

    sample_name = cfg.DATA.SAMPLE_NAME
    db_host = cfg.DATA.DB_HOST
    db_name = cfg.DATA.DB_NAME
    merge_function = 'hist_quant_55'

    extract_segmentation(
        sample_name=sample_name,
        fragments_file=cfg.DATA.SAMPLE,
        fragments_dataset=cfg.INS_SEGMENT.OUT_FRAGS_DS,
        edges_collection=sample_name + "_edges_" + merge_function,
        threshold=0.0,
        block_size=Coordinate(
            (512, 512, 512)),  # A random large number?? donno what to put
        out_file=cfg.DATA.SAMPLE,
        out_dataset='volumes/final_segmentation',
        num_workers=cfg.SYSTEM.NUM_WORKERS,
        roi_offset=None,
        roi_shape=None,
        run_type=None, )

    logging.info(f"Took {time.time() - start} seconds to extract segmentation from LUT")
