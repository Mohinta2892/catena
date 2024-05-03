"""Post-processing, this does not work inside the docker, so run outside with the conda lsd env in cardona-gpu1
"""
import numpy as np
from data_utils.preprocess_volumes.utils import read_zarr
from .watershed_helpers import get_segmentation
from tqdm import tqdm
import logging
from funlib.persistence import Array, graphs, open_ds
from funlib.geometry import Roi, Coordinate
from scipy.ndimage import measurements

logger = logging.getLogger(__name__)


def run_waterz(cfg):
    results_zarr = read_zarr(cfg.DATA.OUTFILE, mode="r+")

    for th in tqdm(cfg.INS_SEGMENT.THRESHOLDS):
        threshold = round(th, 2)
        print(f"Running waterz with threshold {threshold}")

        # path to predicted affinity matrix is hard-coded 
        results = np.array(results_zarr[f"volumes/pred_affs"])
        if results.dtype == np.uint8:
            results = (results / 255).astype(np.float32)

        segmentation = get_segmentation(results, threshold)

        voxel_size = cfg.MODEL.VOXEL_SIZE
        # raw = ArrayKey('RAW')
        # data_sources, raw_roi = read_source(raw, raw_file, raw_dataset)
        # total_roi = raw_roi.get_shape()

        results_zarr[f"volumes/segmentation_{str(threshold).replace('.', '')}"] = segmentation
        results_zarr[f"volumes/segmentation_{str(threshold).replace('.', '')}"].attrs["offset"] = (
            0, 0, 0)  # total_roi.get_offset()
        results_zarr[f"volumes/segmentation_{str(threshold).replace('.', '')}"].attrs["resolution"] = voxel_size

        print(f"-----------------------")


def run_waterz_parallel(affs,
                        block,
                        context,
                        rag_provider,
                        fragments_out,
                        num_voxels_in_block,
                        mask,
                        fragments_in_xy,
                        epsilon_agglomerate,
                        filter_fragments
                        ):
    total_roi = affs.roi  # this is used in replace sections

    logger.debug("reading affs from %s", block.read_roi)

    affs = affs.intersect(block.read_roi)
    affs.materialize()

    if affs.dtype == np.uint8:
        logger.info("Assuming affinities are in [0,255]")
        max_affinity_value = 255.0
        affs.data = affs.data.astype(np.float32)
    else:
        max_affinity_value = 1.0

    if mask is not None:
        logger.debug("reading mask from %s", block.read_roi)
        mask_data = get_mask_data_in_roi(mask, affs.roi, affs.voxel_size)
        logger.debug("masking affinities")
        affs.data *= mask_data

    # Get segmentation here, remember this is both watershed and agglomerated
    # hence we will bypass for now all the post-processing steps on fragments (filtering)
    # such as 
    fragments_data = get_segmentation(affs.data, threshold=epsilon_agglomerate,
                                      max_affinity_value=max_affinity_value)

    if mask is not None:
        fragments_data *= mask_data.astype(np.uint64)

    fragments = Array(fragments_data, affs.roi, affs.voxel_size)

    # crop fragments to write_roi
    fragments = fragments[block.write_roi]
    fragments.materialize()
    max_id = fragments.data.max()

    # ensure we don't have IDs larger than the number of voxels (that would
    # break uniqueness of IDs below)
    if max_id > num_voxels_in_block:
        logger.warning(
            "fragments in %s have max ID %d, relabelling...",
            block.write_roi, max_id)
        fragments.data, max_id = relabel(fragments.data)

        assert max_id < num_voxels_in_block

    # ensure unique IDs, now daisy sets block_ids as tuples (name, id)
    id_bump = block.block_id[1] * num_voxels_in_block
    logger.debug("bumping fragment IDs by %i", id_bump)
    fragments.data[fragments.data > 0] += id_bump
    fragment_ids = range(id_bump + 1, id_bump + 1 + int(max_id))

    # store fragments
    logger.debug("writing fragments to %s", block.write_roi)
    fragments_out[block.write_roi] = fragments

    # following only makes a difference if fragments were found
    if max_id == 0:
        return

    # get fragment centers
    fragment_centers = {
        fragment: block.write_roi.get_offset() + affs.voxel_size * Coordinate(center)
        for fragment, center in zip(
            fragment_ids,
            measurements.center_of_mass(fragments.data, fragments.data, fragment_ids))
        if not np.isnan(center[0])
    }

    # store nodes
    rag = rag_provider[block.write_roi]
    rag.add_nodes_from([
        (node, {
            'center_z': c[0],
            'center_y': c[1],
            'center_x': c[2]
        }
         )
        for node, c in fragment_centers.items()
    ])
    rag.write_nodes(block.write_roi)
