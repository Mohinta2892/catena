from __future__ import annotations

import glob
import sys
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import argparse
import logging

# add current directory to path and allow absolute imports
sys.path.insert(0, '.')
from data_utils.preprocess_volumes.utils import create_data, find_and_interactively_delete_zero_levels
from data_utils.preprocess_volumes.histogram_match import match_histograms
from config.config import get_cfg_defaults

logger = logging.getLogger(__name__)


def create_data_in_parallel(z, out_dir, cfg, sections=None, squeeze=True):
    out_zarr = create_data(
        url_or_path=z,
        outfile_path=out_dir,
        offset=cfg.PREPROCESS.SOURCE_DATA_OFFSET,
        resolution=cfg.PREPROCESS.SOURCE_DATA_RESOLUTION,
        sections=sections,
        squeeze=squeeze,
        background_value=cfg.DATA.BACKGROUND_LABEL,
        # for mtlsdmito; neuron_ids are needed!
        is_mito=cfg.DATA.WITH_MITO
    )


def main(args):
    cfg = get_cfg_defaults()
    # can be used to override pre-defined settings
    # TODO test with explicit path for example setups: ssTEM CREMI, FIBSEM: Hemibrain
    if os.path.exists("./experiment.yaml"):
        cfg.merge_from_file("experiment.yaml")

    if args.background_value is None:
        dataset_name = cfg.DATA.BRAIN_VOL.lower()
        if dataset_name == 'hemi':
            cfg.DATA.BACKGROUND_LABEL = int(np.array(-3).astype(np.uint64))
        else:
            cfg.DATA.BACKGROUND_LABEL = 0
    else:
        cfg.DATA.BACKGROUND_LABEL = int(background_value)

    if cfg.DATA.DIM_2D:
        # Warning: hard-coded
        dimensionality = ["data_3d"]
        # from cfg we are only interested in the data path keys
        base_data_dir = os.path.join(cfg.DATA.HOME, cfg.DATA.DATA_DIR_PATH)
        data_dir = os.path.join(base_data_dir, cfg.DATA.BRAIN_VOL)
        # directory where results are going to be stored, this is preset
        out_dir = f"{data_dir}/data_2d/train"
        # expected 3d zarrs in this path
        data_dir_3d = os.path.join(data_dir, dimensionality[0], "train")
        print(f"INPUT Data dir is :{data_dir_3d}")

        if cfg.PREPROCESS.EXPORT_2D_FROM_3D:
            """
            Create 2D zarrs from their 3D counter-parts placed in `/path/DATASET`/data_3d`.
            This conversion can be slow as it has not been parallelised yet. Also, there is no resume functionality,
            it will always overwrite existing datasets.
            
            Expected keys in 2D zarrs:
            `volumes/raw/0...{n-zslices}`
            `volumes/labels/0...{n-zslices}`
            `volumes/labels_mask/0...{n-zslices}`
         
            """
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

            # TODO: parallelise this based on available cpus
            zarr_files = glob.glob(os.path.join(data_dir_3d, "*.*"), recursive=True)

            assert len(zarr_files), \
                f"There are no files in the input data directory!! Please place zarrs at {data_dir_3d}"

            # # Create a ThreadPoolExecutor with a specified number of threads (adjust as needed)
            num_threads = 4
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Use map to process all Zarr files concurrently
                results = list(executor.map(create_data_in_parallel, zarr_files, [out_dir] * len(zarr_files),
                                            [cfg] * len(zarr_files)))

            # deletes a proportion of zero regions; training is very slow in 2D otherwise
            # TODO: parallelise this based on available cpus
            zarr_files = glob.glob(os.path.join(out_dir, "*.*"), recursive=True)
            for z in zarr_files:
                find_and_interactively_delete_zero_levels(z)

    # these are the datasets that must be hist-matched
    if cfg.PREPROCESS.HISTOGRAM_MATCH is not None and len(cfg.PREPROCESS.HISTOGRAM_MATCH) > 1:
        """
        Match histograms of 2D source and target datasets. Hence, both datasets must exist in path.
        Design logic: We are forward thinking here, such that we can train a model to be slightly agnostic to
        the looks of the images (contrast/intensity) distributions. We piggy back on existing labelled datasets
        to generalise on our target dataset. However, this is still experimental, may (not) work. 
        Additionally, this has to be a preprocessing step, because you do not want load your targets for on-the-fly 
        augmentations.                
        """

        logger.debug("Warning: You want to \n \
            Match histograms of 2D source and target datasets. Hence, both datasets must exist in path.\n \
            Design logic: We are forward thinking here, such that we can train a model to be slightly agnostic to\n \
            the looks of the images (contrast/intensity) distributions. We piggy back on existing labelled datasets\n \
            to generalise on our target dataset. However, this is still experimental, may (not) work.")

        match_histograms(base_data_dir, cfg.PREPROCESS.HISTOGRAM_MATCH,
                         ["data_2d"] if cfg.DATA.DIM_2D else ["data_3d"],
                         cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description here.")
    parser.add_argument("--background_value", type=int, default=None, help="Background value for labels")
    args = parser.parse_args()
    main(args)
