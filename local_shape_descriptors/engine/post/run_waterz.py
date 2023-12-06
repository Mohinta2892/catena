"""Post-processing, this does not work inside the docker, so run outside with the conda lsd env in cardona-gpu1
"""
import numpy as np
from data_utils.preprocess_volumes.utils import read_zarr
from .watershed_helpers import get_segmentation
from tqdm import tqdm


def run_waterz(cfg):
    results_zarr = read_zarr(cfg.DATA.OUTFILE, mode="r+")

    for th in tqdm(cfg.INS_SEGMENT.THRESHOLDS):
        threshold = round(th, 2)
        print(f"Running waterz with threshold {threshold}")

        # path to predicted affinity matrix is hard-coded 
        results = np.array(results_zarr[f"volumes/pred_affs"])

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
