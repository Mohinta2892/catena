"""
This script has been adapted from Funkelab's daisy examples:
https://github.com/funkelab/daisy/blob/master/examples/hdf_to_zarr.py

You must have `batch_task.py` accessible. A local sqlite db gets created. Logs saved in the same folder as this script.

Upgrades:
    - Uses funlib.persistence and funlib.geometry for dependencies
    - Multiple datasets can be converted if `in_ds_name` is passed as None.
      Only works with volumes that have voxel_size > 2 and voxel_size values > (1,).
      Hence, you cannot save annotations yet with this. Please use the generic `hdf_to_zarr.py` for that.

Author: Samia Mohinta
Affiliation: Cardona lab, Cambridge University, UK
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import numpy as np
import skimage.measure

import daisy
import h5py
from funlib.geometry import Coordinate, Roi
from funlib.persistence import open_ds, prepare_ds

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from data_utils.preprocess_volumes.utils import list_keys
from data_utils.download_data.batch_task_daisy import BatchTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HDF2ZarrTask")


def calculateNearIsotropicDimensions(voxel_size, max_voxel_count):
    dims = len(voxel_size)

    voxel_count = 1
    vol_size = [k for k in voxel_size]
    voxel_dims = [1 for k in voxel_size]

    while voxel_count < max_voxel_count:
        for i in range(0, dims):
            if voxel_count >= max_voxel_count:
                continue
            if vol_size[i] == min(vol_size):
                vol_size[i] *= 2
                voxel_count *= 2
                voxel_dims[i] *= 2

    return voxel_dims


class HDF2ZarrTask(BatchTask):

    def _task_init(self):

        logger.info(f"Accessing {self.in_ds_name} in {self.in_file}")
        self.in_ds_list = []
        if self.in_ds_name is not None:
            try:
                self.in_ds = open_ds(self.in_file, self.in_ds_name)
                self.in_ds_list.append(self.in_ds)
            except Exception as e:
                logger.info(f"EXCEPTION: {e}")
                exit(1)
        elif self.in_ds_name is None:
            try:
                ds_name_all = list_keys(h5py.File(self.in_file))
                for ds in ds_name_all:
                    # all datasets loaded!
                    self.in_ds_list.append(open_ds(self.in_file, ds))

            except Exception as e:
                logger.info(f"EXCEPTION: {e}")
                exit(1)
        else:
            raise Exception("File type/dataset not supported!")

        # for 1 or many datasets
        for ds_name, in_ds in zip(ds_name_all, self.in_ds_list):
            self.in_ds = in_ds
            voxel_size = self.in_ds.voxel_size
            self.in_ds_name = ds_name

            if len(voxel_size) <= 2 and (voxel_size == (1,) or voxel_size == (1, 1)):
                # Let's just say we cannot fully tackle saving locations, ids with this script yet!
                # We can only convert 3D arrays with voxel size more than (1,1).
                logger.info(f"skipped {ds_name}")
                continue

            if self.in_ds.n_channel_dims == 0:
                num_channels = None
            elif self.in_ds.n_channel_dims == 1:
                num_channels = self.in_ds.shape[0]
            else:
                raise RuntimeError("more than one channel not yet implemented, sorry...")

            self.ds_roi = self.in_ds.roi

            sub_roi = None
            if self.roi_offset is not None or self.roi_shape is not None:
                assert self.roi_offset is not None and self.roi_shape is not None
                self.schedule_roi = daisy.Roi(tuple(self.roi_offset), tuple(self.roi_shape))
                sub_roi = self.schedule_roi
            else:
                self.schedule_roi = self.in_ds.roi

            if self.chunk_shape_voxel is None:
                self.chunk_shape_voxel = calculateNearIsotropicDimensions(
                    voxel_size, self.max_voxel_count
                )
                logger.info(voxel_size)
                logger.info(self.chunk_shape_voxel)
            self.chunk_shape_voxel = Coordinate(self.chunk_shape_voxel)

            self.schedule_roi = self.schedule_roi.snap_to_grid(voxel_size, mode="grow")
            out_ds_roi = self.ds_roi.snap_to_grid(voxel_size, mode="grow")

            self.write_size = self.chunk_shape_voxel * voxel_size

            scheduling_block_size = self.write_size

            self.write_roi = Roi((0,) * self.ds_roi.dims, scheduling_block_size)

            if sub_roi is not None:
                # with sub_roi, the coordinates are absolute
                # so we'd need to align total_roi to the write size too
                self.schedule_roi = self.schedule_roi.snap_to_grid(
                    self.write_size, mode="grow"
                )
                out_ds_roi = out_ds_roi.snap_to_grid(self.write_size, mode="grow")

            logger.info(f"out_ds_roi: {out_ds_roi}")
            logger.info(f"schedule_roi: {self.schedule_roi}")
            logger.info(f"write_size: {self.write_size}")
            logger.info(f"voxel_size: {voxel_size}")

            if self.out_file is None:
                self.out_file = ".".join(self.in_file.split(".")[0:-1]) + ".zarr"
            if self.out_ds_name is None:
                self.out_ds_name = self.in_ds_name

            delete = self.overwrite == 2

            self.out_ds = prepare_ds(
                self.out_file,
                self.out_ds_name,
                total_roi=out_ds_roi,
                voxel_size=Coordinate(voxel_size),
                write_size=self.write_size,
                dtype=self.in_ds.dtype,
                num_channels=num_channels,
                force_exact_write_size=True,
                compressor='default',
                delete=delete,
            )

    def prepare_task(self):

        assert len(self.chunk_shape_voxel) == 3

        logger.info(
            "Rechunking %s/%s to %s/%s with chunk_shape_voxel %s (write_size %s, scheduling %s)"
            % (
                self.in_file,
                self.in_ds_name,
                self.out_file,
                self.out_ds_name,
                self.chunk_shape_voxel,
                self.write_size,
                self.write_roi,
            )
        )
        logger.info("ROI: %s" % self.schedule_roi)

        worker_filename = os.path.realpath(__file__)
        self._write_config(worker_filename, extra_config=None)

        return self._prepare_task(
            total_roi=self.schedule_roi,
            read_roi=self.write_roi,
            write_roi=self.write_roi,
            check_fn=lambda b: self.check_fn(b),
        )

    def _worker_impl(self, block):
        """Worker function implementation"""
        self.out_ds[block.write_roi] = self.in_ds[block.write_roi]

    def check_fn(self, block):

        write_roi = self.out_ds.roi.intersect(block.write_roi)
        if write_roi.empty:
            return True

        return super()._default_check_fn(block)


def none_or_str(value):
    """
    Datatype to parse None passed.
    Copied from: https://stackoverflow.com/questions/48295246/how-to-pass-none-keyword-as-command-line-argument
    :param value: str
    :return: None or str
    """
    if value == 'None':
        return None
    return value


if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "run_worker":
        task = HDF2ZarrTask(config_file=sys.argv[2])
        task.run_worker()

    else:
        ap = argparse.ArgumentParser(description="Create a zarr/N5 container from hdf.")
        ap.add_argument("in_file", type=str, help="The input container")
        ap.add_argument("in_ds_name", type=none_or_str, nargs='?', default=None, help="The name of the dataset")
        ap.add_argument(
            "--out_file",
            type=str,
            default=None,
            help="The output container, defaults to be the same as in_file+.zarr",
        )
        ap.add_argument(
            "--out_ds_name",
            type=str,
            default=None,
            help="The name of the dataset, defaults to be in_ds_name",
        )
        ap.add_argument(
            "--chunk_shape_voxel",
            type=int,
            help="The size of a chunk in voxels",
            nargs="+",
            default=None,
        )
        ap.add_argument(
            "--max_voxel_count",
            type=int,
            default=256 * 1024,
            help="If chunk_shape_voxel is not given, use this value to calculate"
                 "a near isotropic chunk shape",
        )
        ap.add_argument("--roi_offset", type=int, help="", nargs="+", default=None)
        ap.add_argument("--roi_shape", type=int, help="", nargs="+", default=None)

        config = HDF2ZarrTask.parse_args(ap)
        task = HDF2ZarrTask(config)
        daisy_task = task.prepare_task()
        done = daisy.run_blockwise([daisy_task])
        if done:
            logger.info("Ran all blocks successfully!")
        else:
            logger.info("Did not run all blocks successfully...")
