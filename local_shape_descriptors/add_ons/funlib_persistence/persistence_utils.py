from funlib.persistence import prepare_ds
from funlib.geometry import Roi, Coordinate
import logging
from data_utils.preprocess_volumes.pad_input import pad_input 

# get the same logger that is defined in parent script
logger = logging.getLogger(__name__)


def prepare_predict_datasets(cfg, dtype, ds_key, ds_size, total_roi, voxel_size, num_channels=None, delete_ds=True):
    """With `funlib.persistence` create the .zarr dataset placeholders for output"""

    logger.warning("Warning: The datasets if they exist in the output  will always be re-created. "
                   "Hence pass delete_ds=False if you wish to preserve them and only add something new.")

    total_roi = Roi(total_roi.get_offset(), total_roi.get_shape())
    if not total_roi.shape.is_multiple_of(voxel_size):
        print(f"total roi {total_roi} is not a multiple of voxel_size: {voxel_size}")
    prepare_ds(
        filename=cfg.DATA.OUTFILE,
        ds_name=ds_key,
        total_roi=Roi(total_roi.get_offset(), total_roi.get_shape()),
        voxel_size=voxel_size,
        dtype=dtype,
        write_size=ds_size,
        num_channels=num_channels,
        delete=delete_ds  # always delete existing datasets
    )
