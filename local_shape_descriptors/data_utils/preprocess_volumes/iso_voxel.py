"""
Credits:
https://gist.github.com/omidalam/0c4848e169717c46ad2d9e52ec27f7b5
"""
import zarr
import numpy as np
from scipy.interpolate import interp1d
import argparse


def parse_slice_args(slice_args_str):
    """
    Parse slice arguments from a string and convert them into slice objects.

    Args:
        slice_args_str (str): String representing slice arguments.

    Returns:
        tuple: Tuple containing slice objects for Z, Y, and X dimensions.
    """
    slices = []
    for dim_str in slice_args_str.split(','):
        start_stop = dim_str.split(':')
        start = int(start_stop[0]) if start_stop[0] else None
        stop = int(start_stop[1]) if start_stop[1] else None
        slices.append(slice(start, stop))
    return tuple(slices)


def iso_voxel(img, z_res=.125, xy_res=0.08):
    """
    In microscopy images z-axis resolution is generally lower than xy. This causes the image voxels to be anisotropic.
    This function make the voxel isotropic by increasing the z-axis resolution to match xy-axes resolution.
    Parameters
    ----------
        img: a 3d array with ZXY dimension order.
        z_res: z-axis resolution in micron.
        xy_res: xy-axis resolution in micron.
    Returns:
    --------
    An image with isotropic voxel,ie. the pixel size is the same in zxy dimensions.
    """

    nz_in, nx, ny = img.shape

    heights = np.linspace(0, 1, nz_in)

    f_out = interp1d(heights, img.copy(), axis=0)

    nz_out = int(nz_in * z_res / xy_res)  # calculate the output z-axis dimensions
    new_heights = np.linspace(0, 1, nz_out)
    return f_out(new_heights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help="Zarr file which contains the dataset to transform")
    parser.add_argument('-ds', default="volumes/raw", help="Dataset inside zarr; default: `volumes/raw`")
    parser.add_argument('-roi', default="None:None, None:None, None:None",
                        help="Dataset inside zarr; default: `None:None,None:None,None:None`;"
                             " Ensure No spaces between indices")

    args = parser.parse_args()

    slices = parse_slice_args(args.roi)
    f = zarr.open(args.f, "a")
    img = f["volumes/raw"][slices]
    iso_out = iso_voxel(img, z_res=0.030, xy_res=0.008)
    f["volumes/raw_iso"] = iso_out.astype(np.uint8)
    f["volumes/raw_iso"].attrs["offset"] = (0, 0, 0)
    f["volumes/raw_iso"].attrs["resolution"] = (8, 8, 8)
