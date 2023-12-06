"""
Credits:
https://gist.github.com/omidalam/0c4848e169717c46ad2d9e52ec27f7b5
"""
import zarr
import numpy as np
from scipy.interpolate import interp1d


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
    f = zarr.open(
        "/media/samia/DATA/ark/connexion/data/PARKER/data_3d/test/parker_roi512_z9520-10032_y7797-8309_x13740-14252_clahe.zarr",
        "a")
    img = f["volumes/raw_original"][:]
    iso_out = iso_voxel(img, z_res=0.016, xy_res=0.008)
    f["volumes/raw_iso"] = iso_out.astype(np.uint8)
    f["volumes/raw_iso"].attrs["offset"] = (0, 0, 0)
    f["volumes/raw_iso"].attrs["resolution"] = (8, 8, 8)
