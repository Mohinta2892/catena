import zarr
import numpy as np


def pad_zarr(in_file, ds):
    # below meta and parameters are specific to seymour, always adapt for other (especially) FIBSEM volumes
    input_shape = (120, 2116, 2116)
    # Define the desired output shape
    output_shape = (238, 2116, 2116)

    # Calculate the padding required for each dimension
    pad_sizes = []
    for dim_in, dim_out in zip(input_shape, output_shape):
        pad_total = dim_out - dim_in
        pad_sizes.append((pad_total // 2, pad_total - pad_total // 2))

    # can only be done if the array is small enough to be loaded to memory
    f = zarr.open(in_file, 'a')
    r = f[ds][:]

    # Padding
    r_pad = np.pad(r, pad_sizes, mode='reflect')

    f['volumes/raw_padded'] = r_pad
    # never forget to set resolution and offset, otherwise LSD does not run
    # original is 50,3.8,3.8, but LSD needs each voxel_size to be a multiple of one another
    f['volumes/raw_padded'].attrs['resolution'] = (48, 4, 4)  # this makes it goes as close as possible in `Z`
    f['volumes/raw_padded'].attrs['offset'] = (0, 0, 0)  # technically needs to be a multiple of voxel_size
    # this z value should apply slicing logic like [value:len(z)-value,...] to get to the original data
    f['volumes/raw_padded'].attrs['pad_sizes'] = pad_sizes

    print("Successfully added `volumes/raw_padded` as a dataset, point to this one when running the LSD prediction")


if __name__ == '__main__':
    in_file = '/media/samia/DATA/ark/dan-samia/lsd/funke/seymour/train/comb0.zarr'
    ds = 'volumes/raw'
    pad_zarr(in_file, ds)
