import numpy
import h5py
import numpy as np
import zarr
import os


def calculate_padding(input_shape, output_shape):
    pad_sizes = []
    for dim_in, dim_out in zip(input_shape, output_shape):
        pad_total = dim_out - dim_in
        pad_sizes.append((pad_total // 2, pad_total - pad_total // 2))

    return pad_sizes


def nearest_multiple_of(input_arr_shape: int, voxel_size: int):
    if input_arr_shape % voxel_size == 0:
        return input_arr_shape
    else:
        return voxel_size * (input_arr_shape // voxel_size + 1)


def nearest_multiple_of_voxel_size(input_arr: np.ndarray, voxel_size: tuple):
    input_arr_shape = input_arr.shape
    return input_arr_shape, tuple(nearest_multiple_of(x, i) for x, i in zip(input_arr_shape, voxel_size))


def pad_input(input_arr, voxel_size, mode='constant'):
    input_shape, output_shape = nearest_multiple_of_voxel_size(input_arr=input_arr, voxel_size=voxel_size)
    if input_shape == output_shape:
        return input_arr
    
    pad_sizes = calculate_padding(input_shape, output_shape)
    # input_arr need 
    input_arr_padded = np.pad(input_arr[:], pad_sizes, mode=mode)

    return input_arr_padded

# if __name__ == '__main__':
#     pass
