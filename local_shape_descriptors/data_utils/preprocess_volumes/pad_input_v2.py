"""
Pads input EM with zeros or in any other mode. Adjusts coordinates
Input must fit into memory (RAM).

"""
import argparse
import numpy as np
import zarr
import h5py
import os


def calculate_padding(input_shape, output_shape):
    pad_sizes = []
    for dim_in, dim_out in zip(input_shape, output_shape):
        pad_total = dim_out - dim_in
        pad_sizes.append((pad_total // 2, pad_total - pad_total // 2))
    return pad_sizes


def nearest_multiple_of(dim, voxel):
    return dim if dim % voxel == 0 else voxel * ((dim // voxel) + 1)


def nearest_multiple_of_voxel_size(input_shape, voxel_size):
    return tuple(nearest_multiple_of(x, v) for x, v in zip(input_shape, voxel_size))


def pad_input_array(input_arr, output_shape, mode='constant'):
    input_shape = input_arr.shape
    if input_shape == output_shape:
        return input_arr
    pad_sizes = calculate_padding(input_shape, output_shape)
    return np.pad(input_arr[:], pad_sizes, mode=mode)


def read_array(filepath, dataset):
    if filepath.endswith('.zarr'):
        return zarr.open(filepath, mode='r')[dataset]
    else:
        with h5py.File(filepath, 'r') as f:
            return f[dataset][:]


def get_voxel_size_nm(arr):
    if 'resolution' in arr.attrs:
        resolution = arr.attrs['resolution']
        voxel_size_nm = tuple(map(float, resolution))
        if len(voxel_size_nm) != 3:
            raise ValueError("Resolution attribute must be a 3-element list (z, y, x)")
        return voxel_size_nm
    else:
        raise KeyError("Resolution attribute not found in dataset")


def adjust_annotation_locations(root, original_shape, padded_shape, voxel_size_nm, save_backup=True):
    location_key = 'annotations/locations'
    offset_voxels = [(p - o) // 2 for o, p in zip(original_shape, padded_shape)]
    offset_nm = np.array(offset_voxels) * np.array(voxel_size_nm)

    if location_key not in root:
        print("No annotations/locations dataset found.")
        return

    locs = root[location_key][:]
    adjusted_locs = locs + offset_nm

    # Backup original
    if save_backup:
        backup_key = location_key + '_original'
        if backup_key not in root:
            root[backup_key] = locs

    # Replace with adjusted annotations
    del root[location_key]
    root[location_key] = adjusted_locs

    print(f"Adjusted {len(locs)} annotations by offset {offset_nm} nm")


def save_array(output_arr, out_path, dataset):
    if out_path.endswith('.zarr'):
        root = zarr.open(out_path, mode='a')
        if dataset in root:
            backup_dataset = dataset + '_original'
            if backup_dataset in root:
                print(f"Backup {backup_dataset} already exists. Skipping backup.")
            else:
                root.copy(dataset, backup_dataset)
            del root[dataset]
        root.create_dataset(dataset, data=output_arr)
    else:
        with h5py.File(out_path, 'a') as f:
            if dataset in f:
                backup_dataset = dataset + '_original'
                if backup_dataset in f:
                    print(f"Backup {backup_dataset} already exists. Skipping backup.")
                else:
                    f.copy(dataset, backup_dataset)
                del f[dataset]
            f.create_dataset(dataset, data=output_arr)


def main(args):
    if args.voxel_size is not None:
        voxel_size = tuple(map(int, args.voxel_size.split(',')))
    if args.output_shape is not None:
        output_shape = tuple(map(int, args.output_shape.split(','))) if args.output_shape else None

    input_arr = read_array(args.file, args.dataset)
    input_shape = input_arr.shape

    if args.file.endswith('.zarr'):
        f = zarr.open(args.file, mode='r')[args.ds]
    else:
        f = h5py.File(args.file, 'r')[args.ds]

    voxel_size_nm = get_voxel_size_nm(f)

    if output_shape is None:
        output_shape = nearest_multiple_of_voxel_size(input_shape, voxel_size)

    padded_arr = pad_input_array(input_arr, output_shape)

    output_path = args.output if args.output else args.file
    save_array(padded_arr, output_path, args.dataset)

    # 🔻 NEW: Adjust annotations in the saved output
    if output_path.endswith('.zarr'):
        root = zarr.open(output_path, mode='a')
        adjust_annotation_locations(root, original_shape=input_shape, padded_shape=output_shape,
                                    voxel_size_nm=voxel_size_nm)
    else:
        with h5py.File(output_path, 'a') as root:
            adjust_annotation_locations(root, original_shape=input_shape, padded_shape=output_shape,
                                        voxel_size_nm=voxel_size_nm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, help="Path to input Zarr or HDF5 file")
    parser.add_argument('-ds', '--dataset', required=True, help="Dataset path inside the file")
    parser.add_argument('-o', '--output', help="Optional output path; defaults to input path")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-vs', '--voxel_size', help="Voxel size, comma-separated, e.g. 32,32,32")
    group.add_argument('-out_shape', '--output_shape', help="Output shape to pad to, comma-separated")

    args = parser.parse_args()
    main(args)
