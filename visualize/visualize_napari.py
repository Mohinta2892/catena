#!/home/samia/anaconda3/envs/napari-env/bin/python

"""
This script uses `napari` to visualize hdf and zarr datasets. The dataset keys are extracted in the
script automatically from the input file.
Assumes: napari is installed, please change the shebang above to point to the python path which has it.
If not installed follow this guide: `https://napari.org/stable/tutorials/fundamentals/installation.html`

To run the script:
/path/to/visualize_napari.py -f /path/to/hdf-zarr -sf [optional] integer -st [optional] integer -s [Z1:Z2,Y1:Y2,X1:X2]

Run: `/path/to/visualize_napari.py --help` to understand what to pass as arguments.

Note: with a shebang you don't need to run the script like `python scriptname.py`, it automatically calls for you.

Author: Samia Mohinta
Affiliation: Cardona lab, Cambridge University
"""
import napari
import numpy as np
import h5py
import zarr
import argparse
from pathlib import Path
import dask.array as da
from skimage.segmentation import relabel_sequential


def list_keys(f):
    """
    List all keys in a zarr/hdf file
    Args:
        path: path to the zarr/hdf file

    Returns:
        list of keys
    """

    def _recursive_find_keys(f, base: Path = Path('/')):
        _list_keys = []
        for key, dataset in f.items():
            if isinstance(dataset, zarr.Group):
                new_base = base / key
                _list_keys += _recursive_find_keys(dataset, new_base)

            elif isinstance(dataset, h5py._hl.group.Group):
                new_base = base / key
                _list_keys += _recursive_find_keys(dataset, new_base)

            elif isinstance(dataset, zarr.Array):
                new_key = str(base / key)
                _list_keys.append(new_key)

            elif isinstance(dataset, h5py._hl.dataset.Dataset):
                new_key = str(base / key)
                _list_keys.append(new_key)

        return _list_keys

    return _recursive_find_keys(f)


def view_pred(f, z_sf=None, z_st=None, slices=None):
    datasets_in_file = list_keys(f)

    v = napari.Viewer()

    # experimental feature to just load a portion of 2D slices.
    # Todo: cope with 2d and 3d simulataneously
    # expected input should be like : [ volumes/raw/0, volumes/raw/1, ... volumes/raw/n]
    if z_sf is not None and z_st is not None:
        datasets_in_file = sorted(datasets_in_file, key=lambda x: int(x.split('/')[-1]))
        datasets_in_file = datasets_in_file[
                           z_sf * 3:z_st * 3]  # HACK: multiply by 3 because we have 3 group-levels: raw,labels,mask
        print(datasets_in_file)

    parsed_slices = ':'
    if slices is not None:
        parsed_slices = parse_slice_args(slices)

    for ds in datasets_in_file:

        if 'label' in ds or 'seg' in ds:
            # hack to relabel mito neuron-ids because all show up as green
            # remember this is only for viz purposes, data used for training will definitely use original label values
            # in uint64 (likely very large numbers for hemibrain)
            if 'neuron_ids' in ds or 'seg' in ds:
                relabelled, forward_map, inverse_map = relabel_sequential(
                    f[ds][parsed_slices] if slices is not None else f[ds][:])
                v.add_labels(relabelled, name=ds, blending='additive', opacity=0.7)
            else:
                v.add_labels(f[ds][parsed_slices] if slices is not None else f[ds][:], name=ds,
                             blending='additive', opacity=0.7)
        else:
            v.add_image(da.from_array(f[ds][parsed_slices] if slices is not None else f[ds][:]), name=ds,
                        blending='additive', opacity=0.7, multiscale=False)

    napari.run()


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


def read_hdf(f, mode='r'):
    """read and return the hdf file as a hdf file object"""
    return h5py.File(f, mode=mode)


def read_zarr(f, mode='r'):
    """read and return the zarr file as a zarr file object"""
    return zarr.open(f, mode=mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", required=True, help="A snapshot zarr or hdf file")
    parser.add_argument("-m", default='r', help="Open the file in `r`ead, `w`rite, `a`ppend mode.")
    parser.add_argument("-sf", type=int, default=None, help="Would you like to load only a few slices of your 2D zarr?"
                                                            "Pass an integer to indicate where to start slicing."
                                                            "Does not work with 3D yet!!")
    parser.add_argument("-st", type=int, default=None, help="Would you like to load only a few slices of your 2D zarr?"
                                                            "Pass an integer to indicate where to end slicing."
                                                            "Does not work with 3D yet!!")
    parser.add_argument("-s", default=None, help="Slices to extract from volume in format Z1:Z2,Y1:Y2:X1:X2")

    args = parser.parse_args()

    if args.f.endswith('.zarr'):
        # LHS named so to avoid any keyword conflict
        file_ = read_zarr(args.f, args.m)

    elif args.f.endswith(('.hdf', '.h5', '.hdf5')):
        file_ = read_hdf(args.f, args.m)

    view_pred(file_, z_sf=args.sf, z_st=args.st, slices=args.s)
