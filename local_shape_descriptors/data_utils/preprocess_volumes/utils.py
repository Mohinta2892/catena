"""
Requirements:

Checks:
1. Train test split based on input zarrs
2. Check files not corrupt and contain voxel_size and resolution, return them to user
3. If Super-resolve is set for anisotropic to isotropic

5. If a folder with images are provided, convert to zarrs. However have to recursively find raw and labels.
User only needs to give the train folder, we split it into val and test if specified.
 Ask user for the following data organisation:
 /some/path/to/root
    - train # user points till here?!
        - Dataset1 # these are all folder paths, could be like `hemi/train/zarr_2D/fb_dhsfdf_545461`
            - raw
                -image1.jpg
                -image*.jpg
            - label
                -image1.jpg
                -image*.jpg
        - Dataset2
            - raw
                -image1.jpg
                -image*.jpg
            - label
                -image1.jpg
                -image*.jpg
Expected output:
Zarr files at path: /some/path/to/root/preprocessed_data/train/fb_dhsfdf_545461.zarr
For 2D files, the dataset keys with change like "volumes/raw/{image_index}". Must address this in the train.py
"""

import os.path
from pathlib import Path
import urllib
import h5py
import zarr
from typing import List, Union
import numpy as np
import matplotlib.pyplot as plt
import re


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


def is_path(url_or_path):
    # """Checks path to input is a folder path. Raises error if it is a path but empty."""
    # if os.path.isdir(url_or_path):
    #     # sanity check: path not empty, must contain files
    #     assert len(os.listdir(url_or_path)), "Your input folder is empty"
    #     return True
    # return False
    pass


def is_url(url_or_path):
    # if urllib.parse.urlparse(url_or_path).scheme != "":
    #     return True
    # else:
    #     return is_path(url_or_path)
    pass


def read_zarr(f, mode='r'):
    return zarr.open(f, mode=mode)


def calculate_min_2d_samples(in_files: List[Path]):
    ds = "volumes/raw"
    min_slices = 0
    for f in in_files:
        z = read_zarr(f)
        if min_slices == 0:
            min_slices = len(z[ds].items())
        elif len(z[ds].items()) < min_slices:
            min_slices = len(z[ds].items())

    return min_slices


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def collect_items(item, item_list):
    item_list.append(int(item))


def find_and_interactively_delete_zero_levels(zarr_store):
    zarr_store = zarr.open(zarr_store, "a")
    visited_items = []
    datasets = list_keys(zarr_store)
    if 'volumes/labels_mask' not in datasets:
        return
    # Determine the total number of levels
    zarr_store['volumes/labels_mask'].visit(lambda item: collect_items(item, visited_items))

    # Initialize a list to store levels that are fully 0
    zero_levels = []

    # Iterate through levels and check if they are fully 0
    for i in visited_items:
        dataset = zarr_store['volumes/labels_mask'][str(i)]
        if (dataset[:] == 0).all():
            zero_levels.append(i)

    # Inform the user how many zero levels exist
    print(f"Total zero levels: {len(zero_levels)}")

    if not len(zero_levels):
        return
    # Ask the user if they want to delete zero levels
    delete_option = input("Do you want to delete some zero levels (yes/no)? ").strip().lower()

    if delete_option in ["yes", "y"]:
        # Ask the user for the percentage of zero levels to delete
        delete_percentage = float(input("Enter the percentage of zero levels to delete (e.g., 20): "))
        num_levels_to_delete = int(delete_percentage / 100 * len(zero_levels))

        # Delete the specified proportion of zero levels
        for i in range(num_levels_to_delete):
            if zero_levels:
                level_to_delete = zero_levels.pop()
                del zarr_store['volumes/labels_mask'][str(level_to_delete)]
                del zarr_store['volumes/labels'][str(level_to_delete)]
                del zarr_store['volumes/raw'][str(level_to_delete)]

        # Now, `zero_levels` contains the levels that are fully 0 after deletion
        return zero_levels

    else:
        # Return the original zero levels if the user chooses not to delete
        return zero_levels


def check_dimensionality(in_files):
    """TODO: Parallelise"""
    for f in in_files:
        z = read_zarr(f)
        # expected ds key
        ds = "volumes/raw"

        # check num of labels == num of raw slices
        assert len(z[ds].items()) == len(z["volumes/labels"].items()), "Mismatch in the num. of raw and label slices"
        # check shape of all raw slices are actually 2D
        for i in range(len(z[ds].items())):
            z_shape = z[f"{ds}/{i}"].shape
            assert len(z_shape) < 3, f"Slice {z[f'{ds}/{i}']} shape {z_shape} is not 2D"


def create_data(
        url_or_path: Union[str, Path],
        outfile_path: Union[str, Path],
        offset: tuple = (0, 0),
        resolution: tuple = (8, 8),
        sections: Union[list, range] = None,
        squeeze: bool = True,
        background_value: np.uint64 = 0,
        is_mito: int = 0):
    """ Credits: Arlo Sheridan
        Change log: url can now be a path
        Todo: add back the url download and auto read. Make it like torch_em
    """
    # we will add url checks later, for now we assume all files hdf/zarr are on disk
    if url_or_path.lower().endswith((".hdf", ".h5", ".hdf5")):
        in_f = h5py.File(url_or_path, mode='r')
    elif url_or_path.lower().endswith((".zarr", ".n5")):
        in_f = zarr.open(url_or_path, mode='r')
    else:
        in_f = None
        assert in_f is not None, "InputError: Input file is None, please check paths!!"

    # TODO: Expected dataset keys are hardcoded for now.
    # We have an utils to navigate this automatically. Will be added later.
    raw = in_f['volumes/raw']
    keys_in_ds = list_keys(in_f)

    if is_mito == 1:
        # should be `mito_ids` but for now is hack, mito_ids == neuron_ids in the 3D datasets
        ds_ids = ["neuron_ids"]
    elif is_mito == 2:
        ds_ids = ["neuron_ids", "mito_ids"]
    else:
        ds_ids = ["neuron_ids"]

    # we will do it explicitly to avoid errors:
    if len(ds_ids) > 1:
        if any(id_ in s for id_ in ds_ids for s in keys_in_ds):
            try:
                labels = in_f[f'volumes/labels/{ds_ids[0]}'][:]
                # we need to create mask for labels just to be cautious in case we don't have labels across all slices
                labels_mask = np.ones_like(labels).astype(np.uint8)
                assert background_value in labels, f"Background value {background_value} must exist in labels array!"
                background_mask = labels == background_value
                labels_mask[background_mask] = 0
                # reset the labels background from 18446744073709551613 to 0 for hemi-brain
                labels[background_mask] = 0

                labels_mito = in_f[f'volumes/labels/{ds_ids[0]}'][:]
                labels_mask_mito = np.ones_like(labels_mito).astype(np.uint8)
                assert background_value in labels_mito, f"Background value {background_value} must exist in labels array!"
                background_mask = labels_mito == background_value
                labels_mask_mito[background_mask] = 0
                # reset the labels background from 18446744073709551613 to 0 for hemi-brain
                labels_mito[background_mask] = 0

            except Exception as e:
                print(f"Have you not stored your labels in the 3D zarr like: `volumes/labels/{ds_ids[0]}?")

    else:
        # either mito_ids or neuron_ids exist
        if any(id_ in s for id_ in ds_ids for s in keys_in_ds):
            try:
                labels = in_f[f'volumes/labels/{ds_ids[0]}'][:]
                # we need to create mask for labels just to be cautious in case we don't have labels across all slices
                labels_mask = np.ones_like(labels).astype(np.uint8)
                assert background_value in labels, f"Background value {background_value} must exist in labels array!"
                background_mask = labels == background_value
                labels_mask[background_mask] = 0
                # reset the labels background from 18446744073709551613 to 0 for hemi-brain
                labels[background_mask] = 0
            except Exception as e:
                print(f"Have you not stored your labels in the 3D zarr like: `volumes/labels/{ds_ids[0]}?")

    outfile_name = os.path.join(outfile_path, os.path.basename(url_or_path))
    container = zarr.open(outfile_name, 'a')

    if sections is None:
        sections = range(raw.shape[0] - 1)

    index_reset_counter = 0
    for index, section in enumerate(sections):

        raw_slice = raw[section]
        for id_ in ds_ids:
            if any(id_ in s for s in keys_in_ds):
                labels_slice = labels[section]
                labels_mask_slice = labels_mask[section]

            if squeeze:
                raw_slice = np.squeeze(raw_slice)
                if any(id_ in s for s in keys_in_ds):
                    labels_slice = np.squeeze(labels_slice)
                    labels_mask_slice = np.squeeze(labels_mask_slice)

            if any(id_ in s for s in keys_in_ds) and not np.sum(labels_mask_slice):
                print(f"Skip writing, mask is empty, meaning the labels do not exist in this {index}")
                continue

            print(f'Writing data for section {section}')

            for ds_name, data in [('volumes/raw', raw_slice)]:
                container[f'{ds_name}/{index_reset_counter}'] = data
                if "offset" in raw.attrs.keys():
                    container[f'{ds_name}/{index_reset_counter}'].attrs['offset'] = offset if raw.attrs[
                                                                                                  "offset"] is None \
                        else raw.attrs["offset"][1:]
                else:
                    container[f'{ds_name}/{index_reset_counter}'].attrs['offset'] = offset

                if "resolution" in raw.attrs.keys():
                    container[f'{ds_name}/{index_reset_counter}'].attrs[
                        'resolution'] = resolution if raw.attrs["resolution"] is None \
                        else raw.attrs["resolution"][1:]
                else:
                    container[f'{ds_name}/{index_reset_counter}'].attrs[
                        'resolution'] = resolution

            # we separate this out since we do not know if labels exist/the offsets differ
            # override by passing offset and resolution as arguments
            if any(id_ in s for s in keys_in_ds):
                for ds_name, data in [('volumes/labels', labels_slice), ('volumes/labels_mask', labels_mask_slice)]:
                    container[f'{ds_name}/{index_reset_counter}'] = data
                    container[f'{ds_name}/{index_reset_counter}'].attrs['offset'] = offset if raw.attrs[
                                                                                                  "offset"] is None \
                        else raw.attrs["offset"][1:]
                    container[f'{ds_name}/{index_reset_counter}'].attrs['resolution'] = resolution \
                        if raw.attrs["resolution"] is None \
                        else raw.attrs["resolution"][1:]  # because cropping along z-dim and data is ZYX
            index_reset_counter += 1

    return outfile_name


def create_lut(labels):
    max_label = np.max(labels)

    lut = np.random.randint(
        low=0,
        high=255,
        size=(int(max_label + 1), 3),
        dtype=np.uint8)

    lut = np.append(
        lut,
        np.zeros(
            (int(max_label + 1), 1),
            dtype=np.uint8) + 255,
        axis=1)

    lut[0] = 0
    colored_labels = lut[labels]

    return colored_labels


def imshow(
        raw=None,
        ground_truth=None,
        target=None,
        prediction=None,
        lsd=None,
        h=None,
        shader='jet',
        subplot=True,
        show=True  # enables plt.show()
):
    """
    Adapted from Arlo Sheridan's imshow example
    Change log:
        - return the figure object
        - condition on showing the plot

    :param raw: raw EM image slice
    :param ground_truth: label corresponding to the raw image slice
    :param target: affinities image calculated from the label
    :param prediction: affinities prediction from the model
    :param lsd: lsd pre from the model
    :param h: horizontal spacing between subplots, default None
    :param shader: cmap passed, default 'jet'
    :param subplot: enables subplot creation, default True
    :param show: enables plt.show() if True
    :return: figure object
    """

    plt.close('all')
    rows = 0

    if raw is not None:
        rows += 1
        cols = raw.shape[0] if len(raw.shape) > 2 else 1
    if ground_truth is not None:
        rows += 1
        cols = ground_truth.shape[0] if len(ground_truth.shape) > 2 else 1
    if target is not None:
        rows += 1
        cols = target.shape[0] if len(target.shape) > 2 else 1
    if prediction is not None:
        rows += 1
        cols = prediction.shape[0] if len(prediction.shape) > 2 else 1
    if lsd is not None:
        rows += 1
        cols = lsd.shape[0] if len(lsd.shape) > 2 else 1

    if subplot:
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(10, 4),
            sharex=True,
            sharey=True,
            squeeze=False)

    if h is not None:
        fig.subplots_adjust(hspace=h)

    def wrapper(data, row, name="raw"):

        if subplot:
            if len(data.shape) == 2:
                if name == 'raw':
                    axes[0][0].imshow(data, cmap='gray')
                    axes[0][0].set_title(name)
                else:
                    axes[row][0].imshow(create_lut(data))
                    axes[row][0].set_title(name)

            elif len(data.shape) == 3:
                for i, im in enumerate(data):
                    if name == 'raw':
                        axes[0][i].imshow(im, cmap='gray')
                        axes[0][i].set_title(name)
                    else:
                        axes[row][i].imshow(create_lut(im))
                        axes[row][i].set_title(name)

            else:
                for i, im in enumerate(data):
                    axes[row][i].imshow(im[0] + im[1], cmap=shader)
                    axes[row][i].set_title(name)

        else:
            if name == 'raw':
                plt.imshow(data, cmap='gray')
            if name == 'labels':
                plt.imshow(data, alpha=0.5)

    row = 0

    if raw is not None:
        wrapper(raw, row=row)
        row += 1
    if ground_truth is not None:
        wrapper(ground_truth, row=row, name='labels')
        row += 1
    if target is not None:
        wrapper(target, row=row, name='target')
        row += 1
    if prediction is not None:
        wrapper(prediction, row=row, name='prediction')
        row += 1
    if lsd is not None:
        wrapper(lsd, row=row, name='lsd')
        row += 1

    if show:
        plt.show()

    return fig


if __name__ == '__main__':
    # create_data(
    #     'https://cremi.org/static/data/sample_A_20160501.hdf',
    #     # more complexity because need to check if should be read as hdf or zarr
    #     'training_data_1.zarr',
    #     offset=[0, 0],
    #     resolution=[4, 4])
    pass
