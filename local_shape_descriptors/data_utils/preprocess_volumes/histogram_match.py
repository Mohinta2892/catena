"""
Adapted from Daniel Franco's EM_Domain_Adaptation:
https://github.com/danifranco/EM_domain_adaptation/blob/main/Histogram%20Matching/hist_match.ipynb
This will probably work when source number of samples == target number of samples
"""
import shutil

import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from skimage import io
from glob import glob
import numpy as np
import random
from PIL import Image
from sklearn.linear_model import LinearRegression
import os
from .utils import read_zarr, list_keys, collect_items, natural_keys
from typing import Union, List
from pathlib import Path
import zarr
import h5py
from sys import stdout


def copy_datasets_from_multiple_sources(out_path, train_input_filepaths, datasets_to_copy, is_2d=False):
    """
    Copy specified datasets from multiple source Zarr stores to the destination Zarr store.

    Args:

    Returns:
        None
    """

    for source_f in train_input_filepaths:
        source_z = read_zarr(source_f)
        source_z_keys = list_keys(source_z)
        with zarr.open(os.path.join(out_path, os.path.basename(source_f)), mode="a") as dest_z:
            for dataset_name in datasets_to_copy:
                # Copy the dataset from source to destination as a whole
                if is_2d:
                    zarr.copy(source_z[dataset_name], dest_z[dataset_name.split('/')[0]], log=stdout,
                              if_exists='replace')
                else:
                    zarr.copy(source_z[dataset_name], dest_z, name=dataset_name,
                              # [f"/{'/'.join(dataset_name.split('/')[:-1])}"],
                              log=stdout, if_exists='replace', dry_run=False)

    print("Datasets copied successfully")


def get_zarr_list(dir, ds_keys=["volumes/raw"]):
    """
    Reads all zarrs in the specified directory and returns a list of numpy arrays representing the images

    Args:
    dir: The directory that contains the images.

    Returns:
    A list of numpy arrays representing the images.
    """
    if dir[-1] == '/':
        dir = dir[:-1]
    train_raw_path = dir + '/*.*'

    train_raw_filenames = glob(train_raw_path, recursive=True)
    train_raw_filenames.sort()

    # with list comprehension we can load all raws of all datasets. However, we need them to be stored separately as
    # pre-processed zarr as we need access to the labels from the source too. Hence, we do it more crudely.
    # train_raw = [read_zarr(x)[f"{k}/{index}"][:]
    #              for x in train_raw_filenames
    #              for k in ds_keys
    #              for index in range(len(read_zarr(x)[k].items()))]

    train_raw = {}

    for x in train_raw_filenames:
        zarr_data = read_zarr(x)  # Read the Zarr file once
        file_datasets = []  # List to store NumPy arrays for this file
        for k in ds_keys:
            num_items = []
            try:
                zarr_data[k].visit(lambda item: collect_items(item, num_items))
                num_items.sort()
                for index in num_items:
                    dataset_key = f"{k}/{index}"
                    dataset_array = zarr_data[dataset_key][:]
                    file_datasets.append(dataset_array)
            except Exception as e:
                # 3D
                dataset_array = zarr_data[k][:]
                file_datasets.append(dataset_array)

        # Assign the list of NumPy arrays to the dictionary with the filename as the key
        train_raw[x] = file_datasets

    return train_raw


def save_zarr(out_path: Union[str, Path], hm_sx: dict, offset: tuple,
              resolution: tuple, ds_keys: str = "volumes/raw", is_2d=False):
    """ We save a different file, to preserve the input as is. Do not want to mistakenly overwrite anything!
    """
    for k in hm_sx.keys():
        with zarr.open(os.path.join(out_path, os.path.basename(k)), "a") as z:
            if is_2d:
                for i in range(len(hm_sx[k])):
                    z[f"{ds_keys}/{i}"] = hm_sx[k][i]
                    z[f"{ds_keys}/{i}"].attrs["offset"] = offset
                    z[f"{ds_keys}/{i}"].attrs["resolution"] = resolution

                print(f"saved slices {i} in {out_path, os.path.basename(k)}")
            else:
                # 3D
                z[f"{ds_keys}"] = hm_sx[k]
                z[f"{ds_keys}"].attrs["offset"] = offset
                z[f"{ds_keys}"].attrs["resolution"] = resolution
                print(f"saved {k} in {out_path, os.path.basename(k)}")


def get_image_list(dir: Union[str, Path]):
    """
        TODO: look and adapt from Biapy this 2D organisation for tiffs if needed
         Reads all the images in the specified directory and returns a list of numpy arrays representing the
         images

         Args:
           dir: The directory that contains the images.

         Returns:
           A list of numpy arrays representing the images.
        """
    if dir[-1] == '/':
        dir = dir[:-1]
    train_label_path = dir + '*.*'

    train_label_filenames = glob(train_label_path, recursive=True)
    train_label_filenames.sort()

    print('Label images loaded: ' + str(len(train_label_filenames)))

    # read training images and labels
    train_lbl = [img_as_ubyte(np.array(io.imread(x, as_gray=True), dtype='uint8')) for x in train_label_filenames]
    return train_lbl


def save_images(imgs, dst_path, name_prefix, fnames, format='.png', convert=''):
    """
     Save images to disk

     Args:
       imgs: The list of images to be saved.
       dst_path: the destination directory where the images will be saved.
       name_prefix: The prefix of the file name.
       fnames: The filenames of the images to be saved.
       format: The format of the output images. Defaults to .png
       convert: 'L' for greyscale, 'RGB' for color, or '' for nothing.
    """
    for i, img in enumerate(imgs):
        im = Image.fromarray(img)
        if convert != '':
            im = im.convert(convert)
        im.save(os.path.join(dst_path, fnames[i] + name_prefix + format), quality=100, subsampling=0)


def create_dir(dir: Union[str, Path]):
    """
     Create a directory if it doesn't exist

     Args:
       dir: The directory where the model will be saved.
    """
    if os.path.exists(dir):
        shutil.rmtree(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

    print(f"Saving preprocessed datasets here {dir}")


# apply histogram matching to any image (not mask) from source dataset, using target mean histogram
# this histogram matching works by matching the given cumulative histogram to target cumulative histogram
def histogram_matching(target_imgs, apply_prob):
    """
     Given a set of images, it will obtain their mean histogram. The number of 0s of this histogram will be predicted
      using Linear regression, with the real number of 1 and 2. It returns a function that apply histogram matching,
      using the calculated histogram. This returned function will apply a random histogram matching to each image with probability
      apply_prob.

     Args:
       target_imgs: the target domain images, from which mean histogram will be obtained (with predicted number of 0s)
       apply_prob: probability of applying the histogram matching

     Returns:
       A function that takes an image as input and returns a modified image or the original image, with
     the given probability.
    """

    LR = LinearRegression()
    hist_mean, _ = np.histogram(np.array(target_imgs).ravel(), bins=np.arange(256))
    reg = LR.fit(np.reshape([1, 2], (-1, 1)),
                 np.reshape(hist_mean[1:3], (-1, 1)))  # use next 2 values to predict using LR
    hist_mean[0] = max(0, float(reg.predict(np.reshape([0, ], (-1, 1)))))  # predict 0 values (due to padding)
    hist_mean = hist_mean / np.array(target_imgs).shape[0]  # number of images

    # calculate normalized quantiles
    # tmpl_size = target_imgs[0].size # once 0 value is predicted,
    # the sum of pixels are not the same as the one in the image, so size is no longer useful
    tmpl_size = np.sum(hist_mean)
    tmpl_quantiles = np.cumsum(hist_mean) / tmpl_size

    # based on scikit implementation.
    # source: https://github.com/scikit-image/scikit-image/blob/v0.18.0/skimage/exposure/histogram_matching.py#L22-L70
    def _match_cumulative_cdf(source, tmpl_quantiles):
        src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                               return_inverse=True,
                                                               return_counts=True)

        # calculate normalized quantiles
        # replace number of 0s with lineal regression in order to avoid padding
        if src_values[0] == 0:
            if src_values[:3].tolist() == [0, 1, 2]:
                reg = LR.fit(np.reshape([1, 2], (-1, 1)),
                             np.reshape(src_counts[1:3], (-1, 1)))  # use next 2 values to predict using LR
                pred_0 = max(0, float(reg.predict(np.reshape([0, ], (-1, 1)))))  # predict 0 values (due to padding)
            else:
                # images can be completely black
                pred_0 = 1 if len(src_counts) == 1 else 0  # 1 if completely black, else 0

            src_size = (source.size - src_counts[0]) + pred_0  # more efficient than 'sum(src_counts)'
            src_counts[0] = pred_0  # replace histogram 0s with predictted value
        else:
            src_size = source.size  # number of pixels

        src_quantiles = np.cumsum(src_counts) / src_size  # normalize
        interp_a_values = np.interp(src_quantiles, tmpl_quantiles, np.arange(len(tmpl_quantiles)))
        if src_values[0] == 0:
            interp_a_values[0] = 0  # we want to keep 0s, (padding)

        return interp_a_values[src_unique_indices].reshape(source.shape)

    def random_histogram_matching(image):
        """ This was meant to happen on the fly as an augmentation for EM domain adaptation.
         However, for us this is just a preprocessing step. Hence, we always apply the matching if user wants it."""
        # if random.random() < apply_prob:
        result = _match_cumulative_cdf(image, tmpl_quantiles)
        # else:
        #     result = image
        return result

    return random_histogram_matching


def match_histograms(data_path: Union[Path, str], datasets: List[str], dimensionality: List[str], cfg: dict = None):
    for source in datasets:
        for target in datasets:
            if source == target:
                continue

            print("\n S: {}\tT: {}\n".format(source, target))
            for dim in dimensionality:
                tx = get_zarr_list(os.path.join(data_path, target, dim, "train"), ds_keys=["volumes/raw"])

                # assumption: the target directory contains one file, all source files will be adapted to the target
                for k in tx.keys():
                    target_train_flat = np.array(tx[k]).ravel()  # for histogram matching
                hist_match = histogram_matching(target_train_flat, 1)
                # plt.hist(target_train_flat, bins=256)
                # plt.show()
                for p in ['train']:
                    in_dir = os.path.join(data_path, source, dim, p)

                    # Paths to the training images and their corresponding labels
                    train_input_path = in_dir

                    # # Read the list of file names
                    train_input_filepaths = glob(train_input_path + '/*.*')
                    # this may not work when alphanumeric strings are involved, failure case:  ['fig1', 'fig10', 'fig2']
                    # we do not care here since we are dealing with zarr and not images saved as fig1.jpg
                    train_input_filepaths.sort()

                    # read training images
                    sx = get_zarr_list(train_input_path)

                    # make this a dict too
                    # hm_sx {k: list[array]} we want only the array, hence list[array][0]
                    hm_sx = {k: [hist_match(t).astype(np.uint8).squeeze() for t in sx[k]][0] for k in sx.keys()}

                    # we keep the .zarr extension
                    train_input_filenames = [(os.path.basename(x)) for x in train_input_filepaths]
                    if dim == "data_3d":
                        out_path = os.path.join(data_path, "preprocessed_3d", source + "_s_t_" + target)
                    else:
                        out_path = os.path.join(data_path, "preprocessed", source + "_s_t_" + target)

                    create_dir(out_path)

                    try:
                        # Warning: Hard-coded offset and resolution and dataset keys
                        save_zarr(out_path=out_path, hm_sx=hm_sx,
                                  offset=cfg.PREPROCESS.SOURCE_DATA_OFFSET,
                                  resolution=cfg.PREPROCESS.SOURCE_DATA_RESOLUTION,
                                  ds_keys="volumes/raw", is_2d=cfg.DATA.DIM_2D)
                        copy_datasets_from_multiple_sources(out_path, train_input_filepaths,
                                                            datasets_to_copy=cfg.PREPROCESS.DATASETS_TO_COPY,
                                                            is_2d=cfg.DATA.DIM_2D)

                    except Exception as e:
                        print(e)

# if __name__ == '__main__':
#     data_path = "/media/samia/DATA/ark/connexion/data"
#     # dataset names inside the data_path (all combinations will be computed)
#     datasets = ['HEMI', "OCTO"]
#     dimensionality = ["data_3d", ]  # "data_3d"]
#     # directory where results are going to be stored
#     # - we do not need this cause be create a preprocessed folder on input path
#     # out_dir = "/media/samia/DATA/ark/connexion/data/HEMI/data_2d/"
#     match_histograms(data_path, datasets, dimensionality)
