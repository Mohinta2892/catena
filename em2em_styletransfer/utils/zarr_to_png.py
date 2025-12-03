import zarr
import tifffile
import numpy as np
import os
from skimage import segmentation
from glob import glob
import skimage.io as io
from scipy.ndimage import binary_erosion


# segmentation to affinity - copied from gunpowder
def seg_to_affgraph(seg, nhood=[[-1, 0], [0, -1]]):
    nhood = np.array(nhood)

    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = seg.shape
    nEdge = nhood.shape[0]
    dims = nhood.shape[1]
    aff = np.zeros((nEdge,) + shape, dtype=np.int32)

    if dims == 2:
        for e in range(nEdge):
            aff[
            e,
            max(0, -nhood[e, 0]): min(shape[0], shape[0] - nhood[e, 0]),
            max(0, -nhood[e, 1]): min(shape[1], shape[1] - nhood[e, 1]),
            ] = (
                    (
                            seg[
                            max(0, -nhood[e, 0]): min(shape[0], shape[0] - nhood[e, 0]),
                            max(0, -nhood[e, 1]): min(shape[1], shape[1] - nhood[e, 1]),
                            ]
                            == seg[
                               max(0, nhood[e, 0]): min(shape[0], shape[0] + nhood[e, 0]),
                               max(0, nhood[e, 1]): min(shape[1], shape[1] + nhood[e, 1]),
                               ]
                    )
                    * (
                            seg[
                            max(0, -nhood[e, 0]): min(shape[0], shape[0] - nhood[e, 0]),
                            max(0, -nhood[e, 1]): min(shape[1], shape[1] - nhood[e, 1]),
                            ]
                            > 0
                    )
                    * (
                            seg[
                            max(0, nhood[e, 0]): min(shape[0], shape[0] + nhood[e, 0]),
                            max(0, nhood[e, 1]): min(shape[1], shape[1] + nhood[e, 1]),
                            ]
                            > 0
                    )
            )

    elif dims == 3:
        for e in range(nEdge):
            aff[
            e,
            max(0, -nhood[e, 0]): min(shape[0], shape[0] - nhood[e, 0]),
            max(0, -nhood[e, 1]): min(shape[1], shape[1] - nhood[e, 1]),
            max(0, -nhood[e, 2]): min(shape[2], shape[2] - nhood[e, 2]),
            ] = (
                    (
                            seg[
                            max(0, -nhood[e, 0]): min(shape[0], shape[0] - nhood[e, 0]),
                            max(0, -nhood[e, 1]): min(shape[1], shape[1] - nhood[e, 1]),
                            max(0, -nhood[e, 2]): min(shape[2], shape[2] - nhood[e, 2]),
                            ]
                            == seg[
                               max(0, nhood[e, 0]): min(shape[0], shape[0] + nhood[e, 0]),
                               max(0, nhood[e, 1]): min(shape[1], shape[1] + nhood[e, 1]),
                               max(0, nhood[e, 2]): min(shape[2], shape[2] + nhood[e, 2]),
                               ]
                    )
                    * (
                            seg[
                            max(0, -nhood[e, 0]): min(shape[0], shape[0] - nhood[e, 0]),
                            max(0, -nhood[e, 1]): min(shape[1], shape[1] - nhood[e, 1]),
                            max(0, -nhood[e, 2]): min(shape[2], shape[2] - nhood[e, 2]),
                            ]
                            > 0
                    )
                    * (
                            seg[
                            max(0, nhood[e, 0]): min(shape[0], shape[0] + nhood[e, 0]),
                            max(0, nhood[e, 1]): min(shape[1], shape[1] + nhood[e, 1]),
                            max(0, nhood[e, 2]): min(shape[2], shape[2] + nhood[e, 2]),
                            ]
                            > 0
                    )
            )

    else:
        raise RuntimeError(f"AddAffinities works only in 2 or 3 dimensions, not {dims}")

    return aff


# function to erode label boundaries
def erode(labels, iterations, border_value):
    foreground = np.zeros_like(labels, dtype=bool)

    # loop through unique labels
    for label in np.unique(labels):

        # skip background
        if label == 0:
            continue

        # mask to label
        label_mask = labels == label

        # erode labels
        eroded_mask = binary_erosion(
            label_mask,
            iterations=iterations,
            border_value=border_value)

        # get foreground
        foreground = np.logical_or(eroded_mask, foreground)

    # and background...
    background = np.logical_not(foreground)

    # set eroded pixels to zero
    labels[background] = 0

    return labels


def split_zarr():
    source_ds_path = '/media/samia/DATA/ark/dan-samia/lsd/funke/otto/output-segmentations/octo-512/pred_otto_z7392-7904_y6586-7098_x5388-5900.zarr'

    target_ds_path = '/media/samia/DATA/ark/dan-samia/lsd/funke/otto/zarr/' \
                     'otto_z7392-7904_y6586-7098_x5388-5900.zarr'

    f_s = zarr.open(source_ds_path)
    # f_t = zarr.open(target_ds_path)
    # dataset path inside zarr
    # ds = 'volumes/labels/neuron_ids'
    ds = 'volumes/raw'

    raw_s = f_s[ds]
    # raw_t = f_t[ds]

    # find z-sections, python loads these arrays as zyx
    sections_s = raw_s.shape[0]
    # sections_t = raw_t.shape[0]

    # min_z = np.minimum(sections_t, sections_s)

    min_z = 776
    # find x and y width
    y_max_s = raw_s.shape[1]
    x_max_s = raw_s.shape[0]

    # y_max_t = raw_t.shape[1]
    # x_max_t = raw_t.shape[0]

    min_y = 776
    min_x = 776
    # min_y = np.minimum(y_max_s, y_max_t)
    # min_x = np.minimum(x_max_s, x_max_t)
    cast =True
    # for s in range(min_z // 2):
    for s in range(0, min_z):  # -128
        # for s in range(0, 512):
        raw_s_slice = raw_s[s, 128:min_y - 128, 128:min_x - 128]
        if cast:
            raw_s_slice = (raw_s_slice // 256).astype(np.uint8)
        print(s, np.unique(raw_s_slice))
        source_2d_path = f"{os.path.dirname(source_ds_path)}/2D-raw-png"
        # target_2d_path = f"{os.path.dirname(target_ds_path)}/2D-labels-png"
        if not os.path.exists(source_2d_path):
            os.makedirs(source_2d_path)

        # if not os.path.exists(target_2d_path):
        #     os.makedirs(target_2d_path)

        # relab, forward_map, inverse_map = segmentation.relabel_sequential(raw_s_slice)
        tifffile.imwrite(os.path.join(source_2d_path, f"img_{s}.png"), raw_s_slice)
        # tifffile.imwrite(os.path.join(target_2d_path, f"img_{s}.png"), raw_t_slice)

    print('Saving png slices complete!')


def create_lut(labels):
    max_label = np.max(labels)

    lut = np.random.randint(
        low=0,
        high=255,
        size=(int(max_label + 1), 3),
        dtype=np.uint64)

    lut = np.append(
        lut,
        np.zeros(
            (int(max_label + 1), 1),
            dtype=np.uint8) + 255,
        axis=1)

    lut[0] = 0
    colored_labels = lut[labels]

    return colored_labels


# @title utility  function to download / save data as zarr

def create_data_from_zarr(
        input_zarr,
        name,
        offset,
        resolution,
        sections=None,
        squeeze=True):
    in_f = zarr.open(input_zarr, 'r')

    raw = in_f['volumes/raw']
    labels = in_f['volumes/labels/neuron_ids']

    f = zarr.open(f"./{name}", 'a')

    if sections is None:
        sections = range(raw.shape[0] - 1)

    for i, r in enumerate(sections):

        print(f'Writing data for section {r}')

        raw_slice = raw[r:r + 1, :, :]
        labels_slice = labels[r:r + 1, :, :]

        if squeeze:
            raw_slice = np.squeeze(raw_slice)
            labels_slice = np.squeeze(labels_slice)

        f[f'raw/{i}'] = raw_slice
        f[f'labels/{i}'] = labels_slice

        f[f'raw/{i}'].attrs['offset'] = offset
        f[f'raw/{i}'].attrs['resolution'] = resolution

        f[f'labels/{i}'].attrs['offset'] = offset
        f[f'labels/{i}'].attrs['resolution'] = resolution


def create_affinities_from_labels():
    label_path = '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-octo/dataset/trainA_labels'
    files = glob(f"{label_path}/*.tif")
    out_path = '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-octo/dataset/trainA_affs'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for f in files:
        fr = io.imread(f)
        e_labels = erode(fr, iterations=1, border_value=1)
        aff = seg_to_affgraph(e_labels)
        tifffile.imwrite(os.path.join(out_path, os.path.basename(f)), aff)


if __name__ == '__main__':
    split_zarr()
    # source_ds_path = '/media/samia/DATA/ark/dan-samia/lsd/funke/hemi/training/zarr/' \
    #                  'eb-inner-groundtruth-with-context-x20172-y2322-z14332.zarr'
    # create_data_from_zarr(source_ds_path, '2d', (0,0,0), (8,8,8))
    # create_affinities_from_labels()
