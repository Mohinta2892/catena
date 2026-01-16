import argparse
import os.path

import numpy as np
import skimage
import zarr


def __grow(gt, gt_mask=None, only_xy=False, background=0):
    """
    Copied from Gunpowder!!

    :param gt:
    :param gt_mask:
    :param only_xy:
    :return:
    """
    if gt_mask is not None:
        assert (
                gt.shape == gt_mask.shape
        ), "GT_LABELS and GT_MASK do not have the same size."

    if only_xy:
        assert len(gt.shape) == 3
        for z in range(gt.shape[0]):
            __grow(gt[z], None if gt_mask is None else gt_mask[z])
        return

    # get all foreground voxels by erosion of each component
    foreground = np.zeros(shape=gt.shape, dtype=bool)
    masked = None
    if gt_mask is not None:
        masked = np.equal(gt_mask, 0)
    for label in np.unique(gt):
        if label == background:
            continue
        label_mask = gt == label
        # Assume that masked out values are the same as the label we are
        # eroding in this iteration. This ensures that at the boundary to
        # a masked region the value blob is not shrinking.
        if masked is not None:
            label_mask = np.logical_or(label_mask, masked)
        eroded_label_mask = ndimage.binary_erosion(
            label_mask, iterations=steps, border_value=1
        )
        foreground = np.logical_or(eroded_label_mask, foreground)

    # label new background
    background = np.logical_not(foreground)
    gt[background] = background

    return gt


def parse_slice(s):
    start, stop = map(int, s.split(':'))
    return slice(start, stop)


def create_mask_from_seg(labels):
    labels_mask = np.ones_like(labels)
    background = labels = 0
    labels_mask[background] = 0

    return labels_mask


def seg_to_affgraph(seg, nhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]]):
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
    print(f"aff values: {np.unique(aff)}")
    return aff


def seg_to_aff(labels, labels_mask, only_xy=False, background=0):
    eroded_labels = __grow(labels, labels_mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help='Segmentation Zarr File')
    parser.add_argument("-ds", default="volume", help='Segmentation Zarr File')
    parser.add_argument("-res", nargs='+', type=int, help='Segmentation Zarr File')
    parser.add_argument("-offset", nargs='+', type=int, help='Segmentation Zarr File')
    parser.add_argument("-roi", default="1024:1500, 1024:1500, 1024:1500", help="Roi dimensions to crop in ZYX")
    parser.add_argument("-of", default="/media/samia/DATA/volumes/Leo_FFN_Crop", help="Output Affinity Zarr File")

    args = parser.parse_args()
    roi_slices = args.roi.replace(',', ' ').split()
    slices = [parse_slice(dim_slice) for dim_slice in roi_slices]

    file_ = zarr.open(args.f)
    labels = file_[args.ds]

    labels = labels[tuple(slices)]  # cropped segmentations
    labels_mask = create_mask_from_seg(labels)

    # watershed only works with float32
    affinities = seg_to_affgraph(labels).astype(dtype=np.float32)
    affinities_mask = seg_to_affgraph(labels_mask).astype(dtype=np.float32)

    # Construct the out_filename
    filename = os.path.splitext(os.path.basename(args.f))[0]
    filename += f"_z{roi_slices[0].replace(':', '_')}"
    filename += f"_y{roi_slices[1].replace(':', '_')}"
    filename += f"_x{roi_slices[2].replace(':', '_')}"

    # save the affinities
    out_file_path = os.path.join(args.of, f"{filename}.zarr")

    out_file = zarr.open(out_file_path, "a")
    out_file["volumes/pred_affs"] = affinities
    out_file["volumes/pred_affs_mask"] = affinities_mask

    res = tuple(args.res)
    offset = tuple(args.offset)
    out_file["volumes/pred_affs"].attrs["resolution"] = res
    out_file["volumes/pred_affs"].attrs["offset"] = offset

    out_file["volumes/pred_affs_mask"].attrs["resolution"] = res
    out_file["volumes/pred_affs_mask"].attrs["offset"] = offset


if __name__ == '__main__':
    main()
