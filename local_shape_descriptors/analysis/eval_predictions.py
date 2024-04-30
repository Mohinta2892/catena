import numpy as np
import skimage
import zarr
import argparse
import os
import matplotlib.pyplot as plt


def evaluate_segmentation(seg, gt, threshold=None):
    # skimage.metrics.variation_of_information/adapted_rand_error
    splits, merges = skimage.metrics.variation_of_information(gt, seg, ignore_labels=(0,))
    error, precision, recall = skimage.metrics.adapted_rand_error(gt, seg, ignore_labels=(0,))

    if threshold is not None:
        print(
            f'Segmentation threshold used to generated over-seg: {float(os.path.basename(threshold).split("_")[-1])}%')
    print(f'Adapted Rand error: {round(error, 6)}')
    print(f'Adapted Rand precision: {round(precision, 6)}')
    print(f'Adapted Rand recall: {round(recall, 6)}')
    print(f'False Splits: {round(splits, 6)}')
    print(f'False Merges: {round(merges, 6)}')


def plot_segmentation_slices(seg, raw, aff=None):
    fig, axs = plt.subplots(2, 3, figsize=(10,10))
    # seg = f["volumes/segmentation_0.55"][...]
    # raw = f["volumes/raw"][...]
    # aff = f["volumes/pred_affs"][...]
    axs[0,0].imshow(raw[10, ...], cmap='gray')
    axs[0,1].imshow(aff[10, ...], cmap='gray')
    axs[0,2].imshow(seg[10, ...], cmap='prism')

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', default='/media/samia/DATA/ark/dan-samia/lsd/funke/otto/tiff/octo_z7392-7904_y6586-7098_x5388-5900_z120-140_y100-300_x100-300_wgt.zarr',
                        help='Pass the gt zarr file')

    parser.add_argument('-gt_ds', default='volumes/labels/neuron_ids', help='Pass the dataset in gt zarr file')
    parser.add_argument('-seg', default='/media/samia/DATA/ark/dan-samia/lsd/funke/otto/tiff/octo_z7392-7904_y6586-7098_x5388-5900_z120-140_y100-300_x100-300_wgt.zarr',
                        help='Pass the predicted seg zarr file')
    parser.add_argument('-seg_ds', default='volumes/segmentation_0.65',
                        help='Pass the dataset in seg zarr file')
    parser.add_argument('-aff', default='/media/samia/DATA/ark/dan-samia/lsd/funke/otto/tiff/octo_z7392-7904_y6586-7098_x5388-5900_z120-140_y100-300_x100-300_wgt.zarr', help='Pass the dataset in affinity zarr file')
    parser.add_argument('-aff_ds', default='volumes/pred_affs', help='Pass the dataset in affinity zarr file')

    args = parser.parse_args()

    seg = zarr.open(args.seg)[args.seg_ds][...]
    gt = zarr.open(args.gt)[args.gt_ds][...]
    aff = zarr.open(args.aff)[args.aff_ds]
    raw = zarr.open(args.gt)["volumes/raw"]
    # threshold = args.seg_ds[args.split('_')[-1]

    evaluate_segmentation(seg=seg, gt=gt, threshold=args.seg_ds)

    plot_segmentation_slices(seg=seg, aff=aff, raw=raw)


if __name__ == '__main__':
    main()
