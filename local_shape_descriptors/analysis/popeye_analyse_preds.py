"""
This script loads `Affinity` prediction results from zarrs and makes a `matplotlib` plot to visualise them.
Popeye: Is Pgmy Squid larva. Data owner: Ana Correia da Silva (Cardona Lab).
Popeye: Acquired in the MRC LMB at resolution 12 x 12 x 30nm.
"""

import zarr
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import seaborn as sns
from matplotlib.gridspec import GridSpec


def read_zarr(filename, dataset, mode='r'):
    # remember this is still an object
    return zarr.open(filename, mode=mode)[dataset]


def viz_valentin_preds(raw_zarr, pred_zarr, slice_num, filename='plt_affs.png'):
    raw = read_zarr(raw_zarr, dataset='volumes/raw')[14:512-14, 46:512-46, 46:512-46]
    preds = read_zarr(pred_zarr, dataset='pred_affs')

    if isinstance(slice_num, list):
        num_slices = len(slice_num)
        num_rows = num_slices  # Calculate the number of rows required

        fig = plt.figure(figsize=(20, 8 * num_rows))
        gs = GridSpec(num_rows, 4, figure=fig)

        for i, slice_idx in enumerate(slice_num):
            row = i

            ax_raw = fig.add_subplot(gs[row, 0])
            ax_raw.imshow(raw[slice_idx, ...], cmap='gray')
            # ax_raw.imshow(preds[0, slice_idx, ...], cmap='gray', alpha=0.5)

            ax_pred1 = fig.add_subplot(gs[row, 1])
            ax_pred1.imshow(preds[0, slice_idx, ...], cmap='gray')

            ax_pred2 = fig.add_subplot(gs[row, 2])
            ax_pred2.imshow(preds[1, slice_idx, ...], cmap='gray')

            ax_pred3 = fig.add_subplot(gs[row, 3])
            ax_pred3.imshow(preds[2, slice_idx, ...], cmap='gray')

            ax_pred2.set_title(f"Slice number: {slice_idx}", fontweight='bold', fontsize=10)

    path_info = filename.split(os.sep)[-2]  # one up model_checkpoint
    fig.suptitle(f'Prediction with: {path_info}', fontsize=16, fontweight='bold')

    sns.despine(fig, left=True, bottom=True)
    plt.tight_layout()

    plt.savefig(f"./{os.path.basename(filename)}_{path_info}.jpg", dpi=300)

    plt.show()


def viz_preds(raw, preds, slice_num, filename='plt_affs.png'):
    if isinstance(slice_num, int):
        fig, axs = plt.subplots(1, 4, figsize=(15, 15))
        # raw
        axs[0].imshow(raw[slice_num, 72:512, 72:512], cmap='gray')
        # aff
        axs[1].imshow(preds[0, slice_num, 72:512, 72:512], cmap='gray')
        axs[2].imshow(preds[1, slice_num, 72:512, 72:512], cmap='gray')
        axs[3].imshow(preds[2, slice_num, 72:512, 72:512], cmap='gray')
    elif isinstance(slice_num, list):
        num_slices = len(slice_num)
        num_rows = num_slices  # Calculate the number of rows required
        fig, axs = plt.subplots(num_rows, 4, figsize=(20, 6 * num_rows))
        for i, slice_idx in enumerate(slice_num):
            # row = i // 4  # Calculate the row index for the subplot
            # col = i % 4  # Calculate the column index for the subplot
            axs[i, 0].imshow(raw[slice_idx, 72:512, 72:512], cmap='gray')
            # aff
            axs[i, 1].imshow(preds[0, slice_idx, 72:512, 72:512], cmap='gray')
            axs[i, 2].imshow(preds[1, slice_idx, 72:512, 72:512], cmap='gray')
            axs[i, 3].imshow(preds[2, slice_idx, 72:512, 72:512], cmap='gray')
            axs[i, 2].set_title(f"Slice number: {slice_idx}", fontweight='bold', fontsize=10)

    path_info = filename.split(os.sep)[-2]  # one up model_checkpoint
    fig.suptitle(f'Prediction with: {path_info}', fontsize=16, fontweight='bold')

    sns.despine(fig, left=True, bottom=True)
    plt.tight_layout()

    plt.savefig(f"./{os.path.basename(filename)}_{path_info}.jpg", dpi=300)

    plt.show()


def main(args):
    dataset_arrs = {}

    if args.z2 is None:
        for ds in args.ds:
            dataset_arrs[ds] = read_zarr(args.z, ds)

        viz_preds(dataset_arrs[args.ds[0]], dataset_arrs[args.ds[1]], slice_num=args.snum, filename=args.z)
    else:
        viz_valentin_preds(args.z, args.z2, slice_num=args.snum, filename=args.z)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-z',
                        default='/media/samia/DATA/ark/dan-samia/lsd/funke/squid/output_segmentations/valentin/popeye/popeye__setup_03_0_2_FBOK_600000_0/popeye2_z4500-5300_y11700-14300_x2978-5578_z0-512_y0-512_x0-512_uint8_CLAHE.zarr',
                        help="Please pass the zarr file.")
    parser.add_argument('-z2',
                        default='/media/samia/DATA/ark/dan-samia/lsd/funke/squid/output_segmentations/valentin/popeye/popeye__setup_03_0_2_FBOK_600000_0/popeye__setup_03_0_2_FBOK_600000_0_pred.zarr',
                        help="Another zarr file if needed.")
    parser.add_argument('-ds', default=['volumes/raw', 'volumes/pred_affs'],
                        help='Zarr file datasets.')
    parser.add_argument('-snum', default=[100, 200, 255, 300], help="specify the z-slice nums you want to plot")
    args = parser.parse_args()

    main(args)
