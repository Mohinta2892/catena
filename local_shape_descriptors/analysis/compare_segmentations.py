import glob
import os.path
from pathlib import Path

import zarr
import napari
import tifffile as t
import numpy as np
import argparse
import sys
import kimimaro  # use this to skeletonise
import skimage
import navis
from typing import Union
import matplotlib.pyplot as plt

# global viewer
v = napari.Viewer()


def parse_neuron_list(neuron_list):
    try:
        return [int(x.strip()) for x in neuron_list.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid comma-separated list of integers")


def calc_center_of_mass():
    pass


def count_slices_with_ids(overseg_f, corr_f, neuron_list):
    # count_slices_overseg = dict()
    # count_slices_corr = dict()
    #
    # for id in neuron_list:
    #     count_slices_overseg[]
    pass


def skeletonise_kimimaro(labels, neuron_list):
    skels = kimimaro.skeletonize(
        labels,  # ndarray
        teasar_params={  # keep defaults for now!
            "scale": 1.5,
            "const": 300,  # physical units
            "pdrf_scale": 100000,
            "pdrf_exponent": 4,
            "soma_acceptance_threshold": 3500,  # physical units in nm
            "soma_detection_threshold": 750,  # physical units
            "soma_invalidation_const": 300,  # physical units
            "soma_invalidation_scale": 2,
            "max_paths": 300,  # default None
        },
        object_ids=neuron_list,  # process only the specified labels
        # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
        # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
        dust_threshold=1000,  # skip connected components with fewer than this many voxels
        anisotropy=(8, 8, 8),  # default True
        fix_branching=True,  # default True
        fix_borders=True,  # default True
        fill_holes=False,  # default False
        fix_avocados=False,  # default False
        progress=True,  # default False, show progress bar
        parallel=3,  # <= 0 all cpu, 1 single process, 2+ multiprocess
        parallel_chunk_size=100,  # how many skeletons to process before updating progress bar
    )

    # skels are ideally dicts!!
    print("Skeletons")
    print(skels)

    return skels


def are_equivalent(label_a: np.ndarray, label_b: np.ndarray) -> bool:
    """
    Copied from: https://github.com/google-research/connectomics/blob/main/connectomics/segmentation/labels.py
    Credits: Google Connectomics Group

    Returns whether two volumes contain equivalent segmentations.

    Segmentations are considered equivalent when there exists a 1:1 map between
    the two arrays -- that is segment shapes and locations have to be exactly
    the same but their IDs can be shuffled.

    Args:
      label_a: numpy array of segments
      label_b: numpy array of segments

    Returns:
      True iff the segmentations 'label_a' and 'label_b' are equivalent.
    """
    if label_a.shape != label_b.shape:
        return False

    a_to_b = {}
    b_to_a = {}

    for a, b in set(zip(label_a.flat, label_b.flat)):
        if a not in a_to_b:
            a_to_b[a] = b
        if b not in b_to_a:
            b_to_a[b] = a

        if a_to_b[a] != b:
            return False

        if b_to_a[b] != a:
            return False

    return True


def gen_skel_viz(zarr_file, dataset, np_file, neuron_list=None, viz_napari=False, skeletonise=False):
    if not viz_napari:
        # close the napari viewer here!
        v.close()

    file_ = zarr.open(zarr_file, mode='r')
    overseg_f = file_[dataset][...]  # load the array???
    corr_f = np.load(np_file, mmap_mode='r')  # load a memmap to save memory!!

    # seg2link corrections need transpose
    corr_f = corr_f.transpose((2, 0, 1))

    # sanity check for RAM, print shape
    print(f'Shape of original segmentation {overseg_f.shape}')
    print(f'Shape of corrected segmentation {corr_f.shape}')

    # This is default. Overwritten by following conditionals!
    neuron_ids = np.arange(0, 10)  # load the background=0, plus 1-9 labels.
    if neuron_list is not None:
        neuron_ids = parse_neuron_list(neuron_list)
    else:
        # we must pick all ids from the oversegmented volume and subtract the uniques in corrected volumes.
        # comes with the problem that the segmented image must fit into RAM, so be careful.
        neuron_ids = np.unique(overseg_f)

    # should we check the intersection of neuron ids across the original and the corrected vols?
    neuron_ids_overseg = set(np.unique(overseg_f))
    neurons_ids_corr = set(np.unique(corr_f))
    print(f"Which ids np longer exist in corrected vol: {neuron_ids_overseg - neurons_ids_corr}")
    print(f"Selected ids : {neuron_ids}")
    print(f"neuron_ids in overseg: {np.sum(np.isin(neuron_ids, overseg_f))}")
    print(f"neuron ids in corr: {np.sum(np.isin(neuron_ids, corr_f))}")

    # if viz_napari:
    # pick these labels from the oversegmented array
    picked_overseg_f = overseg_f * np.isin(overseg_f, neuron_ids)  # original * mask
    picked_corr_f = corr_f * np.isin(corr_f, neuron_ids)  # corrected * mask

    out_folder = str(Path(__file__).resolve().cwd())
    print

    if skeletonise:
        # skeletonise corrected/overseg files
        skels_corr_f = skeletonise_kimimaro(corr_f, neuron_ids)
        skels_overseg_f = skeletonise_kimimaro(overseg_f, neuron_ids)

        # make the out dir
        # TODO: make a function
        out_folder_corr = os.path.join(out_folder, os.path.basename(dataset),
                                       os.path.basename(np_file).replace('.npy', ''))
        print(f"Saving corrected swc files here: {out_folder_corr}")
        if not os.path.exists(out_folder_corr):
            os.makedirs(out_folder_corr)

        # save corrected skeletons
        for label, skel in skels_corr_f.items():
            fname = os.path.join(out_folder_corr, f"{label}.swc")
            with open(fname, "wt") as f:
                f.write(skel.to_swc())

        # make the out dir. NB: same variable.
        # TODO: make a function
        out_folder_overseg = os.path.join(out_folder, os.path.basename(dataset),
                                          os.path.basename(zarr_file).replace('.zarr', ''))
        print(f"Saving over-segmented swc files here: {out_folder_overseg}")
        if not os.path.exists(out_folder_overseg):
            os.makedirs(out_folder_overseg)

        for label, skel in skels_overseg_f.items():
            fname = os.path.join(out_folder_overseg, f"{label}.swc")
            with open(fname, "wt") as f:
                f.write(skel.to_swc())

    if args.v:
        # add the raw for context
        v.add_image(file_["volumes/raw"], blending='additive',
                    opacity=0.5)  # must exist like this; TODO make it an arg.
        v.add_labels(picked_overseg_f)
        v.add_labels(picked_corr_f)

    return out_folder, picked_overseg_f, picked_corr_f  # one above


def evaluate_segmentation(picked_overseg_f, picked_corr_f, threshold=None):
    # skimage.metrics.variation_of_information/adapted_rand_error
    splits, merges = skimage.metrics.variation_of_information(picked_corr_f, picked_overseg_f, ignore_labels=(0,))
    error, precision, recall = skimage.metrics.adapted_rand_error(picked_corr_f, picked_overseg_f, ignore_labels=(0,))

    if threshold is not None:
        print(
            f'Segmentation threshold used to generated over-seg: {int(os.path.basename(threshold).split("_")[-1])}%')
    print(f'Adapted Rand error: {round(error, 6)}')
    print(f'Adapted Rand precision: {round(precision, 6)}')
    print(f'Adapted Rand recall: {round(recall, 6)}')
    print(f'False Splits: {round(splits,6)}')
    print(f'False Merges: {round(merges,6)}')


def evaluate_skeletons_navis(skel_folder, picked_overseg_f):
    # expand this to intake provided folder paths!
    sub_folders = glob.glob(f"{skel_folder}/*/", recursive=True)
    # this is corrected one
    corr_sub = [element for element in sub_folders if 'modified' in element]
    overseg_sub = [element for element in sub_folders if 'modified' not in element]

    corr_skels = navis.read_swc(corr_sub)
    overseg_skels = navis.read_swc(overseg_sub)

    print(f"corrected skeletons of neurons {corr_skels[0]}")
    print(f"oversegmented skeletons of neurons {overseg_skels[0]}")

    # compare the cable lengths
    mis_match_corr = []
    mis_match_overseg = []
    for x, y in zip(overseg_skels, corr_skels):
        # print(f"overseg id {x.name}, cable length: {x.cable_length}")
        # print(f"corr id {y.name}, cable length: {y.cable_length}")
        if x.name == y.name and x.cable_length != y.cable_length:
            print(f"id {x.name}, overseg cable length: {x.cable_length}, corr cable length: {y.cable_length}")
            mis_match_corr.append(y)
            mis_match_overseg.append(x)

    # fig = mis_match_corr[0].plot3d(color='red', hover_name=True, backend='plotly')
    # mis_match_overseg[0].plot3d(color='green', hover_name=True, backend='plotly')
    # navis.plot3d(picked_overseg_f, hover_name=True, backend='plotly')
    # # for i, j in zip(mis_match_corr[1:], mis_match_overseg[1:]):
    # #     i.plot3d()
    #
    # fig.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-z',
                        default='/media/samia/DATA/ark/lsd_outputs/AFF/3d/run-aclsd-together/'
                                'segmented/crop_A1_z16655-17216_y13231-13903_x7650-8468.zarr',
                        help="Please pass the zarr file.")
    parser.add_argument('-ds', default='volumes/segmentation_055', help='Zarr file segmentation dataset.')
    parser.add_argument('-v', default=False, help='Want to display in napari? Default: False.')
    parser.add_argument('-rs', default=False, help='Regenerate skels? Default: False.')
    parser.add_argument('-n', default='/home/samia/Downloads/seg-modified-2024-Mar-13-PM04-30.npy',
                        help="Please pass the downloaded numpy file from Seg2Link file.")
    parser.add_argument('-nl', default='1,2,3,4,5,6,7,8,9, 336, 319,44, 1064,88,55,84,79,175,225,158,178,61,'
                                       '49,74,50,11,29', help='Neuron IDs in format `ID1,ID2,ID3`.'
                                                              ' This would be colour numbers!'
                                                              ' Default: None, means all neurons.')
    parser.add_argument('-c', default=None, help='Neuron IDs in csv.'
                                                 ' This would be colour numbers! Default: None, means all neurons.')
    args = parser.parse_args()
    if args.c is not None:
        args.nl = None  # turn this off if csv is passed for now!!

    # generates skels and visualises in `Napari`
    output_skel_folder, picked_overseg_f, picked_corr_f = gen_skel_viz(args.z, args.ds, args.n, args.nl,
                                                                       viz_napari=args.v,
                                                                       skeletonise=args.rs)

    # evaluates generated skeletons with `naVIS`
    # evaluate_skeletons_navis(output_skel_folder, picked_overseg_f)

    # this will work if the threshold passed was the one corrected too and there we no errors?
    print(f'Are the segmentations equivalent: {are_equivalent(picked_overseg_f_f, picked_corr_f)}')
    # evaluate labels
    evaluate_segmentation(picked_overseg_f, picked_corr_f, threshold=args.ds)
    if args.v:
        napari.run()
