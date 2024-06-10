"""
This script plots presyn/postsyn locations over raw EM with napari.
Please pass datasets (e.g. volumes/labels/clefts, volumes/labels/neuron_ids) to plot clefts/segmentation.
Followed napari tutorial: https://github.com/adamltyson/napari-tutorials/blob/master/tutorials/points.md

"""
import h5py
import zarr
import numpy as np
import napari
import argparse

def load_zarr(inputfilename, dataset):
    f = zarr.open(inputfilename, 'r')
    offset = (0, 0, 0)
    if dataset in f:
        data = f[dataset][:]
        if 'offset' in f[dataset].attrs.keys():
            offset = f[dataset].attrs['offset']
        print(dataset, data.shape, data.dtype)
    else:
        data = None
        print(dataset, 'does not exist')
    # f.close()
    return data, offset


def load_hdf5(inputfilename, dataset):
    f = h5py.File(inputfilename, 'r')
    offset = (0, 0, 0)
    if dataset in f:
        data = f[dataset][:]
        if 'offset' in f[dataset].attrs.keys():
            offset = f[dataset].attrs['offset']
        print(dataset, data.shape, data.dtype)
    else:
        data = None
        print(dataset, 'does not exist')
    f.close()
    return data, offset


def parse_nargs(in_str):
    # Custom function to parse a resolution string into a tuple of integers

    try:
        if isinstance(in_str, list) and len(in_str) == 1:
            in_str = in_str[0]
        parts = in_str.split(',')
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("Resolution must contain exactly three integers separated by commas.")
        return tuple(int(part.strip()) for part in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("Resolution must contain only integers separated by commas.")


def plot_syn(filename, res=(40, 4, 4)):
    raw, _ = load_hdf5(filename, "volumes/raw")

    cleft, cleft_offset = load_hdf5(filename, "volumes/labels/clefts")

    locs, offsets = load_hdf5(filename, 'annotations/locations')

    locs = [loc + np.array(offsets, dtype=np.float32) for loc in locs]
    # locs = [loc + offset for loc in locs]
    print(locs)

    partners, offset = load_hdf5(filename, 'annotations/presynaptic_site/partners')

    annotation_ids, offset = load_hdf5(filename, 'annotations/ids')

    (pre_sites, post_sites, connectors) = ([], [], [])
    for (pre, post) in partners:
        pre_index = int(np.where(pre == annotation_ids)[0][0])
        post_index = int(np.where(post == annotation_ids)[0][0])
        # print(pre_index, post_index)
        pre_site = locs[pre_index]
        post_site = locs[post_index]

        # convert presyn point annotations from nm to pixels
        pre_temp = []
        for p, r in zip(pre_site, res):
            p = p / r
            pre_temp.append(p)
        pre_sites.append(np.array(pre_temp))

        # convert postsyn point annotations from nm to pixels
        post_temp = []
        for p, r in zip(post_site, res):
            p = p / r
            post_temp.append(p)
        post_sites.append(np.array(post_temp))

        connectors.append([pre_temp, post_temp])
        print(connectors)

    v = napari.Viewer()

    v.add_image(raw, name="raw")

    if cleft is not None:
        v.add_labels(cleft, name="cleft", opacity=0.7, blending="additive")

    v.add_points(pre_sites, name="pre_syn", symbol='triangle_down', face_color="red", size=20)
    v.add_points(post_sites, name="post_syn", symbol='star', face_color="blue", size=20)
    v.add_shapes(connectors, shape_type='line', edge_color='cyan', face_color='cyan', edge_width=5)
    napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt',
                        default='/Users/sam/Downloads/conn_159456.hdf5',
                        help='Pass the Ground truth h5py file')
    parser.add_argument('-res', nargs='+',
                        default='40,4,4',
                        help='Pass the imaging resolution of this volume')

    args = parser.parse_args()

    plot_syn(args.gt, res=parse_nargs(args.res))


if __name__ == '__main__':
    main()
