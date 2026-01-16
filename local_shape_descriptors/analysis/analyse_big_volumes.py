import zarr
import numpy as np
import napari


def viz_segmentation(viewer, zfile, ds):
    if 'raw' in ds:
        viewer.add_image(zfile[ds])
    elif 'seg' in ds:
        viewer.add_labels(zfile[ds])
    else:
        viewer.add_image(zfile[ds])


def main():
    # todo expand to argparse
    file = "/media/samia/DATA/ark/lsd_outputs/AFF/3d/run-aclsd-together/segmented/octo-AL-crop-Albrt.zarr"
    raw_ds = "volumes/raw"
    seg_ds = "volumes/segmentation_055"
    aff_ds = "volumes/pred_affs"

    # load zarr
    zfile = zarr.open(file, mode='r')

    v = napari.Viewer()
    viz_segmentation(viewer=v, zfile=zfile, ds=raw_ds)
    viz_segmentation(viewer=v, zfile=zfile, ds=seg_ds)

    napari.run()


if __name__ == '__main__':
    main()
