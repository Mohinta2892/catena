import os.path

import skimage.io as io
import numpy as np
import tifffile
from skimage.transform import resize
from glob import glob


def resize_labels(filenames, out_f, resize_shape=(256, 256)):
    if not os.path.exists(out_f):
        os.makedirs(out_f, exist_ok=True)

    for f in filenames:
        read_f = io.imread(f, dtype=np.uint64)
        resize_f = resize(read_f, resize_shape, preserve_range=True, anti_aliasing=True, clip=True)
        print(resize_f.astype(np.uint64), resize_f)
        tifffile.imwrite(os.path.join(out_f, os.path.basename(f)), resize_f.astype(np.uint64))


if __name__ == '__main__':
    resize_labels(sorted(glob(
        '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-octo/dataset/tifs/trainA_labels/*tif')),
        '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-octo/dataset/tifs/trainA_labels_resized256',
        resize_shape=(256, 256))
