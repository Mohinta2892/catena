from skimage.metrics import structural_similarity
import skimage
import numpy as np
from glob import glob
from skimage.color import rgb2gray
import cv2


def compare_ssim(img1, img2):
    img1_r = skimage.io.imread(img1)
    img2_r = skimage.io.imread(img2)

    # Convert images to grayscale when images are colored
    if len(img1_r.shape) > 2:
        img1_r_gray = rgb2gray(img1_r)
    else:
        img1_r_gray = img1_r
    if len(img2_r.shape) > 2:
        img2_r_gray = rgb2gray(img2_r)
    else:
        img2_r_gray = img2_r

    if img2_r_gray.shape != img1_r_gray.shape:
        # create patches of the smallest
        pass

    # Compute SSIM between two images
    (score, diff) = structural_similarity(img1_r_gray, img2_r_gray, full=True)
    print("Image similarity", score)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1]
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(img1_r.shape, dtype='uint8')
    filled_after = after.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img1_r, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
            cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

    cv2.imshow('img1_r', img1_r)
    cv2.imshow('after', after)
    cv2.imshow('diff', diff)
    cv2.imshow('mask', mask)
    cv2.imshow('filled after', filled_after)
    cv2.waitKey(0)


if __name__ == '__main__':
    # for now let's hard code inputs
    dir1 = "/media/samia/DATA/PhD/codebases/MitoEM/data/data_organised/Lucchi++/train/x"  # directory containing img1s'
    dir2 = "/media/samia/DATA/PhD/codebases/MitoEM/data/translation/experiment_hemi_to_kasthuri/trainA"  # directory containing img2s'

    # expected to be pngs/tifs/jpgs
    files_dir1 = sorted(glob(f"{dir1}/*.*"))  # sorting should work if images are appropriately number.
    files_dir2 = sorted(glob(f"{dir2}/*.*"))  # to be changed to a complex sort if does not

    # for paired comparison between images which are of the same object/location, numbers of them must match in each dir
    paired = False
    if paired:
        assert len(files_dir1) == len(
            files_dir2), "For paired image comparisons, numbers of images in each dir must be the same"

    # if unpaired let's just loop till the fewest number of images
    for i, j in zip(files_dir1, files_dir2):
        compare_ssim(i, j)
