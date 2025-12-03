import os
import numpy as np
from PIL import Image


def find_nans_in_images(directory):
    nan_images = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            file_path = os.path.join(directory, filename)
            try:
                with Image.open(file_path) as img:
                    img_data = np.array(img)
                    if np.isnan(img_data).sum():
                        nan_images.append(filename)
            except Exception as e:
                print(f"Could not process image {filename}: {e}")
    return nan_images


directory = '/media/samia/DATA/mounts/cephfs/img2img-turbo/data/2d_pngs/pb-groundtruth-with-context-x8472-y2892-z9372_clahed/neuron_ids'  # Replace with your directory
nan_images = find_nans_in_images(directory)

if nan_images:
    print("Images containing NaNs:")
    for img in nan_images:
        print(img)
else:
    print("No NaNs found in the images.")
