import matplotlib.pyplot as plt
import numpy as np
import zarr
from PIL import Image
import seaborn as sns


# Function to calculate contrast
def calculate_contrast(image):
    return image.max() - image.min()


# # hemi
# volume1 = "/media/samia/DATA/ark/connexion/data/HEMI/data_3d/train/eb-inner-groundtruth-with-context-x20172-y2322-z14332_clahed.zarr"
# # octo
# volume2 = "/media/samia/DATA/ark/connexion/data/MITO-HEMI/data_3d/test/otto_z7392-7904_y6586-7098_x5388-5900_clahed.zarr"
# # tremont
# volume3 = "/media/samia/DATA/ark/connexion/data/TREMONT/data_3d/train/g2019s_x8250_y7250_z650_588x588x196_clahed.zarr"

# # hemi
volume1 = "/media/samia/DATA/ark/connexion/data/HEMI/data_3d/original_hemi/eb-inner-groundtruth-with-context-x20172-y2322-z14332.zarr"
# octo
volume2 = "/media/samia/DATA/ark/connexion/data/preprocessed_3d/TEST-OCTO_s_t_TEST-HEMI/otto_z7392-7904_y6586-7098_x5388-5900.zarr"
# tremont
volume3 = "/media/samia/DATA/ark/connexion/data/preprocessed_3d/TEST-TREMONT_s_t_TEST-HEMI/g2019s_x8250_y7250_z650_588x588x196.zarr"


# List of image file paths
# image_paths_hemi = [
#     "/media/samia/DATA/ark/code-experiments/img2img-turbo/data/em_hemi_octo/test_A/img_647.png",
#     "/media/samia/DATA/ark/code-experiments/img2img-turbo/data/em_hemi_octo/test_A/img_646.png",
#     "/media/samia/DATA/ark/code-experiments/img2img-turbo/data/em_hemi_octo/test_A/img_645.png",
#     "/media/samia/DATA/ark/code-experiments/img2img-turbo/data/em_hemi_octo/test_A/img_644.png",
#     "/media/samia/DATA/ark/code-experiments/img2img-turbo/data/em_hemi_octo/test_A/img_643.png",
#     "/media/samia/DATA/ark/code-experiments/img2img-turbo/data/em_hemi_octo/test_A/img_642.png"
# ]
#
# # List of image file paths
# image_paths_octo = [
#     "/media/samia/DATA/ark/code-experiments/img2img-turbo/data/em_hemi_octo/test_B/img_461.png",
#     "/media/samia/DATA/ark/code-experiments/img2img-turbo/data/em_hemi_octo/test_B/img_462.png",
#     "/media/samia/DATA/ark/code-experiments/img2img-turbo/data/em_hemi_octo/test_B/img_463.png",
#     "/media/samia/DATA/ark/code-experiments/img2img-turbo/data/em_hemi_octo/test_B/img_464.png",
#     "/media/samia/DATA/ark/code-experiments/img2img-turbo/data/em_hemi_octo/test_B/img_465.png",
#     "/media/samia/DATA/ark/code-experiments/img2img-turbo/data/em_hemi_octo/test_B/img_466.png"
# ]
#
# image_paths_tremont = [
#     "/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-tremont/dataset/pngs/size_256/testB/img_1.png",
#     "/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-tremont/dataset/pngs/size_256/testB/img_3.png",
#     "/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-tremont/dataset/pngs/size_256/testB/img_4.png",
#     "/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-tremont/dataset/pngs/size_256/testB/img_5.png",
#     "/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-tremont/dataset/pngs/size_256/testB/img_6.png",
#     "/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-tremont/dataset/pngs/size_256/testB/img_7.png"
# ]

# Load images and stack them into a 3D array (volume)
# images = [np.array(Image.open(p)) for p in image_paths_hemi]
# volume1 = np.stack(images, axis=0)
#
# np.savez("./hemi_stack.npz", volume1)
#
# images = [np.array(Image.open(p)) for p in image_paths_octo]
# volume2 = np.stack(images, axis=0)
# np.savez("./octo_stack.npz", volume2)
#
# images = [np.array(Image.open(p)) for p in image_paths_tremont]
# volume3 = np.stack(images, axis=0)
# np.savez("./tremont_stack.npz", volume2)

# [:, :100, :100] gives back a flatter plot, but using full data gives the best flat graphs
volume1 = zarr.open(volume1)["volumes/raw"][...]
volume2 = zarr.open(volume2)["volumes/raw"][...]  # [:, :100, :100]
volume3 = zarr.open(volume3)["volumes/raw"][...]  # [:, :100, :100]

volumes = [volume1, volume2, volume3]
vol_names = ["Hemi-Brain", "Octo", "Tremont"]

# Plot intensity histograms
plt.figure(figsize=(15, 6))
for i, volume in enumerate(volumes):
    plt.subplot(1, len(volumes), i + 1)
    plt.hist(volume.ravel(), bins=256, color='gray', alpha=0.7)
    plt.title(f'{vol_names[i]}')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
sns.despine()
plt.suptitle("Intensity Histogram")
plt.tight_layout()
# plt.savefig("./intensity_hist.png", dpi=300)
plt.show()

# Plot contrast plots
plt.figure(figsize=(8, 6))
for i, volume in enumerate(volumes):
    contrasts = [calculate_contrast(slice) for slice in volume]
    plt.plot(contrasts, label=f'{vol_names[i]}')
plt.title('Contrast')
plt.xlabel('Image Slice')
plt.ylabel('Contrast')
plt.legend()
sns.despine()
plt.grid(True)
plt.tight_layout()
# plt.savefig("./contrast.png", dpi=300)
plt.show()
