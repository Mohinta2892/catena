import zarr
import numpy as np
from sklearn.linear_model import LinearRegression
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from joypy import joyplot
from PIL import Image


# hemi
# reference = "/media/samia/DATA/ark/connexion/data/HEMI/data_3d/original_hemi/eb-inner-groundtruth-with-context-x20172-y2322-z14332.zarr"
# vol1 = zarr.open(reference)["volumes/raw"][200:300, 200:300, 200:300]
#
# reference_clahed = "/media/samia/DATA/ark/connexion/data/HEMI/data_3d/clahed_hemi/eb-inner-groundtruth-with-context-x20172-y2322-z14332_clahed.zarr"
# vol11 = zarr.open(reference_clahed)["volumes/raw"][200:300, 200:300, 200:300]
# # octo
# octo = "/media/samia/DATA/ark/connexion/data/TEST-OCTO/data_3d/train/otto_z7392-7904_y6586-7098_x5388-5900.zarr"
# vol2 = zarr.open(octo)["volumes/raw"][200:300, 200:300, 200:300]
#
# octo_clahed = "/media/samia/DATA/ark/connexion/data/TEST-OCTO/data_3d/train/otto_z7392-7904_y6586-7098_x5388-5900_clahed.zarr"
# vol21 = zarr.open(octo_clahed)["volumes/raw"][200:300, 200:300, 200:300]
#
# hemi_match_octo = "/media/samia/DATA/ark/connexion/data/preprocessed_3d/TEST-HEMI_s_t_TEST-OCTO/eb-inner-groundtruth-with-context-x20172-y2322-z14332.zarr"
# vol3 = zarr.open(hemi_match_octo)["volumes/raw"][200:300, 200:300, 200:300]
#
# octo_match_hemi = "/media/samia/DATA/ark/connexion/data/preprocessed_3d/TEST-OCTO_s_t_TEST-HEMI/otto_z7392-7904_y6586-7098_x5388-5900.zarr"
# vol4 = zarr.open(octo_match_hemi)["volumes/raw"][200:300, 200:300, 200:300]
#
# volumes = [vol1, vol11, vol3, vol2, vol21, vol4]
# vol_names = ["Hemi-Brain", "Hemi-Brain-Clahed", "Hemi-to-Octo", "Octo", "Octo-Clahed", "Octo-to-Hemi"]

# images
image_paths = ['/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/pytorch-CycleGAN-and-pix2pix/results/hemi2octo/test_latest/real_A/img_128_real_A.png',
               '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/pytorch-CycleGAN-and-pix2pix/results/hemi2octo/test_latest/fake_B/img_128_fake_B.png',
               '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/pytorch-CycleGAN-and-pix2pix/results/hemi2octo/test_latest/real_B/img_128_real_B.png',
               '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/pytorch-CycleGAN-and-pix2pix/results/hemi2octo/test_latest/fake_A/img_128_fake_A.png']
vol_names = ["Hemi-Brain", "Fake Hemi-Brain", "Octo", "Fake Octo"]

# Prepare data for ridgeline plot
data = []
for path, name in zip(image_paths, vol_names):
    image = Image.open(path).convert('L')  # Convert to grayscale
    flattened = np.array(image).ravel()
    df = pd.DataFrame({'Intensity': flattened, 'Image': name})
    data.append(df)
data = pd.concat(data)

# Plot horizontal ridgeline plot
fig, axes = joyplot(
    data=data,
    by='Image',
    column='Intensity',
    kind='kde',
    fill=True,
    overlap=0.1,  # Reduce overlap to decrease spacing
    fade=False,    # Smooth transition between ridges
    linecolor='white',
    xlabelsize=30,
    ylabelsize=30,
    x_range=np.arange(-1, 256),
    linewidth=1,
    figsize=(10, 10),
    color='gray',
    alpha=0.7
)


# # Plot intensity histograms
# plt.figure(figsize=(15, 6))
# for i, volume in enumerate(volumes):
#     plt.subplot(1, len(volumes), i + 1)
#     plt.hist(volume.ravel(), bins=256, color='gray', alpha=0.7)
#     plt.title(f'{vol_names[i]}')
#     plt.xlabel('Intensity')
#     plt.ylabel('Frequency')
# sns.despine()
# plt.suptitle("Intensity Histogram")
# plt.tight_layout()
# # plt.savefig("./intensity_hist.png", dpi=300)
# plt.show()

# --------------Volumes -------------#
# Prepare data for ridgeline plot - volumes
# data = []
# for vol, name in zip(volumes, vol_names):
#     flattened = vol.ravel()
#     df = pd.DataFrame({'Intensity': flattened, 'Volume': name})
#     data.append(df)
#
# data = pd.concat(data)

# Plot ridgeline plot
# joyplot(
#     data=data,
#     by='Volume',
#     column='Intensity',
#     kind='kde',
#     fill=True,
#     overlap=1.5,
#     linecolor='white',
#     xlabelsize=30,
#     ylabelsize=30,
#     xrot=270,
#     yrot=0,
#     # xlabels=np.arange(0, 256, 50),
#     x_range=np.arange(-1, 256),
#     linewidth=1,
#     figsize=(15, 10),
#     color='gray',
#     alpha=0.7,
# )
# plt.title("Intensity Distribution Ridgeline Plot")
# plt.xlabel('Intensity', fontsize=30)
# plt.ylabel('Volume', fontdict={'size': 25})

# Plot vertical ridgeline plot using seaborn
# plt.figure(figsize=(15, 10))
# g = sns.FacetGrid(data, row="Volume", hue="Volume", aspect=4, height=1.5, palette="gray")
# g.map(sns.kdeplot, "Intensity", fill=True, alpha=0.7)
# g.set_titles(row_template="{row_name}")
# g.set(yticks=[], ylabel="", xlabel="Intensity")
# g.despine(left=True, bottom=False)
# plt.subplots_adjust(hspace=0.5)


plt.tight_layout()
# plt.savefig("./ridgeplot_hist_real_fake.png", dpi=300)
plt.savefig("./ridgeplot_hist_real_fake.svg", dpi=300)
plt.show()
