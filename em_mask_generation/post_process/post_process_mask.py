"""
This post-processing script fills the holes in the base brain/non-brain prediction tiff.
It also enlarges and erodes the predicted mask to make it properly overlay on top the EM.
TODO: Properly align the mask over the EM without growing it too much.

Author: Peter Hague
Affiliation: MRC LMB UK
"""

import skimage.io as io
from tqdm import tqdm
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, dilation, disk, reconstruction
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def fill_3d_holes(binary_mask, structure=None):
    """
    Fill holes in a 3D binary mask using morphological reconstruction.
    
    Parameters:
    -----------
    binary_mask : ndarray
        3D binary array where 1s represent the object and 0s represent background/holes
    structure : ndarray, optional
        Structuring element used for the morphological operation.
        If None, uses a 3D cross-shaped structure (6-connectivity)
    
    Returns:
    --------
    ndarray
        Binary mask with holes filled
    """
    # Create default structure if none provided
    if structure is None:
        structure = np.array([[[0, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]],
                              [[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]],
                              [[0, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]]], dtype=bool)
    # Identify background regions
    background = ~binary_mask
    # Label background regions
    labels, num_labels = ndimage.label(background, structure=structure)
    # Find background regions that don't touch the border
    if num_labels > 0:
        # Get unique labels on the border
        border_labels = np.unique(np.concatenate([
            labels[0, :, :].ravel(),  # front face
            labels[-1, :, :].ravel(),  # back face
            labels[:, 0, :].ravel(),  # left face
            labels[:, -1, :].ravel(),  # right face
            labels[:, :, 0].ravel(),  # top face
            labels[:, :, -1].ravel()  # bottom face
        ]))
        # Create mask of holes (background regions that don't touch border)
        holes = np.zeros_like(binary_mask, dtype=bool)
        for i in tqdm(range(1, num_labels + 1), desc="Filling holes"):
            if i not in border_labels:
                holes[labels == i] = True
        # Fill the holes
        filled_mask = binary_mask | holes
    else:
        filled_mask = binary_mask
    return filled_mask


# Load mask
mask = io.imread(
    "/media/samia/DATA/PhD/codebases/synapse_extra_projects/Camb/EM_mask_generation/plots_results/run6/sam_vol_pred_mask_w_transforms.tif")
# size = (1085, 670, 640)

# Create a structuring element (disk kernel)
selem = disk(radius=3)  # You can adjust the radius based on your needs
processed_mask = np.zeros_like(mask)
# Process each slice of the 3D mask
for z in tqdm(range(mask.shape[2]), desc="Processing slices"):
    # Perform erosion
    eroded = binary_erosion(mask[z], selem)
    # Perform dilation on the eroded image
    dilated = dilation(eroded, selem)
    # Store the result
    processed_mask[z] = dilated

# Get mask regions
print("Getting mask regions...")
mask_labelled = label(processed_mask)
labels = regionprops(mask_labelled)

# Get region sizes
print("Getting region sizes...")
region_sizes = [label.area for label in labels]

# Get region with the largest size
print("Getting largest region...")
largest_region = max(labels, key=lambda x: x.area)
largest_region_id = largest_region.label

# Get largest region mask
print("Getting largest region mask...")
largest_region_mask = mask_labelled == largest_region_id

# Fill holes in mask
# largest_region_mask_reconstructed = reconstruction(largest_region_mask, largest_region_mask, method="erosion")
print("Filling holes in largest region mask...")
largest_region_mask_filled = fill_3d_holes(largest_region_mask)


# Create a solid sphere of 1s inscribed in a 50x50x50 array
def create_solid_sphere(radius, shape):
    assert len(shape) == 3, "Shape must be a 3-tuple"
    z, y, x = np.indices(shape)
    center = np.array(shape) // 2
    distance = np.sqrt((x - center[2]) ** 2 + (y - center[1]) ** 2 + (z - center[0]) ** 2)
    sphere = distance <= radius
    return sphere


sphere = create_solid_sphere(radius=15, shape=(30, 30, 30))

# Convolve the sphere with the largest_region_mask
print("Convolving sphere with largest region mask...")
convolved_mask = ndimage.convolve(largest_region_mask_filled.astype(float), sphere.astype(float), mode='constant',
                                  cval=0.0)

largest_region_mask_filled = convolved_mask > 0.5  # Threshold to get binary mask

# Cheat, and copy the -40 slice to all the slice after it
# largest_region_mask_filled[-40:] = largest_region_mask_filled[-40]

# Plot region sizes
n_slices = 10
slices_ = np.linspace(0, mask.shape[2], n_slices)[1:-1].astype(int)
n_slices = len(slices_)
fig, ax = plt.subplots(n_slices, 2, figsize=(10, 5 * n_slices))
for i, s in tqdm(enumerate(slices_), desc="Plotting slices"):
    ax[i, 0].imshow(mask[s], cmap="binary", origin="lower", aspect="auto")
    ax[i, 1].imshow(largest_region_mask_filled[s], cmap="binary", origin="lower", aspect="auto")

plt.tight_layout()
plt.savefig("eroded_regions.png")

np.save("sam_vol_largest_mask.npy", largest_region_mask_filled)

# Erode the mask
# eroded_mask = binary_erosion(largest_region_mask, disk(3))

# Dilate the eroded mask
# cleaned_mask = dilation(eroded_mask, disk(3))

# Update the largest_region_mask with the cleaned version
# largest_region_mask = cleaned_mask
