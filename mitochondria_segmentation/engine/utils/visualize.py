# 1. Import necessary libraries
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import os
import zarr  # Make sure zarr is installed: pip install zarr

# Assuming dataset.py is in the same directory, import EMDataset
from dataset import EMDataset

# Assuming you've installed scipy for 3D processing (recommended)
try:
    import scipy.ndimage

    print("Scipy is installed, 3D interpolation will be used for preprocessing.")
except ImportError:
    print("Scipy is NOT installed. Fallback to 2D slice-by-slice resizing for preprocessing.")


# --- Define the visualization function (copy the visualize_batch code here or in a .py file) ---

def visualize_batch(batch_X, batch_Y, num_samples_to_display=2, slice_to_display='middle',
                    find_prominent_slice=False, plot_all_labeled_slices=False):
    """
    Visualizes a batch of 3D EM data and their corresponding labels.
    Can find the most prominent slice or plot all slices containing labels.

    Args:
        batch_X (torch.Tensor): A batch of image data tensors (B, C, Z, Y, X).
        batch_Y (torch.Tensor): A batch of label data tensors (B, C, Z, Y, X).
        num_samples_to_display (int): How many samples from the batch to visualize.
        slice_to_display (str or int): 'middle' to display the middle Z-slice,
                                        or an integer Z-index to display a specific slice.
                                        Ignored if find_prominent_slice or plot_all_labeled_slices is True.
        find_prominent_slice (bool): If True, find and display the Z-slice with the most
                                     positive label pixels within each patch.
        plot_all_labeled_slices (bool): If True, plot all Z-slices within each patch
                                        that contain at least one positive label pixel.
                                        This overrides find_prominent_slice.
    """
    batch_size = batch_X.shape[0]
    display_count = min(num_samples_to_display, batch_size)

    print(f"Visualizing {display_count} samples from the batch. Batch shape: {batch_X.shape}")

    for i in range(display_count):
        img_3d = batch_X[i, 0].cpu().numpy()  # Remove batch and channel dim, move to CPU, to numpy
        label_3d = batch_Y[i, 0].cpu().numpy()  # Remove batch and channel dim, move to CPU, to numpy

        # Denormalize image data from [-1, 1] to [0, 255] for display
        img_3d_denormalized = ((img_3d + 1) * 127.5).astype(np.uint8)

        # Determine which slices to display based on parameters
        z_dim = img_3d_denormalized.shape[0]
        slices_to_plot = []

        if plot_all_labeled_slices:
            # Find all slices that contain any label pixels
            for z_idx in range(z_dim):
                if np.sum(label_3d[z_idx, :, :]) > 0:
                    slices_to_plot.append(z_idx)
            if not slices_to_plot:  # If no labeled slices, just show the middle one as fallback
                slices_to_plot.append(z_dim // 2)
                print(f"Sample {i + 1}: No labeled slices found in this patch. Displaying middle slice.")
            else:
                print(f"Sample {i + 1}: Displaying {len(slices_to_plot)} labeled slices.")
        elif find_prominent_slice:
            # Find the slice with the maximum number of positive pixels
            label_sums_per_slice = np.sum(label_3d, axis=(1, 2))
            if np.max(label_sums_per_slice) > 0:
                prominent_z_idx = np.argmax(label_sums_per_slice)
                slices_to_plot.append(prominent_z_idx)
                print(f"Sample {i + 1}: Displaying most prominent slice (Z-slice: {prominent_z_idx}).")
            else:  # If no labels at all in the patch (shouldn't happen with weighted sampling if labels exist)
                slices_to_plot.append(z_dim // 2)
                print(f"Sample {i + 1}: No labels found in this patch. Displaying middle slice.")
        else:
            # Default behavior: use slice_to_display
            if slice_to_display == 'middle':
                z_slice_idx = z_dim // 2
            else:
                z_slice_idx = int(slice_to_display)
            z_slice_idx = max(0, min(z_slice_idx, z_dim - 1))  # Ensure index is valid
            slices_to_plot.append(z_slice_idx)
            print(f"Sample {i + 1}: Displaying Z-slice: {z_slice_idx}.")

        # Plot the selected slices
        for plot_idx, z_slice_idx in enumerate(slices_to_plot):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(
                f"Sample {i + 1} (Z-slice: {z_slice_idx}){' - Labeled Slice ' + str(plot_idx + 1) if plot_all_labeled_slices and len(slices_to_plot) > 1 else ''}")

            # Display Image
            axes[0].imshow(img_3d_denormalized[z_slice_idx, :, :], cmap='gray')
            axes[0].set_title('Image')
            axes[0].axis('off')

            # Display Label
            # Use 'Reds' or a similar highly contrasting colormap for better visibility of sparse labels
            # Ensure vmin/vmax for binary labels
            axes[1].imshow(label_3d[z_slice_idx, :, :], cmap='Reds', vmin=0, vmax=1)
            axes[1].set_title('Label')
            axes[1].axis('off')

            plt.show()

    print("Batch visualization complete.")
