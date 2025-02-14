"""
This script creates and saves a gif given an EM (downsampled version) and masks. More than one mask is accepted.
Ensures that the gif is compressed. However, this functionality may not give correct results.
Please use online compression websites to compress otherwise.
Author: Samia Mohinta
Affiliation: Cambridge University, UK
"""

import numpy as np
from PIL import Image
import tifffile
import matplotlib.pyplot as plt
import imageio
import os
from pathlib import Path
from tqdm import tqdm
from matplotlib.patches import Patch


def create_volume_gif(raw_path, mask_paths, output_path='output.gif',
                      fps=5, mask_colors=None, mask_alphas=None, mask_names=None):
    """
    Create an animated GIF from a raw volume and multiple segmentation masks.

    Parameters:
    -----------
    raw_path : str
        Path to the raw TIF file
    mask_paths : list of str
        List of paths to NPY mask files
    output_path : str
        Path where the output GIF should be saved
    fps : int
        Frames per second for the GIF
    mask_colors : list of str or None
        List of colors for each mask. If None, will use default colors
    mask_alphas : list of float or None
        List of transparency values for each mask. If None, will use 0.3 for all
    mask_names : list of str or None
        List of names for each mask. If None, will use "Mask 1", "Mask 2", etc.
    """
    # Load the raw data
    raw_data = tifffile.imread(raw_path)

    # Load all masks
    masks = [np.load(mask_path) if mask_path.endswith('.npy') else tifffile.imread(mask_path) for mask_path in
             mask_paths]

    # Verify dimensions match
    for mask in masks:
        assert mask.shape == raw_data.shape, f"Mask shape {mask.shape} doesn't match raw data shape {raw_data.shape}"

    # Set default colors if not provided
    if mask_colors is None:
        default_colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
        mask_colors = default_colors[:len(masks)]

    # Set default alphas if not provided
    if mask_alphas is None:
        mask_alphas = [0.3] * len(masks)

    # Set default mask names if not provided
    if mask_names is None:
        mask_names = [f"Mask {i + 1}" for i in range(len(masks))]

    # Create temporary directory for frames
    frames_dir = Path('temp_frames')
    frames_dir.mkdir(exist_ok=True)

    # Create frames
    frames = []
    for z in tqdm(range(raw_data.shape[0])):
        # Create figure with square aspect ratio
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()

        # Plot raw data
        plt.imshow(raw_data[z], cmap='gray')

        # Overlay each mask
        for mask, color, alpha in zip(masks, mask_colors, mask_alphas):
            plt.imshow(mask[z], alpha=alpha * (mask[z] > 0), cmap=plt.cm.colors.ListedColormap([color]))

        # Create legend elements
        legend_elements = [Patch(facecolor=color, alpha=alpha, label=name)
                           for color, alpha, name in zip(mask_colors, mask_alphas, mask_names)]

        # Add legend inside the plot
        plt.legend(handles=legend_elements,
                   loc='upper right',  # Position inside the plot
                   bbox_to_anchor=(0.98, 0.98),  # Fine-tune position
                   fontsize=10,
                   framealpha=0.7)  # Semi-transparent background

        # Remove axes
        plt.axis('off')

        # Save frame
        frame_path = frames_dir / f'frame_{z:04d}.png'
        plt.savefig(frame_path, bbox_inches='tight', pad_inches=0, dpi=100)
        frames.append(imageio.v2.imread(frame_path))
        plt.close()

    # Create GIF
    imageio.mimsave(output_path, frames, fps=fps, optimize=True,  # Enable optimization
                    subrectangles=True  # Enable sub-rectangles optimization
                    )
    # Calculate and print file size
    gif_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
    print(f"GIF size: {gif_size:.2f} MB")

    # If GIF is still too large, create a more compressed version
    if gif_size > 10:  # If larger than 10MB
        print("Creating compressed version...")
        with Image.open(output_path) as img:
            # Convert to PIL images and reduce colors
            pil_frames = []
            for frame in frames:
                pil_frame = Image.fromarray(frame).convert('RGB').quantize(colors=64)
                pil_frames.append(pil_frame)

            # Save with maximum compression
            compressed_path = output_path.replace('.gif', '_compressed.gif')
            pil_frames[0].save(
                compressed_path,
                save_all=True,
                append_images=pil_frames[1:],
                optimize=True,
                # duration=1000 / fps,
                fps=fps * 3,  # make frames go up?
                loop=0,
                quality=1  # Maximum compression
            )
            compressed_size = os.path.getsize(compressed_path) / (1024 * 1024)
            print(f"Compressed GIF size: {compressed_size:.2f} MB")

    # Clean up temporary files
    for frame in frames_dir.glob('*.png'):
        frame.unlink()
    frames_dir.rmdir()


# Example usage
if __name__ == "__main__":
    # Example paths and parameters
    raw_path = "/media/samia/DATA/mounts/fibserver1/smohinta_data/catena_data/OCTO/data_3d/test/octo_cns_s5.tif"
    mask_paths = [
        "/media/samia/DATA/PhD/codebases/synapse_extra_projects/Camb/EM_mask_generation/plots_results/run6/octo_cns_s5_pred_mask_w_transforms.tif",
        "/media/samia/DATA/mounts/cephfs/auto_mask_generation/largest_region_mask.npy"]
    mask_colors = ['magenta', 'yellow']
    mask_alphas = [0.35, 0.3]
    mask_names = ["ML Prediction", "Post-Processed"]  # Custom names for the masks

    create_volume_gif(
        raw_path=raw_path,
        mask_paths=mask_paths,
        output_path='/media/samia/DATA/mounts/cephfs/auto_mask_generation/pred_post_octo.gif',
        fps=5000,
        mask_colors=mask_colors,
        mask_alphas=mask_alphas,
        mask_names=mask_names
    )
