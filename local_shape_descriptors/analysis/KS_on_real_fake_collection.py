import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy.stats import ttest_ind, ks_2samp
import glob
import os

# ===================================================================
# --- Style and Font Configuration ---
# ===================================================================
# Set the seaborn theme first to establish a base style.
# Using 'talk' context provides a good starting point for larger plots.
sns.set_theme(style="white", context='talk', rc={"axes.facecolor": (0, 0, 0, 0), 'axes.linewidth': 2.5})

# Now, override specific settings with rcParams for fine-grained control.
# These settings will now correctly apply on top of the seaborn theme.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['font.size'] = 22
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 22
plt.rcParams['figure.titlesize'] = 36


# ===================================================================
# --- Consistent Color Palette ---
# ===================================================================
# Define a consistent color scheme for all plots.
# Real versions are darker, Fake versions are lighter shades of the same color.
color_palette = {
    "real_hemi": "#005f73",  # Dark Teal
    "fake_hemi": "#0a9396",  # Medium Teal
    "real_octo": "#9b2226",  # Dark Red
    "fake_octo": "#ee9b00"   # Medium Orange/Gold
}


# ===================================================================
# --- IMPORTANT: SET YOUR DIRECTORY PATHS HERE ---
# ===================================================================
# Please update these paths to point to the folders containing your images.
directory_paths = {
    "real_hemi": "/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/pytorch-CycleGAN-and-pix2pix/results/hemi2octo/test_latest/real_A/",
    "fake_hemi": "/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/pytorch-CycleGAN-and-pix2pix/results/hemi2octo/test_latest/fake_B/",
    "real_octo": "/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/pytorch-CycleGAN-and-pix2pix/results/hemi2octo/test_latest/real_B/",
    "fake_octo": "/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/pytorch-CycleGAN-and-pix2pix/results/hemi2octo/test_latest/fake_A/"
}

def load_images_from_directory(directory_path, image_format='png'):
    """
    Loads all images from a directory, flattens their pixel intensities,
    and returns them as a single NumPy array (for domain-level analysis).
    """
    search_pattern = os.path.join(directory_path, f'*.{image_format}')
    # Sort the paths to ensure consistent processing order, though not critical for this function
    image_paths = sorted(glob.glob(search_pattern))

    if not image_paths:
        print(f"Warning: No images with format '.{image_format}' found in '{directory_path}'")
        return None

    print(f"Found {len(image_paths)} images in '{os.path.basename(os.path.normpath(directory_path))}' directory for domain analysis.")

    all_intensities = []
    for path in image_paths:
        try:
            with Image.open(path).convert('L') as img:
                all_intensities.extend(list(img.getdata()))
        except Exception as e:
            print(f"Could not process image {path}: {e}")

    return np.array(all_intensities)

def load_images_as_dict(directory_path, image_format='png'):
    """
    Loads all images from a directory into a dictionary where keys are
    filenames and values are flattened pixel intensity arrays (for paired analysis).
    """
    search_pattern = os.path.join(directory_path, f'*.{image_format}')
    # Sort the paths to ensure a consistent order
    image_paths = sorted(glob.glob(search_pattern))

    if not image_paths:
        print(f"Warning: No images with format '.{image_format}' found in '{directory_path}'")
        return None

    print(f"Found {len(image_paths)} images in '{os.path.basename(os.path.normpath(directory_path))}' directory for paired analysis.")

    image_dict = {}
    for path in image_paths:
        try:
            filename = os.path.basename(path)
            with Image.open(path).convert('L') as img:
                image_dict[filename] = np.array(list(img.getdata()))
        except Exception as e:
            print(f"Could not process image {path}: {e}")

    return image_dict


# --- Data Loading and Preparation ---
try:
    # --- Load data for Domain-Level Analysis (Analyses 1, 2, 3) ---
    real_hemi_domain = load_images_from_directory(directory_paths["real_hemi"])
    fake_hemi_domain = load_images_from_directory(directory_paths["fake_hemi"])
    real_octo_domain = load_images_from_directory(directory_paths["real_octo"])
    fake_octo_domain = load_images_from_directory(directory_paths["fake_octo"])

    if any(d is None for d in [real_hemi_domain, fake_hemi_domain, real_octo_domain, fake_octo_domain]):
        raise ValueError("One or more directories did not contain images for domain analysis. Aborting.")

    # --- Load data for Paired Analysis (Analysis 4) ---
    real_hemi_paired = load_images_as_dict(directory_paths["real_hemi"])
    fake_hemi_paired = load_images_as_dict(directory_paths["fake_hemi"])
    real_octo_paired = load_images_as_dict(directory_paths["real_octo"])
    fake_octo_paired = load_images_as_dict(directory_paths["fake_octo"])

    if any(d is None for d in [real_hemi_paired, fake_hemi_paired, real_octo_paired, fake_octo_paired]):
        raise ValueError("One or more directories did not contain images for paired analysis. Aborting.")


    # ===================================================================
    # --- ANALYSIS 1: Real vs. Fake (Within-Domain) ---
    # ===================================================================
    print("\n--- Running Analysis 1: Real vs. Fake ---")
    ks_hemi_vs_fake = ks_2samp(real_hemi_domain, fake_hemi_domain)
    ks_octo_vs_fake = ks_2samp(real_octo_domain, fake_octo_domain)

    fig1, axes1 = plt.subplots(1, 2, figsize=(24, 10))
    fig1.suptitle('Analysis 1: Within-Domain Comparison (Real vs. Fake)')

    sns.kdeplot(data=real_hemi_domain, ax=axes1[0], color=color_palette['real_hemi'], fill=True, label='Real Hemi')
    sns.kdeplot(data=fake_hemi_domain, ax=axes1[0], color=color_palette['fake_hemi'], fill=True, label='Fake Hemi')
    axes1[0].set_title('Hemi-Brain: Real vs. Fake')
    axes1[0].legend(frameon=False)
    stats_text_hemi = f"KS Test (D): {ks_hemi_vs_fake.statistic:.3f}\np-value: {ks_hemi_vs_fake.pvalue:.2e}"
    axes1[0].text(0.95, 0.80, stats_text_hemi, transform=axes1[0].transAxes, va='top', ha='right', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.6))

    sns.kdeplot(data=real_octo_domain, ax=axes1[1], color=color_palette['real_octo'], fill=True, label='Real Octo')
    sns.kdeplot(data=fake_octo_domain, ax=axes1[1], color=color_palette['fake_octo'], fill=True, label='Fake Octo')
    axes1[1].set_title('Octo: Real vs. Fake')
    axes1[1].legend(frameon=False)
    stats_text_octo = f"KS Test (D): {ks_octo_vs_fake.statistic:.3f}\np-value: {ks_octo_vs_fake.pvalue:.2e}"
    axes1[1].text(0.95, 0.80, stats_text_octo, transform=axes1[1].transAxes, va='top', ha='right', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.6))

    for ax in axes1:
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Density')
    sns.despine()
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig1.savefig("./intensity_comparison_plot_real_vs_fake_GLOB.svg", dpi=300, bbox_inches='tight')
    plt.show()

    # ===================================================================
    # --- ANALYSIS 2: Domain vs. Domain ---
    # ===================================================================
    print("\n--- Running Analysis 2: Domain vs. Domain ---")
    ks_real_vs_real = ks_2samp(real_hemi_domain, real_octo_domain)
    ks_fake_vs_fake = ks_2samp(fake_hemi_domain, fake_octo_domain)

    fig2, axes2 = plt.subplots(1, 2, figsize=(24, 10))
    fig2.suptitle('Analysis 2: Comparison Between Domains')

    sns.kdeplot(data=real_hemi_domain, ax=axes2[0], color=color_palette['real_hemi'], fill=True, label='Real Hemi')
    sns.kdeplot(data=real_octo_domain, ax=axes2[0], color=color_palette['real_octo'], fill=True, label='Real Octo')
    axes2[0].set_title('Real Images: Hemi vs. Octo')
    axes2[0].legend(frameon=False)
    stats_text_real = f"KS Test (D): {ks_real_vs_real.statistic:.3f}\np-value: {ks_real_vs_real.pvalue:.2e}"
    axes2[0].text(0.95, 0.80, stats_text_real, transform=axes2[0].transAxes, va='top', ha='right', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.6))

    sns.kdeplot(data=fake_hemi_domain, ax=axes2[1], color=color_palette['fake_hemi'], fill=True, label='Fake Hemi')
    sns.kdeplot(data=fake_octo_domain, ax=axes2[1], color=color_palette['fake_octo'], fill=True, label='Fake Octo')
    axes2[1].set_title('Fake Images: Hemi vs. Octo')
    axes2[1].legend(frameon=False)
    stats_text_fake = f"KS Test (D): {ks_fake_vs_fake.statistic:.3f}\np-value: {ks_fake_vs_fake.pvalue:.2e}"
    axes2[1].text(0.95, 0.80, stats_text_fake, transform=axes2[1].transAxes, va='top', ha='right', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.6))

    for ax in axes2:
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Density')
    sns.despine()
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig("./intensity_comparison_plot_domain_vs_domain_GLOB.svg", dpi=300, bbox_inches='tight')
    plt.show()

    # ===================================================================
    # --- ANALYSIS 3: Cross-Domain Translation Evaluation ---
    # ===================================================================
    print("\n--- Running Analysis 3: Cross-Domain Translation Evaluation ---")
    ks_hemi_vs_fake_octo = ks_2samp(real_hemi_domain, fake_octo_domain)
    ks_octo_vs_fake_hemi = ks_2samp(real_hemi_domain, fake_hemi_domain)

    fig3, axes3 = plt.subplots(1, 2, figsize=(24, 10))
    fig3.suptitle('Analysis 3: Evaluating Cross-Domain Translation')

    # Plot 3a: Real Hemi vs. Fake Octo
    sns.kdeplot(data=real_hemi_domain, ax=axes3[0], color=color_palette['real_hemi'], fill=True, label='Real Hemi (Target)')
    sns.kdeplot(data=fake_octo_domain, ax=axes3[0], color=color_palette['fake_octo'], fill=True, label='Fake Octo (Translated)')
    axes3[0].set_title('Hemi Style Transfer to Octo Domain')
    axes3[0].legend(frameon=False)
    stats_text_hemi_target = f"KS Test (D): {ks_hemi_vs_fake_octo.statistic:.3f}\np-value: {ks_hemi_vs_fake_octo.pvalue:.2e}"
    axes3[0].text(0.95, 0.80, stats_text_hemi_target, transform=axes3[0].transAxes, va='top', ha='right', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.6))

    # Plot 3b: Real Octo vs. Fake Hemi
    sns.kdeplot(data=real_octo_domain, ax=axes3[1], color=color_palette['real_octo'], fill=True, label='Real Octo (Target)')
    sns.kdeplot(data=fake_hemi_domain, ax=axes3[1], color=color_palette['fake_hemi'], fill=True, label='Fake Hemi (Translated)')
    axes3[1].set_title('Octo Style Transfer to Hemi Domain')
    axes3[1].legend(frameon=False)
    stats_text_octo_target = f"KS Test (D): {ks_octo_vs_fake_hemi.statistic:.3f}\np-value: {ks_octo_vs_fake_hemi.pvalue:.2e}"
    axes3[1].text(0.95, 0.80, stats_text_octo_target, transform=axes3[1].transAxes, va='top', ha='right', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.6))

    for ax in axes3:
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Density')
    sns.despine()
    fig3.tight_layout(rect=[0, 0, 1, 0.95])
    fig3.savefig("./intensity_comparison_plot_translation_eval_GLOB.svg", dpi=300, bbox_inches='tight')
    plt.show()

    # ===================================================================
    # --- ANALYSIS 4: Paired Image-by-Image Comparison (NEW) ---
    # ===================================================================
    print("\n--- Running Analysis 4: Paired Image-by-Image Comparison ---")

    ks_distances_A_to_B = []
    # Compare each real_A image to its corresponding fake_B image
    for filename, real_a_pixels in real_hemi_paired.items():
        # CycleGAN often saves fake images with a suffix, e.g., 'img_1_real_A.png' -> 'img_1_fake_B.png'
        # We need to construct the expected fake filename.
        # This part might need adjustment based on your exact filename convention.
        base_name = filename.replace('_real_A', '')
        fake_b_filename = base_name.replace('.png', '_fake_B.png')

        if fake_b_filename in fake_octo_paired:
            fake_b_pixels = fake_octo_paired[fake_b_filename]
            ks_stat, _ = ks_2samp(real_a_pixels, fake_b_pixels)
            ks_distances_A_to_B.append(ks_stat)
        else:
            print(f"Warning: Could not find matching fake_B file for {filename}")

    ks_distances_B_to_A = []
    # Compare each real_B image to its corresponding fake_A image
    for filename, real_b_pixels in real_octo_paired.items():
        base_name = filename.replace('_real_B', '')
        fake_a_filename = base_name.replace('.png', '_fake_A.png')

        if fake_a_filename in fake_hemi_paired:
            fake_a_pixels = fake_hemi_paired[fake_a_filename]
            ks_stat, _ = ks_2samp(real_b_pixels, fake_a_pixels)
            ks_distances_B_to_A.append(ks_stat)
        else:
            print(f"Warning: Could not find matching fake_A file for {filename}")

    if not ks_distances_A_to_B or not ks_distances_B_to_A:
        print("\nCould not perform paired analysis. Check filename matching logic.")
    else:
        # Create a DataFrame for plotting
        paired_df_list = []
        if ks_distances_A_to_B:
            paired_df_list.append(pd.DataFrame({
                'KS Distance (D)': ks_distances_A_to_B,
                'Comparison': 'Real Hemi vs. Fake Octo'
            }))
        if ks_distances_B_to_A:
            paired_df_list.append(pd.DataFrame({
                'KS Distance (D)': ks_distances_B_to_A,
                'Comparison': 'Real Octo vs. Fake Hemi'
            }))

        paired_df = pd.concat(paired_df_list, ignore_index=True)

        # Plotting the distribution of KS distances
        fig4, ax4 = plt.subplots(figsize=(14, 8))
        fig4.suptitle('Analysis 4: Distribution of Paired Image Distances')

        sns.boxplot(x='Comparison', y='KS Distance (D)', data=paired_df, ax=ax4)
        sns.stripplot(x='Comparison', y='KS Distance (D)', data=paired_df, ax=ax4, color=".25", size=6)

        ax4.set_title('Lower values indicate better translation for individual images')
        ax4.set_xlabel('Paired Comparison')
        ax4.set_ylabel('Kolmogorov-Smirnov D-statistic')

        sns.despine()

        fig4.tight_layout(rect=[0, 0, 1, 0.95])
        fig4.savefig("./intensity_comparison_plot_paired_eval_GLOB.svg", dpi=300, bbox_inches='tight')
        plt.show()


except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
