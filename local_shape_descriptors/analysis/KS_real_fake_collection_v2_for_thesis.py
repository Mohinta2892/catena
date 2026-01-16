import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy.stats import ks_2samp
import glob
import os

# ===================================================================
# --- 1. EXTRA LARGE STYLE CONFIGURATION ---
# ===================================================================
# Setting the theme with transparent axes
sns.set_theme(style="white", context='talk', rc={"axes.facecolor": (0, 0, 0, 0), 'axes.linewidth': 3.5})

# Overriding fonts and sizes for maximum readability
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

# SIGNIFICANTLY INCREASED FONT SIZES
plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelsize'] = 38      # X/Y Label size
plt.rcParams['axes.titlesize'] = 44      # Subplot Title size
plt.rcParams['xtick.labelsize'] = 34     # Tick numbers size
plt.rcParams['ytick.labelsize'] = 34
plt.rcParams['legend.fontsize'] = 32     # Legend text size
plt.rcParams['figure.titlesize'] = 52    # Main Figure Title size
plt.rcParams['lines.linewidth'] = 4      # Thicker plotting lines

# ===================================================================
# --- 2. PATHS & COLORS ---
# ===================================================================
base_path = "/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/pytorch-CycleGAN-and-pix2pix/results/hemi2octo/test_latest/"

paths = {
    "real_hemi": base_path + "real_A/",      # Source
    "real_octo": base_path + "real_B/",      # Target
    "stylized_hemi": base_path + "fake_B/"   # Generated (Hemi -> Octo)
}

colors = {
    "real_hemi": "#005f73",      # Blue
    "real_octo": "#9b2226",      # Red
    "stylized_hemi": "#ee9b00"   # Orange
}

# ===================================================================
# --- 3. DATA LOADING ---
# ===================================================================
def load_flat(directory):
    """Loads all pixels flattened (For Plot 1: Population Gap)"""
    files = sorted(glob.glob(os.path.join(directory, '*.png')))
    pixels = []
    print(f"Loading global data from {os.path.basename(directory)}...")
    for f in files:
        with Image.open(f).convert('L') as img:
            pixels.extend(list(img.getdata()))
    return np.array(pixels)

def load_dict(directory):
    """Loads images by filename (For Plot 2: Paired Magnitude)"""
    files = sorted(glob.glob(os.path.join(directory, '*.png')))
    img_dict = {}
    print(f"Loading paired data from {os.path.basename(directory)}...")
    for f in files:
        filename = os.path.basename(f)
        with Image.open(f).convert('L') as img:
            img_dict[filename] = np.array(list(img.getdata()))
    return img_dict

# ===================================================================
# --- 4. EXECUTION ---
# ===================================================================
try:
    print("Loading data...")
    # Load for Plot 1
    real_hemi_flat = load_flat(paths["real_hemi"])
    real_octo_flat = load_flat(paths["real_octo"])
    stylized_hemi_flat = load_flat(paths["stylized_hemi"])
    
    # Load for Plot 2
    real_hemi_dict = load_dict(paths["real_hemi"])
    stylized_hemi_dict = load_dict(paths["stylized_hemi"])

    # ---------------------------------------------------------------
    # PLOT 1: THE DOMAIN GAP PROOF (Population Level)
    # ---------------------------------------------------------------
    ks_gap = ks_2samp(real_hemi_flat, real_octo_flat)
    ks_sol = ks_2samp(stylized_hemi_flat, real_octo_flat)

    # Increased figure size to accommodate larger fonts
    fig1, axes = plt.subplots(1, 2, figsize=(28, 12))
    
    # Subplot A: The Problem
    sns.kdeplot(real_hemi_flat, ax=axes[0], color=colors['real_hemi'], fill=True, label='Real Hemi')
    sns.kdeplot(real_octo_flat, ax=axes[0], color=colors['real_octo'], fill=True, label='Real Octo')
    axes[0].set_title(f"A. The Domain Gap\n(Real vs. Real)\nD = {ks_gap.statistic:.3f}")
    axes[0].set_xlabel("Pixel Intensity")
    axes[0].set_ylabel("Density")
    axes[0].legend(facecolor='white', framealpha=1, frameon=True, loc='upper right')

    # Subplot B: The Solution
    sns.kdeplot(real_octo_flat, ax=axes[1], color=colors['real_octo'], fill=True, label='Real Octo')
    sns.kdeplot(stylized_hemi_flat, ax=axes[1], color=colors['stylized_hemi'], fill=True, label='Stylized Hemi')
    axes[1].set_title(f"B. Domain Adaptation\n(Stylized vs. Real)\nD = {ks_sol.statistic:.3f}")
    axes[1].set_xlabel("Pixel Intensity")
    axes[1].set_ylabel("Density")
    axes[1].legend(facecolor='white', framealpha=1, frameon=True, loc='upper right')

    sns.despine()
    fig1.tight_layout()
    fig1.savefig("Thesis_Plot1_DomainGap_Large.svg")
    print("Saved 'Thesis_Plot1_DomainGap_Large.svg'")


    # ---------------------------------------------------------------
    # PLOT 2: THE PAIRED COMPARISON (Boxplot)
    # ---------------------------------------------------------------
    d_stats_paired = []
    
    for filename, input_pixels in real_hemi_dict.items():
        target_name = filename.replace('real_A', 'fake_B')
        if target_name not in stylized_hemi_dict:
             target_name = filename 
        if target_name in stylized_hemi_dict:
            output_pixels = stylized_hemi_dict[target_name]
            stat, _ = ks_2samp(input_pixels, output_pixels)
            d_stats_paired.append(stat)

    if d_stats_paired:
        df_paired = pd.DataFrame({
            'D-Statistic': d_stats_paired,
            'Comparison': 'Real Hemi vs.\nStylized Hemi'
        })

        # Increased figure size for the boxplot
        fig2, ax2 = plt.subplots(figsize=(16, 12))
        
        sns.boxplot(x='Comparison', y='D-Statistic', data=df_paired, ax=ax2, 
                    color=colors['stylized_hemi'], width=0.4, linewidth=3.5)
        
        # Increased dot size for visibility
        sns.stripplot(x='Comparison', y='D-Statistic', data=df_paired, ax=ax2, 
                      color='black', alpha=0.5, size=10, jitter=True)
        
        ax2.set_title(f"C. Magnitude of Style Transfer (Paired)\nMean D = {np.mean(d_stats_paired):.3f}")
        ax2.set_ylabel("KS Distance (Input vs Output)")
        ax2.set_xlabel("")

        sns.despine()
        fig2.tight_layout()
        fig2.savefig("Thesis_Plot2_PairedBoxplot_Large.svg")
        print("Saved 'Thesis_Plot2_PairedBoxplot_Large.svg'")
    else:
        print("WARNING: No pairs found for Plot 2.")
    
    plt.show()

except Exception as e:
    print(f"Error: {e}")
