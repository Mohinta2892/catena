import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy.stats import ttest_ind, ks_2samp

# ===================================================================
# --- Style and Font Configuration ---
# ===================================================================
# Set a publication-ready style with larger, sans-serif fonts
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['font.size'] = 25
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 40
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['legend.fontsize'] = 25
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), 'axes.linewidth': 2.5})


# --- User-provided image paths ---
image_paths = [
    '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/pytorch-CycleGAN-and-pix2pix/results/hemi2octo/test_latest/real_A/img_128_real_A.png', # Real Hemi
    '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/pytorch-CycleGAN-and-pix2pix/results/hemi2octo/test_latest/fake_A/img_128_fake_A.png', # Fake Hemi (Octo -> Hemi)
    '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/pytorch-CycleGAN-and-pix2pix/results/hemi2octo/test_latest/real_B/img_128_real_B.png', # Real Octo
    '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/pytorch-CycleGAN-and-pix2pix/results/hemi2octo/test_latest/fake_B/img_128_fake_B.png'  # Fake Octo (Hemi -> Octo)
]
vol_names = ["Hemi-Brain (Real)", "Hemi-Brain (Fake)", "Octo (Real)", "Octo (Fake)"]

# --- Data Loading and Preparation ---
try:
    data = []
    for path, name in zip(image_paths, vol_names):
        image = Image.open(path).convert('L')
        flattened = np.array(image).ravel()
        df = pd.DataFrame({'Intensity': flattened, 'Image': name})
        data.append(df)
    data_df = pd.concat(data)

    # --- Data Separation for Tests ---
    real_hemi = data_df[data_df['Image'] == 'Hemi-Brain (Real)']['Intensity']
    fake_hemi = data_df[data_df['Image'] == 'Hemi-Brain (Fake)']['Intensity']
    real_octo = data_df[data_df['Image'] == 'Octo (Real)']['Intensity']
    fake_octo = data_df[data_df['Image'] == 'Octo (Fake)']['Intensity']

    # ===================================================================
    # --- ANALYSIS 1: Real vs. Fake (from previous request) ---
    # ===================================================================

    # Perform Statistical Tests
    ttest_hemi_vs_fake = ttest_ind(real_hemi, fake_hemi, equal_var=False)
    ks_hemi_vs_fake = ks_2samp(real_hemi, fake_hemi)
    ttest_octo_vs_fake = ttest_ind(real_octo, fake_octo, equal_var=False)
    ks_octo_vs_fake = ks_2samp(real_octo, fake_octo)

    print("--- ANALYSIS 1: Real vs. Fake Image Comparisons ---")
    print("\n1. Hemi-Brain (Real vs. Fake):")
    print(f"  T-test: statistic={ttest_hemi_vs_fake.statistic:.4f}, p-value={ttest_hemi_vs_fake.pvalue:.4e}")
    print(f"  KS-test: statistic (D)={ks_hemi_vs_fake.statistic:.4f}, p-value={ks_hemi_vs_fake.pvalue:.4e}")
    print("\n2. Octo (Real vs. Fake):")
    print(f"  T-test: statistic={ttest_octo_vs_fake.statistic:.4f}, p-value={ttest_octo_vs_fake.pvalue:.4e}")
    print(f"  KS-test: statistic (D)={ks_octo_vs_fake.statistic:.4f}, p-value={ks_octo_vs_fake.pvalue:.4e}")
    print("-" * 50)

    # Plotting for Analysis 1
    fig1, axes1 = plt.subplots(1, 2, figsize=(20, 8))
    fig1.suptitle('Analysis 1: Comparison of Intensity Distributions (Real vs. Fake)', fontsize=24)
    sns.kdeplot(data=real_hemi, ax=axes1[0], color='#1f77b4', fill=True, label='Real Hemi')
    sns.kdeplot(data=fake_hemi, ax=axes1[0], color='#ff7f0e', fill=True, label='Fake Hemi')
    axes1[0].set_title('Hemi-Brain: Real vs. Fake')
    axes1[0].legend(loc="upper left")
    stats_text_hemi = f"T-test: p={ttest_hemi_vs_fake.pvalue:.2e}\nKS Test (D): {ks_hemi_vs_fake.statistic:.3f}, p={ks_hemi_vs_fake.pvalue:.2e}"
    axes1[0].text(0.95, 0.95, stats_text_hemi, transform=axes1[0].transAxes, fontsize=14, va='top', ha='right', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    sns.kdeplot(data=real_octo, ax=axes1[1], color='#2ca02c', fill=True, label='Real Octo')
    sns.kdeplot(data=fake_octo, ax=axes1[1], color='#d62728', fill=True, label='Fake Octo')
    axes1[1].set_title('Octo: Real vs. Fake')
    axes1[1].legend(loc="upper right")
    stats_text_octo = f"T-test: p={ttest_octo_vs_fake.pvalue:.2e}\nKS Test (D): {ks_octo_vs_fake.statistic:.3f}, p={ks_octo_vs_fake.pvalue:.2e}"
    axes1[1].text(0.95, 0.95, stats_text_octo, transform=axes1[1].transAxes, fontsize=14, va='top', ha='right', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    for ax in axes1:
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Density')
    sns.despine()
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    # fig1.savefig("./intensity_comparison_plot_real_vs_fake.png", dpi=300, bbox_inches='tight')
    plt.show()


    # ===================================================================
    # --- ANALYSIS 2: Real vs. Real and Fake vs. Fake (New request) ---
    # ===================================================================

    # Perform Statistical Tests
    ttest_real_vs_real = ttest_ind(real_hemi, real_octo, equal_var=False)
    ks_real_vs_real = ks_2samp(real_hemi, real_octo)
    ttest_fake_vs_fake = ttest_ind(fake_hemi, fake_octo, equal_var=False)
    ks_fake_vs_fake = ks_2samp(fake_hemi, fake_octo)

    print("\n--- ANALYSIS 2: Domain vs. Domain Comparisons ---")
    print("\n1. Real Hemi vs. Real Octo:")
    print(f"  T-test: statistic={ttest_real_vs_real.statistic:.4f}, p-value={ttest_real_vs_real.pvalue:.4e}")
    print(f"  KS-test: statistic (D)={ks_real_vs_real.statistic:.4f}, p-value={ks_real_vs_real.pvalue:.4e}")
    print("\n2. Fake Hemi vs. Fake Octo:")
    print(f"  T-test: statistic={ttest_fake_vs_fake.statistic:.4f}, p-value={ttest_fake_vs_fake.pvalue:.4e}")
    print(f"  KS-test: statistic (D)={ks_fake_vs_fake.statistic:.4f}, p-value={ks_fake_vs_fake.pvalue:.4e}")
    print("-" * 50)

    # Plotting for Analysis 2
    fig2, axes2 = plt.subplots(1, 2, figsize=(20, 8))
    fig2.suptitle('Analysis 2: Comparison of Intensity Distributions (Domain vs. Domain)', fontsize=24)

    # Plot 1: Real Hemi vs Real Octo
    sns.kdeplot(data=real_hemi, ax=axes2[0], color='#1f77b4', fill=True, label='Real Hemi')
    sns.kdeplot(data=real_octo, ax=axes2[0], color='#2ca02c', fill=True, label='Real Octo')
    axes2[0].set_title('Real Images: Hemi vs. Octo')
    axes2[0].legend()
    stats_text_real = f"T-test: p={ttest_real_vs_real.pvalue:.2e}\nKS Test (D): {ks_real_vs_real.statistic:.3f}, p={ks_real_vs_real.pvalue:.2e}"
    axes2[0].text(0.95, 0.95, stats_text_real, transform=axes2[0].transAxes, fontsize=14, va='top', ha='right', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    # Plot 2: Fake Hemi vs Fake Octo
    sns.kdeplot(data=fake_hemi, ax=axes2[1], color='#ff7f0e', fill=True, label='Fake Hemi')
    sns.kdeplot(data=fake_octo, ax=axes2[1], color='#d62728', fill=True, label='Fake Octo')
    axes2[1].set_title('Fake Images: Hemi vs. Octo')
    axes2[1].legend()
    stats_text_fake = f"T-test: p={ttest_fake_vs_fake.pvalue:.2e}\nKS Test (D): {ks_fake_vs_fake.statistic:.3f}, p={ks_fake_vs_fake.pvalue:.2e}"
    axes2[1].text(0.95, 0.95, stats_text_fake, transform=axes2[1].transAxes, fontsize=14, va='top', ha='right', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    for ax in axes2:
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Density')
    sns.despine()
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    # fig2.savefig("./intensity_comparison_plot_domain_vs_domain.png", dpi=300, bbox_inches='tight')
    plt.show()


except FileNotFoundError:
    print("Error: One or more image files were not found.")
    print("Please ensure the file paths in the 'image_paths' list are correct.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")