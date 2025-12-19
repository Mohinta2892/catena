import numpy as np
import pandas as pd
import zarr
import os
from skimage.measure import regionprops
from skimage.morphology import binary_dilation, binary_erosion, cube
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Patch

# --- CONFIGURATION ---
plt.rcParams.update({
    'font.size': 30,
    'axes.titlesize': 30,
    'axes.labelsize': 30,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 30,
    'figure.titlesize': 30,
    'font.family': 'sans-serif'
})
sns.set_style("ticks")

# 1. COLOR & STYLE DEFINITIONS (Enforcing Consistency)
# Group Colors (Blue vs Red)
GROUP_PALETTE = {
    "Hemibrain": "#3498db",  # Blue
    "Octo": "#e74c3c",  # Red
    "Other": "#95a5a6"  # Grey
}

# Group Markers (Circle vs X)
GROUP_MARKERS = {
    "Hemibrain": "o",
    "Octo": "X",
    "Other": "s"
}

# Category Colors (Green/Yellow/Orange/Purple - Distinct from Groups)
CATEGORY_PALETTE = {
    "Easy": "#2ecc71",  # Green
    "Crowded": "#f1c40f",  # Yellow
    "Fragile": "#e67e22",  # Orange
    "Complex": "#9b59b6"  # Purple
}

# 2. ANALYSIS SETTINGS
VOXEL_RES = (8, 8, 8)  # (z, y, x) in nanometers
RAW_KEY = 'volumes/raw'
LABEL_KEY = 'volumes/labels/neuron_ids'
MIN_LOG_VOLUME = -2.5


def get_group_name(sample_name):
    """Collapses complex filenames into simple Groups."""
    name_lower = sample_name.lower()
    if 'hemi' in name_lower:
        return 'Hemibrain'
    elif 'octo' in name_lower:
        return 'Octo'
    else:
        return 'Other'


def load_zarr_arrays(path, raw_key, label_key):
    try:
        f = zarr.open(path, mode='r')
        print(f"  Loading raw from {raw_key}...")
        raw = f[raw_key][:]
        print(f"  Loading labels from {label_key}...")
        labels = f[label_key][:]
        return raw, labels
    except Exception as e:
        print(f"  Error loading keys from {path}: {e}")
        return None, None


def calculate_mito_metrics(label_vol, raw_vol, voxel_res, sample_name="Unknown"):
    print(f"  Calculating Region Properties for {sample_name}...")
    label_vol = label_vol.astype(int)
    props = regionprops(label_vol, intensity_image=raw_vol)
    data = []
    selem = cube(3)
    total_mitos = len(props)

    for idx, prop in enumerate(props):
        vol_phys = prop.area * np.prod(voxel_res) / 1e9
        log_vol = np.log10(vol_phys + 1e-9)

        if log_vol < MIN_LOG_VOLUME:
            dci, efi = np.nan, np.nan
            elongation = np.nan
            mean_intensity = prop.mean_intensity
        else:
            eig_vals = prop.inertia_tensor_eigvals
            elongation = 0 if eig_vals[-1] == 0 else np.sqrt(eig_vals[0] / eig_vals[-1])
            mean_intensity = prop.mean_intensity

            min_z, min_y, min_x, max_z, max_y, max_x = prop.bbox
            local_mask = prop.image
            dilated_mask = binary_dilation(local_mask, selem)
            local_labels_crop = label_vol[min_z:max_z, min_y:max_y, min_x:max_x]
            overlapping_labels = local_labels_crop[dilated_mask]
            unique_neighbors = np.unique(overlapping_labels)
            dci = np.sum((unique_neighbors != 0) & (unique_neighbors != prop.label))

            eroded_mask = binary_erosion(local_mask, selem)
            if not np.any(eroded_mask):
                efi = 0
            else:
                from skimage.measure import label as separate_components
                _, efi = separate_components(eroded_mask, return_num=True, connectivity=3)

        data.append({
            'Sample': sample_name,
            'Group': get_group_name(sample_name),
            'Label_ID': prop.label,
            'Volume_um3': vol_phys,
            'Log_Volume': log_vol,
            'Elongation': elongation,
            'Mean_Intensity': mean_intensity,
            'DCI': dci,
            'EFI': efi
        })
        if idx % 500 == 0 and idx > 0:
            print(f"  Processed {idx}/{total_mitos}...")
    return pd.DataFrame(data)


def generate_cumulative_plots(df, save_prefix="cumulative_analysis"):
    df_clean = df.dropna(subset=['DCI', 'EFI']).copy()
    print(f"Plotting {len(df_clean)} valid mitochondria.")

    df_clean['DCI_Jitter'] = df_clean['DCI'] + np.random.normal(0, 0.1, size=len(df_clean))
    df_clean['EFI_Jitter'] = df_clean['EFI'] + np.random.normal(0, 0.1, size=len(df_clean))

    # --- 1. SIZE DISTRIBUTION ---
    plt.figure(figsize=(12, 10))
    sns.histplot(data=df_clean, x='Log_Volume', kde=True, color="#34495e", alpha=0.6)
    plt.xlabel('Log10 Volume ($\mu m^3$)')
    plt.ylabel('Count')
    plt.title('Mitochondria Size Distribution')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_1_size.svg", dpi=300)
    plt.close()

    # --- 2. SHAPE VS SIZE (Consistent Colors/Markers) ---
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df_clean, x='Log_Volume', y='Elongation',
                    hue='Group', style='Group',
                    palette=GROUP_PALETTE, markers=GROUP_MARKERS,
                    alpha=0.7, s=120, edgecolor='w')
    plt.xlabel('Log10 Volume ($\mu m^3$)')
    plt.ylabel('Elongation Index')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.title('Shape Consistency')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_2_shape.svg", dpi=300)
    plt.close()

    # --- 3. CONTRAST (Consistent Colors) ---
    plt.figure(figsize=(12, 10))
    sns.kdeplot(data=df_clean, x='Mean_Intensity', hue='Group',
                palette=GROUP_PALETTE,
                fill=True, alpha=0.3)
    plt.xlabel('Mean Intensity')
    plt.title('Contrast Consistency')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_3_contrast.svg", dpi=300)
    plt.close()

    # --- 4. HARDNESS (Volume Weighted) ---
    plt.figure(figsize=(12, 10))
    sc = plt.scatter(df_clean['DCI_Jitter'], df_clean['EFI_Jitter'], c=df_clean['Log_Volume'],
                     cmap='viridis', alpha=0.8, s=120, edgecolors='k', linewidth=0.5)
    cbar = plt.colorbar(sc)
    cbar.set_label('Log10 Volume')
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=1.2, color='gray', linestyle='--', alpha=0.5)

    plt.text(0.6, df_clean['EFI'].max() * 0.9, 'Hard: Fragile', fontsize=18, color='crimson')
    plt.text(-0.2, 0.5, 'Easy', fontsize=18, color='green')
    plt.xlabel('Dilation Collision Index (DCI)')
    plt.ylabel('Erosion Fragility Index (EFI)')
    plt.title('Hardness (Volume Weighted)')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_4_hardness_volume.svg", dpi=300)
    plt.close()

    # --- 5. HARDNESS BY GROUP (Consistent Colors/Markers) ---
    plt.figure(figsize=(14, 10))
    sns.scatterplot(data=df_clean, x='DCI_Jitter', y='EFI_Jitter',
                    hue='Group', style='Group',
                    palette=GROUP_PALETTE, markers=GROUP_MARKERS,
                    alpha=0.8, s=140, edgecolor='k')

    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=1.2, color='gray', linestyle='--', alpha=0.5)

    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title="Dataset", frameon=False)
    plt.xlabel('Dilation Collision Index (DCI)')
    plt.ylabel('Erosion Fragility Index (EFI)')
    plt.title('Hardness Distribution by Group')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_5_hardness_by_group.svg", dpi=300)
    plt.close()

    # --- 6. MASTER PANEL ---
    fig, axes = plt.subplots(2, 2, figsize=(22, 20))

    # A. Size
    sns.histplot(data=df_clean, x='Log_Volume', kde=True, color="#34495e", alpha=0.6, ax=axes[0, 0])
    axes[0, 0].set_title('A. Size Distribution', loc='left')

    # B. Shape (Consistent)
    sns.scatterplot(data=df_clean, x='Log_Volume', y='Elongation',
                    hue='Group', style='Group', palette=GROUP_PALETTE, markers=GROUP_MARKERS,
                    alpha=0.6, ax=axes[0, 1])
    axes[0, 1].set_title('B. Shape Consistency', loc='left')
    axes[0, 1].legend(frameon=False)

    # C. Contrast (Consistent)
    sns.kdeplot(data=df_clean, x='Mean_Intensity', hue='Group',
                palette=GROUP_PALETTE, fill=True, alpha=0.3, ax=axes[1, 0])
    axes[1, 0].set_title('C. Contrast Consistency', loc='left')
    axes[1, 0].legend(frameon=False)

    # D. Hardness (Volume)
    sc = axes[1, 1].scatter(df_clean['DCI_Jitter'], df_clean['EFI_Jitter'], c=df_clean['Log_Volume'],
                            cmap='viridis', alpha=0.7, s=80, edgecolors='k', linewidth=0.2)
    cbar = plt.colorbar(sc, ax=axes[1, 1])
    axes[1, 1].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(y=1.2, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('D. Segmentation Difficulty', loc='left')

    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_master_panel.svg", dpi=300)
    plt.close()


def generate_quadrant_summary(df, save_prefix="cumulative_analysis"):
    """
    Generates Modern Capsule Stacked Bars with % Text Labels.
    """
    print("\nGenerating Modern Capsule Stacked Bars with Labels...")

    def categorize(row): # values 1.2 and 0.5 threshold are based on mito-EM 2.0 paper
        if np.isnan(row['DCI']) or np.isnan(row['EFI']): return "Excluded"
        if row['DCI'] > 0.5 and row['EFI'] <= 1.2: return "Crowded" #  high-DCI/low-EFI objects are densely packed but topologically robust,
        if row['DCI'] <= 0.5 and row['EFI'] > 1.2: return "Fragile" #  low-DCI/high-EFI objects are isolated but topologically fragile  for example exhibiting constrictions, thin extensions, or mitochondrial nanotunnels
        if row['DCI'] > 0.5 and row['EFI'] > 1.2: return "Complex" # we assign this class ourselves
        return "Easy"  # low-DCI, low-EFI region, leaving a gap in coverage of cases prone to merge- and split-error

    df_cat = df.dropna(subset=['DCI', 'EFI']).copy()
    df_cat['Category'] = df_cat.apply(categorize, axis=1)

    summary = df_cat.groupby(['Group', 'Category']).size().reset_index(name='Count')
    total_per_group = df_cat.groupby('Group').size().reset_index(name='Total')
    summary = summary.merge(total_per_group, on='Group')
    summary['Percentage'] = (summary['Count'] / summary['Total']) * 100

    pivot_df = summary.pivot(index='Group', columns='Category', values='Percentage').fillna(0)
    order = ["Easy", "Crowded", "Fragile", "Complex"]
    for col in order:
        if col not in pivot_df.columns: pivot_df[col] = 0
    pivot_df = pivot_df[order]

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(10, 9))

    # Map colors from global palette
    bar_colors = [CATEGORY_PALETTE[c] for c in order]

    # 1. Standard Plot (Hidden later)
    pivot_df.plot(kind='bar', stacked=True, color=bar_colors, ax=ax, width=0.5, edgecolor='none', rot=0)

    # 2. Add Fancy Capsules and Text Labels
    new_patches = []

    for container_idx, container in enumerate(ax.containers):
        for bar in container:
            x, y = bar.get_xy()
            w, h = bar.get_width(), bar.get_height()

            # Draw ALL bars, even tiny ones, to avoid floating gaps
            # Dynamic rounding size to prevent artifacts on tiny bars
            r_size = 0.1 if h > 5 else 0.02

            # Create Capsule
            p = FancyBboxPatch((x, y), w, h,
                               boxstyle=f"round,pad=-0.01,rounding_size={r_size}",
                               mutation_scale=1,
                               fc=bar.get_facecolor(),
                               ec="white",
                               linewidth=2.5,
                               clip_on=False)
            new_patches.append(p)

            # Only add text if bar is tall enough to fit it
            if h > 5.0:
                cx = x + w / 2
                cy = y + h / 2
                ax.text(cx, cy, f"{h:.1f}%",
                        color='white', ha='center', va='center',
                        fontsize=14, fontweight='bold')

            # Hide original
            bar.set_visible(False)

    for p in new_patches:
        ax.add_patch(p)

    plt.title('Segmentation Difficulty by Group')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Dataset Group')

    # Manual Legend (Guarantees it appears correctly)
    legend_handles = [Patch(facecolor=CATEGORY_PALETTE[c], label=c) for c in order]
    plt.legend(handles=legend_handles, title='Category', bbox_to_anchor=(1.0, 1.0), loc='upper left', frameon=False)

    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_6_quadrant_counts_grouped.svg", dpi=300)
    plt.close()
    print(f"Saved modern capsule stacked bar plot with labels.")


if __name__ == "__main__":
    # DEFINE YOUR DIRECTORY HERE
    root_directory = "/media/samia/DATA/mounts/gpu2/mito-seg/catena_data/HEMI-OCTO-CORRECT/data_3d/train"

    zarr_paths = [os.path.join(root_directory, d) for d in os.listdir(root_directory)
                  if d.endswith('.zarr') or os.path.isdir(os.path.join(root_directory, d)) and 'zarr' in d]
    zarr_paths = sorted(zarr_paths)

    if not zarr_paths:
        print("No .zarr files found!")
        exit()

    print(f"Found {len(zarr_paths)} Zarr datasets.")
    all_dataframes = []

    for path in zarr_paths:
        sample_name = os.path.basename(path).replace('.zarr', '')
        print(f"\n--- Processing: {sample_name} ---")
        raw, labels = load_zarr_arrays(path, RAW_KEY, LABEL_KEY)
        if raw is not None and labels is not None:
            if raw.shape != labels.shape:
                print(f"Shape mismatch: {raw.shape} vs {labels.shape}. Skipping.")
                continue
            df_sample = calculate_mito_metrics(labels, raw, VOXEL_RES, sample_name)
            all_dataframes.append(df_sample)
            del raw
            del labels

    if all_dataframes:
        print("\nCombining data...")
        master_df = pd.concat(all_dataframes, ignore_index=True)
        master_df.to_csv("cumulative_mito_analysis.csv", index=False)
        generate_cumulative_plots(master_df, save_prefix="hemi-octo-grouped")
        generate_quadrant_summary(master_df, save_prefix="hemi-octo-grouped")
    else:
        print("No data processed.")
