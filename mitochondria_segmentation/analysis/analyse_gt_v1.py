"""
Note:
Value on X-Axis,Real Volume (μm3),Meaning
0,1.0,A large mitochondrion
-1,0.1,A medium/small mitochondrion
-2,0.01,A small fragment
-3,0.001,A tiny speck (likely noise)
-infinity,0.0,Empty space (Volume = 0)

"""

import numpy as np
import pandas as pd
import zarr
import tifffile
import os
from skimage.measure import regionprops
from skimage.morphology import binary_dilation, binary_erosion, ball
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION FOR PUBLICATION PLOTS ---
# Setting large fonts and clear styles as requested
plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 25,
    'axes.labelsize': 22,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
    'figure.titlesize': 25,
    'font.family': 'sans-serif'
})
sns.set_style("ticks")

def load_data(path, key_path=None):
    """
    Loads data from Tiff or Zarr.
    For Zarr, 'key_path' specifies the internal path (e.g., 'volumes/labels').
    """
    if path.endswith('.tiff') or path.endswith('.tif'):
        print(f"Loading TIFF: {path}...")
        return tifffile.imread(path)
    elif path.endswith('.zarr') or os.path.isdir(path):
        print(f"Loading Zarr: {path} [{key_path}]...")
        f = zarr.open(path, mode='r')
        if key_path:
            return f[key_path][:]
        else:
            return f[:]
    else:
        raise ValueError("Unsupported format. Use .tiff or .zarr")

def calculate_mito_metrics(label_vol, raw_vol=None, voxel_res=(30, 8, 8)):
    """
    Calculates size, shape, contrast, and MitoEM 2.0 metrics (DCI/EFI).
    voxel_res: (z, y, x) resolution in nm.
    """
    print("Calculating Region Properties (this may take a moment)...")
    
    # Ensure label volume is integer
    label_vol = label_vol.astype(int)
    
    # 1. Standard RegionProps (Size, Shape, Intensity)
    # We pass raw_vol to get intensity (contrast) metrics
    props = regionprops(label_vol, intensity_image=raw_vol)
    
    data = []
    
    # Structuring element for DCI/EFI (3x3x3 ball as per MitoEM paper)
    selem = ball(1) 
    
    print(f"Analyzing {len(props)} mitochondria instances...")
    
    for idx, prop in enumerate(props):
        # -- Basic Metrics --
        # Size
        vol_voxels = prop.area
        vol_phys = vol_voxels * np.prod(voxel_res) / 1e9  # Volume in cubic microns
        
        # Elongation (using inertia tensor eigenvalues)
        # eigenvalues: l1 >= l2 >= l3. Elongation approx sqrt(l1 / l3)
        eig_vals = prop.inertia_tensor_eigvals
        if eig_vals[-1] == 0:
            elongation = 0 # Handle degenerate cases
        else:
            elongation = np.sqrt(eig_vals[0] / eig_vals[-1])
            
        # Contrast / Intensity
        # Mean intensity of the mito vs assumption of background is difficult without a background mask.
        # Here we take mean intensity of the organelle itself.
        mean_intensity = prop.mean_intensity if raw_vol is not None else np.nan
        
        # -- MitoEM 2.0 Advanced Metrics (DCI & EFI) --
        # We process these locally (bbox) to save memory/time
        
        # Extract local crops
        min_z, min_y, min_x, max_z, max_y, max_x = prop.bbox
        local_mask = prop.image # Binary mask of the object in bounding box
        
        # [cite_start]DCI: Dilation Collision Index [cite: 90]
        # Dilate current mask
        dilated_mask = binary_dilation(local_mask, selem)
        
        # Get the corresponding crop from the original label volume
        local_labels_crop = label_vol[min_z:max_z, min_y:max_y, min_x:max_x]
        
        # Find labels overlapping with the dilated mask
        # We mask the label crop with the *dilated* binary mask
        overlapping_labels = local_labels_crop[dilated_mask]
        unique_neighbors = np.unique(overlapping_labels)
        
        # DCI = count of unique neighbors excluding 0 (bg) and self (prop.label)
        dci = np.sum((unique_neighbors != 0) & (unique_neighbors != prop.label))
        
        # [cite_start]EFI: Erosion Fragility Index [cite: 129]
        # Erode current mask
        eroded_mask = binary_erosion(local_mask, selem)
        
        # Count connected components in eroded mask
        # 26-connectivity for 3D
        if not np.any(eroded_mask):
             # If erosion wipes it out completely, it is very fragile (or small)
             # We treat this as high fragility. 
             # Technically 0 components, but logically implies it broke/vanished.
             # The paper counts components, so strictly 0. 
             efi = 0 
        else:
            from skimage.measure import label as separate_components
            _, efi = separate_components(eroded_mask, return_num=True, connectivity=3)
            
        data.append({
            'Label_ID': prop.label,
            'Volume_um3': vol_phys,
            'Voxel_Count': vol_voxels,
            'Elongation': elongation,
            'Mean_Intensity': mean_intensity,
            'DCI': dci,
            'EFI': efi
        })
        
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(props)}...")

    return pd.DataFrame(data)

def generate_publication_plots(df, save_prefix="mito_analysis"):
    """
    Generates publication-ready plots including CONTRAST analysis.
    """
    # Create a layout with 2 rows, 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Use log volume for better visualization
    df['Log_Volume'] = np.log10(df['Volume_um3'] + 1e-9)

    # --- PLOT 1: Size Distribution (Top Left) ---

    # "Hemibrain": "#3498db",  # Blue
    # "Octo": "#e74c3c",  # Red
    # sns.histplot(data=df, x='Log_Volume', kde=True, color="#2c3e50", alpha=0.6, ax=axes[0, 0]) # original
    sns.histplot(data=df, x='Log_Volume', kde=True, color="#3498db", alpha=0.6, ax=axes[0, 0]) # hemibrain
    # sns.histplot(data=df, x='Log_Volume', kde=True, color="#e74c3c", alpha=0.6, ax=axes[0, 0]) # octo
    axes[0, 0].set_xlabel('Log10 Volume ($\mu m^3$)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('A. Mitochondria Size Distribution', loc='left', fontweight='bold')

    # --- PLOT 2: Shape Analysis (Top Right) ---
    sns.scatterplot(data=df, x='Log_Volume', y='Elongation',
                    alpha=0.6, color="#e74c3c", edgecolor='w', s=80, ax=axes[0, 1])
    axes[0, 1].set_xlabel('Log10 Volume ($\mu m^3$)')
    axes[0, 1].set_ylabel('Elongation Index')
    axes[0, 1].set_title('B. Shape: Volume vs. Elongation', loc='left', fontweight='bold')

    # --- PLOT 3: Contrast/Intensity Distribution (Bottom Left) ---
    # This shows the "darkness" or "lightness" of the mitochondria
    if df['Mean_Intensity'].isnull().all():
        axes[1, 0].text(0.5, 0.5, "No Raw Data Provided\n(Cannot compute Contrast)",
                        ha='center', va='center', fontsize=14)
    else:
        sns.histplot(data=df, x='Mean_Intensity', kde=True, color="#27ae60", alpha=0.6, ax=axes[1, 0])
        axes[1, 0].set_xlabel('Mean Intensity (0-255)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('C. Intensity Contrast Distribution', loc='left', fontweight='bold')

    # --- PLOT 4: Intensity vs Volume (Bottom Right) ---
    # Helps spot False Positives: irregular fragments often have different intensity
    if df['Mean_Intensity'].isnull().all():
        axes[1, 1].text(0.5, 0.5, "No Raw Data Provided",
                        ha='center', va='center', fontsize=14)
    else:
        sc = axes[1, 1].scatter(df['Log_Volume'], df['Mean_Intensity'],
                                c=df['Elongation'], cmap='magma', alpha=0.7, s=80, edgecolors='k', linewidth=0.5)
        plt.colorbar(sc, ax=axes[1, 1], label='Elongation')
        axes[1, 1].set_xlabel('Log10 Volume ($\mu m^3$)')
        axes[1, 1].set_ylabel('Mean Intensity')
        axes[1, 1].set_title('D. Intensity vs. Size', loc='left', fontweight='bold')

    sns.despine()
    plt.tight_layout()

    # Save the combined figure
    save_path = f"{save_prefix}_combined_panel.png"
    save_path = f"{save_prefix}_combined_panel.svg"
    plt.savefig(save_path, dpi=300)
    print(f"Saved combined plot to {save_path}")
    plt.show()



# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # REPLACE THESE PATHS WITH YOUR DATA
    vol_path = "/media/samia/DATA/mounts/gpu2/mito-seg/catena_data/HEMI-AL/data_3d/train/hemi_x30725-31237_y31431-31943_z26420-26932.zarr"
    lab_path = "/media/samia/DATA/mounts/gpu2/mito-seg/catena_data/HEMI-AL/data_3d/train/hemi_x30725-31237_y31431-31943_z26420-26932.zarr"
    
    # Example for Zarr:
    raw = load_data(vol_path, key_path='volumes/raw_clahe')
    labels = load_data(lab_path, key_path='volumes/labels/mito_ids_relab')
    
    # DUMMY DATA FOR DEMONSTRATION (Remove this block when using real data)
    # print("Generating dummy data for demonstration...")
    # raw = np.random.rand(100, 256, 256)
    # labels = np.zeros((100, 256, 256), dtype=int)
    # # Create some blobs
    # from skimage.draw import ellipsoid
    # for i in range(1, 20):
    #     rr, cc, dd = ellipsoid(np.random.randint(5,15), np.random.randint(5,15), np.random.randint(5,15))
    #     # place randomly
    #     z, y, x = np.random.randint(0, 100-30), np.random.randint(0, 256-30), np.random.randint(0, 256-30)
    #     labels[z:z+rr.shape[0], y:y+rr.shape[1], x:x+rr.shape[2]] = i * (rr != 0)

    # Run Analysis
    df_metrics = calculate_mito_metrics(labels, raw_vol=raw, voxel_res=(8, 8, 8))

    # Save statistics to CSV
    df_metrics.to_csv(f"{os.path.basename(vol_path)}_mito_analysis_results.csv", index=False)
    print("Analysis complete. Saved to mito_analysis_results.csv")
    
    # Plot
    generate_publication_plots(df_metrics, save_prefix=f"{os.path.basename(vol_path)}_mito_analysis")
