# beast: conda activate funkelsd

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

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


def extract_class(name):
    """Extracts the algorithm class (e.g., MutexWS, GASP, SimpleITK)."""
    if 'beta' in str(name):
        return name.split('beta')[0]
    return "Other"

def rank_and_visualize(csv_path, top_n_per_class=2):
    # Load data using your provided path
    df = pd.read_csv(csv_path)

    # 1. Feature Engineering (Preserving your sum_vi logic)
    df['sum_vi'] = df['false_splits'] + df['false_merges']
    df['mean_vi'] = df['sum_vi'] / 2
    df['f1_score'] = (2 * df['precision'] * df['recall']) / (df['precision'] + df['recall'])
    df['f1_score'] = df['f1_score'].fillna(0)
    df['class'] = df['folder_name'].apply(extract_class)

    # 2. Find Top N per Config Class
    # Logic: Group by class, sort by F1 score (desc), pick head
    top_df = df.sort_values(['class', 'f1_score', 'rand_error'], ascending=[True, False, True])
    top_per_class = top_df.groupby('class').head(top_n_per_class).reset_index(drop=True)
    
    # Save the specific top configs
    top_per_class.to_csv("top_configs_per_class.csv", index=False)
    print(f"Top {top_n_per_class} per class saved to 'top_configs_per_class.csv'")

    # 3. Publication Quality Pareto Plot
    # Set global style for academic standards
    plt.rcParams.update({
        'font.size': 16, 
        'font.family': 'sans-serif',
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5
    })
    
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use a high-contrast, publication-worthy palette
    palette = sns.color_palette("bright", n_colors=df['class'].nunique())
    
    # Plot all points faintly to show the search space
    sns.scatterplot(
        data=df, x='sum_vi', y='f1_score', hue='class', 
        alpha=0.2, palette=palette, s=120, ax=ax, legend=False
    )

    # Plot Pareto-dominant candidates prominently
    scatter = sns.scatterplot(
        data=top_per_class, x='sum_vi', y='f1_score', hue='class', 
        palette=palette, s=300, edgecolor='black', linewidth=2, ax=ax
    )

    # Styling for clarity
    ax.set_title("Pareto Front: Segmentation Performance by Method Class", fontweight='bold', pad=20)
    ax.set_xlabel("Sum Variation of Information (Lower is Better)", fontweight='bold', labelpad=12)
    ax.set_ylabel("F1-Score (Higher is Better)", fontweight='bold', labelpad=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # Place legend with larger font
    plt.legend(title='Method Class', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=25, frameon=False)
    
    sns.despine()

    # Save outputs as publication formats
    plt.tight_layout()
    plt.savefig("segmentation_pareto_v2.png", dpi=300, bbox_inches='tight')
    plt.savefig("segmentation_pareto_v2.pdf", bbox_inches='tight')
    
    print("Visualizations saved as PNG (300 DPI) and PDF.")
    plt.show()

if __name__ == "__main__":
    # Preserving your exact data path
    PATH = "/media/samia/DATA/plant-seg/segmentation-runs/data-20-200-200-3Drun/evaluation_metrics.csv"
    
    if os.path.exists(PATH):
        rank_and_visualize(PATH, top_n_per_class=2)
    else:
        print(f"Error: Could not find file at {PATH}")
