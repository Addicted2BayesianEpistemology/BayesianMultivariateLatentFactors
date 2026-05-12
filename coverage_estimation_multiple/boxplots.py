# %%
import numpy as np
import pandas as pd

import os

# create a list of all the subdriectories of the directory Explorations
subdirs = [d for d in os.listdir('Explorations') if os.path.isdir(os.path.join('Explorations', d))]

subdirs

# %%
dataframes = {}
for subdir in subdirs:
    dataframes[subdir] = pd.read_csv(os.path.join('Explorations', subdir, 'combined_coverage_all_methods.csv'))

# %%
dataframes

# %%
# create a single dataframe with a column called subdir, one called method, which is taken from the datafame and the coverage_pct and mse
combined_df = pd.DataFrame()
for subdir, df in dataframes.items():
    df['subdir'] = subdir
    combined_df = pd.concat([combined_df, df], ignore_index=True)

combined_df

# %%
import matplotlib.colors as mcolors

# 1. Define the base colors (Hue depends on K)
colors = {
    'N10_K2': 'C0', 'N10_K3': 'C1', 'N10_K4': 'C3',
    'N25_K2': 'C0', 'N25_K3': 'C1', 'N25_K4': 'C3',
    'N50_K2': 'C0', 'N50_K3': 'C1', 'N50_K4': 'C3',
    'N75_K2': 'C0', 'N75_K3': 'C1', 'N75_K4': 'C3',
    'N100_K2': 'C0', 'N100_K3': 'C1', 'N100_K4': 'C3',
}

# 2. Iterate and modify Saturation in HSV space
for key in colors:
    # Get the RGB values of the base color (e.g., 'C0')
    base_rgb = mcolors.to_rgb(colors[key])
    
    # Convert RGB to HSV [Hue, Saturation, Value]
    # hsv is a tuple: (h, s, v)
    hsv = mcolors.rgb_to_hsv(base_rgb)
    
    # Determine new Saturation (S) based on N
    # Lower S = "Lighter"/Pastel, Higher S = "Darker"/Intense
    if 'N100' in key:
        new_s = 1.0  # Light (30% saturation)
    elif 'N25' in key:
        new_s = 0.35  # Medium (60% saturation)
    elif 'N50' in key:
        new_s = 0.55  # Dark/Intense (100% saturation)
    elif 'N75' in key:
        new_s = 0.75  # Dark/Intense (100% saturation)
    elif 'N10' in key:
        new_s = 0.2  # Dark/Intense (100% saturation)
    else:
        new_s = hsv[1] # Default to original if unknown
        
    # Create the new HSV tuple (keeping Hue and Value original)
    new_hsv = (hsv[0], new_s, hsv[2])
    
    # Convert back to RGB, then to Hex string for easy storage/use
    new_rgb = mcolors.hsv_to_rgb(new_hsv)
    colors[key] = mcolors.to_hex(new_rgb)

# Verification
import pprint
pprint.pprint(colors)

# %%
# export figure in high resolution
fig.savefig('boxplots_mse_coverage.png', dpi=600, bbox_inches='tight')

# %%


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# --- Global font scale ---
sns.set_context("notebook", font_scale=1.5)

fig, axs = plt.subplots(
    2, 1,
    figsize=(10, 9),
    sharex=True,
    gridspec_kw={'hspace': 0.08}
)

# --- Plot 1: MSE ---
sns.boxplot(
    data=combined_df,
    x='method',
    y='mse',
    palette=colors,
    hue='subdir',
    ax=axs[0]
)
axs[0].set_yscale('log')
axs[0].legend_.remove()
axs[0].grid(True, axis='y', linestyle='--', linewidth=0.5)
axs[0].set_ylabel('MSE')
axs[0].set_xlabel('')

# --- Plot 2: Coverage ---
sns.boxplot(
    data=combined_df,
    x='method',
    y='coverage_pct',
    hue='subdir',
    palette=colors,
    ax=axs[1]
)

handles, labels = axs[1].get_legend_handles_labels()
new_labels = [
    label.replace('_', ' ').replace('N', 'N=').replace('K', 'K=')
    for label in labels
]

axs[1].legend(
    handles,
    new_labels,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.2),
    ncol=3
)

axs[1].axhline(80, color='C2', linestyle='--')
axs[1].grid(True, axis='y', linestyle='--', linewidth=0.5)
axs[1].set_ylabel(r'$\mathfrak{C}_{80\%}$')

# --- Custom x tick labels ---
axs[1].set_xticklabels([
    r'$\text{BMLF}_{\text{GP}}$',
    r'$\text{BMLF}_{\text{BS}}$',
    r'$\text{PLS}$',
    r'$\text{fPLS}_{\text{GP}}$',
    r'$\text{fPLS}_{\text{BS}}$',
])

# --- Panel labels (a), (b) outside upper-left corners ---
axs[0].text(
    -0.10, 1.03, '(a)',
    transform=axs[0].transAxes,
    fontsize=18,
    fontweight='bold',
    ha='left',
    va='bottom'
)

axs[1].text(
    -0.10, 1.03, '(b)',
    transform=axs[1].transAxes,
    fontsize=18,
    fontweight='bold',
    ha='left',
    va='bottom'
)

# --- Final spacing ---
fig.subplots_adjust(
    left=0.12,
    right=0.98,
    top=0.97,
    bottom=0.18,
    hspace=0.08
)

plt.show()

# %%
# export figure in high resolution
fig.savefig('boxplots_mse_coverage.png', dpi=600, bbox_inches='tight')

# %%



