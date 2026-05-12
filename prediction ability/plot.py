# %%
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# open all csv files in the current directory and store in a dict, where the key is the filename without extension
data_files = {}
for file in os.listdir('.'):
    if file.endswith('.csv'):
        key = os.path.splitext(file)[0]
        data_files[key] = pd.read_csv(file)

# %%

# --- build a long dataframe across all csvs and all methods ---
rows = []
for key, df in data_files.items():
    aux = key.split('_')
    n = int(aux[0][1:])
    k = int(aux[1][1:])

    # keep all methods
    df2 = df.copy()
    df2["n"] = n
    df2["k"] = k
    rows.append(df2[["n", "k", "method", "pred_mse"]])

df_long = pd.concat(rows, ignore_index=True)

# Make categorical ordering stable
n_order = sorted(df_long["n"].unique())
k_order = sorted(df_long["k"].unique())

# --- faceted boxplots ---
g = sns.catplot(
    data=df_long,
    x="n", y="pred_mse",
    hue="k",
    col="method",
    kind="box",
    col_wrap=3,          # adjust to taste
    order=n_order,
    hue_order=k_order,
    height=3.2,
    aspect=1.1,
    sharey=True
)

g = sns.catplot(
    data=df_long,
    x="n", y="pred_mse",
    hue="k",
    col="method",
    kind="box",
    col_wrap=3,
    order=n_order,
    hue_order=k_order,
    height=3.2,
    aspect=1.1,
    sharey=True
)

# Overlay means as red X
g.map_dataframe(
    sns.pointplot,
    x="n", y="pred_mse",
    hue="k",
    order=n_order,
    hue_order=k_order,
    estimator=np.mean,
    errorbar=None,       # seaborn>=0.12
    dodge=0.8,           # align with box dodge
    join=False,
    markers="x",
    linestyles="",
    color="red"
)

g.set(yscale="log")
g.set_axis_labels("n", "pred_mse")
# catplot already has a legend; pointplot adds another -> keep only one
g._legend.remove()
g.add_legend(title="k")
plt.tight_layout()
plt.show()


# log scale on all facets
g.set(yscale="log")
g.set_axis_labels("n", "pred_mse")
g.add_legend(title="k")
plt.tight_layout()
plt.show()


# %%


n_order = sorted(df_long["n"].unique())
k_order = sorted(df_long["k"].unique())

g = sns.catplot(
    data=df_long,
    x="n", y="pred_mse",
    hue="method",
    col="k",
    kind="box",
    col_order=k_order,
    order=n_order,
    height=3.4,
    aspect=1.1,
    sharey=True
)

g.set(yscale="log")
g.set_axis_labels("n", "pred_mse")
g.add_legend(title="method")
plt.tight_layout()
plt.show()


# %%


mpl.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "legend.title_fontsize": 18,
    "figure.titlesize": 22,
    "lines.linewidth": 2.5,
    "lines.markersize": 10,
})



# decide method order
method_order = list(df_long["method"].unique())  # or sorted(...)

# let seaborn choose colors, but keep them stable by giving the order
palette = sns.color_palette(n_colors=len(method_order))
color_map = dict(zip(method_order, palette))

g = sns.catplot(
    data=df_long,
    x="n", y="pred_mse",
    hue="method",
    hue_order=method_order,
    palette=color_map,
    col="k",
    kind="box",
    col_order=k_order,
    order=n_order,
    height=5.0,
    aspect=1.1,
    sharey=True,
    legend=False
)

g.set(yscale="log")
g.set_axis_labels("sample size", "prediction MSE")

# --- proxy legend handles (colored squares) ---
handles = [mpatches.Patch(color=color_map[m], label=m) for m in method_order]

ncol = min(len(method_order), 4)  # choose what fits

g.fig.legend(
    handles=handles,
    title="",
    loc="lower center",
    bbox_to_anchor=(0.5, -0.00),
    ncol=ncol,
    frameon=False
)

g.fig.subplots_adjust(bottom=0.25)
plt.show()


# %%
df_long

# %%
df_long['stringa'] = df_long.apply(lambda x: f"N={x['n']} K={x['k']}", axis=1)

# %%
# sort df_long by n and k
df_long = df_long.sort_values(by=['n', 'k'])

# %%

def build_nk_palette(
    df,
    n_col="n",
    k_col="k",
    label_fmt="N{n}_K{k}",
    s_min=0.10,
    s_max=1.00,
    s_power=1.0,   # >1 emphasizes differences among larger N; <1 emphasizes smaller N
    cmap_name="tab10",
):
    n_vals = np.sort(df[n_col].unique())
    k_vals = np.sort(df[k_col].unique())

    # Map each K to a distinct base hue (from a colormap)
    cmap = plt.get_cmap(cmap_name)
    # k_to_rgb = {k: cmap(i)[:3] for i, k in enumerate(k_vals)}  # RGB in [0,1]
    k_to_rgb = {
        2: cmap(0)[:3],
        3: cmap(1)[:3],
        4: cmap(3)[:3],
    }

    # Map each N to a saturation value automatically (rank-based)
    if len(n_vals) == 1:
        n_to_s = {n_vals[0]: (s_min + s_max) / 2}
    else:
        t = np.linspace(0, 1, len(n_vals)) ** s_power
        s_list = s_min + (s_max - s_min) * t
        n_to_s = dict(zip(n_vals, s_list))

    # Build palette dict for every (N,K) label
    palette = {}
    for n in n_vals:
        for k in k_vals:
            base_rgb = k_to_rgb[k]
            h, s, v = mcolors.rgb_to_hsv(base_rgb)
            new_rgb = mcolors.hsv_to_rgb((h, n_to_s[n], v))
            label = label_fmt.format(n=n, k=k)
            palette[label] = mcolors.to_hex(new_rgb)

    return palette


# %%


# Make a label like your old "subdir": N10_K2 etc.
df_long = df_long.copy()
df_long["subdir"] = df_long.apply(lambda r: f"N{int(r['n'])}_K{int(r['k'])}", axis=1)

# Dynamic palette: Hue = K, Saturation = N
colors = build_nk_palette(df_long, n_col="n", k_col="k", label_fmt="N{n}_K{k}",
                          s_min=0.20, s_max=1.00, s_power=1.0, cmap_name="tab10")




# %%

def build_nk_palette_and_maps(
    df, n_col="n", k_col="k",
    s_min=0.20, s_max=1.00, s_power=1.0,
    cmap_name="tab10",
):
    n_vals = np.sort(df[n_col].unique())
    k_vals = np.sort(df[k_col].unique())

    # base hues for each K
    cmap = plt.get_cmap(cmap_name)
    # k_to_rgb = {k: cmap(i)[:3] for i, k in enumerate(k_vals)}
    k_to_rgb = {
        2: cmap(0)[:3],
        3: cmap(1)[:3],
        4: cmap(3)[:3],
    }

    # saturation for each N (rank-based)
    if len(n_vals) == 1:
        n_to_s = {n_vals[0]: (s_min + s_max) / 2}
    else:
        t = (np.linspace(0, 1, len(n_vals)) ** s_power)
        s_list = s_min + (s_max - s_min) * t
        n_to_s = dict(zip(n_vals, s_list))

    # full palette for every (N,K)
    palette = {}
    for n in n_vals:
        for k in k_vals:
            h, s, v = mcolors.rgb_to_hsv(k_to_rgb[k])
            new_rgb = mcolors.hsv_to_rgb((h, n_to_s[n], v))
            palette[f"N{int(n)}_K{int(k)}"] = mcolors.to_hex(new_rgb)

    return palette, n_vals, k_vals, n_to_s, k_to_rgb


# --- prepare labels like before ---
dfp = df_long.copy()
dfp["subdir"] = dfp.apply(lambda r: f"N{int(r['n'])}_K{int(r['k'])}", axis=1)

# --- palette + maps ---
colors, n_vals, k_vals, n_to_s, k_to_rgb = build_nk_palette_and_maps(
    dfp, n_col="n", k_col="k",
    s_min=0.10, s_max=1.00, s_power=1.0, cmap_name="tab10"
)

# --- plot (single axis, log scale) ---
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(
    data=dfp,
    x="method",
    y="pred_mse",
    hue="subdir",
    palette=colors,
    ax=ax
)

ax.set_yscale("log")
ax.set_ylabel(r"$Pred\text{-}MSE$")
ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# remove seaborn's combined legend
if ax.legend_ is not None:
    ax.legend_.remove()

# --- Legend for K (hue) ---
k_handles = []
k_labels = []
for k in k_vals:
    h, s, v = mcolors.rgb_to_hsv(k_to_rgb[k])
    rgb = mcolors.hsv_to_rgb((h, 1.0, v))  # full saturation for clarity
    k_handles.append(mpatches.Patch(color=mcolors.to_hex(rgb)))
    k_labels.append(f"K={int(k)}")

leg_k = ax.legend(
    k_handles, k_labels,
    title="",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=len(k_vals),
    frameon=False
)
ax.add_artist(leg_k)

# --- Legend for N (saturation) ---
# pick a single reference hue (e.g., first K) and vary saturation by N
k_ref = k_vals[0]
h_ref, s_ref, v_ref = mcolors.rgb_to_hsv(k_to_rgb[k_ref])

n_handles = []
n_labels = []
for n in n_vals:
    rgb = mcolors.hsv_to_rgb((h_ref, n_to_s[n], v_ref))
    n_handles.append(mpatches.Patch(color=mcolors.to_hex(rgb)))
    n_labels.append(f"N={int(n)}")

ncol_n = min(len(n_vals), 6)  # adjust if many N values
ax.legend(
    n_handles, n_labels,
    title="",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.24),
    ncol=ncol_n,
    frameon=False
)
ax.set_xlabel('')


# make room for both legends
fig.subplots_adjust(bottom=0.32)

ax.set_xticklabels(
    [
        r'$\text{BMLF}_{\text{Fourier}}$',
        r'$\text{BMLF}_{\text{GP}}$',
        r'$\text{PLS}$',
        r'$\text{fPLS}_{\text{Fourier}}$',
    ]
)

plt.show()


# %%
# export figure in high resolution
fig.savefig("prediction_mse_boxplot_nk_palette.png", dpi=600, bbox_inches='tight')

# %%



