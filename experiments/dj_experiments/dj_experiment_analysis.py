# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

# from probcs.datajoint.exc_tables import ExcConfig, ExcResult, schema
from probcs.datajoint.grating_tables import GratingConfig, GratingResult2


# %%

config_table = GratingConfig()

result = config_table * GratingResult2()

g_dim_restr = 4
# %%
df = result.fetch(download_path="/tmp", as_dict=True)
# %%

g_type = "exc stimuli"

zero_line_color = "black"
zero_line_linestyle = "dashed"
zero_line_linewidth = 1

sde_alpha = 0.2
mean_perc_change_linestyle = "dotted"
mean_perc_change_linewidth = 1

xlabel_fontsize = 10
ylabel_fontsize = 10

xtickfontsize = 8
ytickfontsize = 8

title_fontsize = 12

markers = [f"${i+1}$" for i in range(len(df))]
fig, axs = plt.subplots(
    nrows=int(g_dim_restr / 2),
    ncols=int(g_dim_restr / 2),
    dpi=300,
    sharex=True,
    # sharey=True,
)
fig_legend, ax_legend = plt.subplots(dpi=300)

for row_idx, (row, marker) in enumerate(zip(df, markers)):
    print(row_idx)
    text = f"{row_idx + 1}: g_dim {row['g_dim']} g_prob {row['g_prob']} x_sigma {row['x_sigma']} i_sigma {row['i_sigma']} patterns_offset {row['patterns_offset']} n_tune {row['n_tune']} n_draws {row['n_draws']}\n"
    ax_legend.text(
        0.0,
        1 - 0.2 * row_idx,
        text,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=10,
    )
    ax_legend.axis("off")
    x_perc_change_means = np.array(row["all_center_x_perc_change_means"])
    x_perc_change_means_sde = np.array(row["all_center_x_perc_change_means_sde"])
    for dim, ax in enumerate(axs.flatten()):
        ax.plot(x_perc_change_means[dim], linestyle="dotted", marker=marker)
        ax.fill_between(
            np.arange(len(x_perc_change_means[dim])),
            x_perc_change_means[dim] - x_perc_change_means_sde[dim],
            x_perc_change_means[dim] + x_perc_change_means_sde[dim],
            alpha=sde_alpha,
        )
        ax.set_xticks(range(0, g_dim_restr))
        xtick_labels = ["C"] + [f"D{i+1}" for i in range(g_dim_restr - 1)]
        ax.set_xticklabels(xtick_labels, fontsize=xtickfontsize)
        # draw a horizontal line at 0
        ax.axhline(
            y=0,
            color=zero_line_color,
            linestyle=zero_line_linestyle,
            linewidth=zero_line_linewidth,
        )
# figure sup x label
fig.supxlabel("Stimulus type", fontsize=xlabel_fontsize)
fig.supylabel("Percentage change in mean response w.r.t MEI", fontsize=ylabel_fontsize)
fig.suptitle(f"Surround modulation using {g_type}", fontsize=title_fontsize)

# %%
fig_stimuli, axs_stimuli = plt.subplots(
    nrows=g_dim_restr,
    ncols=g_dim_restr + 1,
    dpi=300,
    # sharey=True,
)
with open(row["all_stimuli"], "rb") as f:
    all_stimuli = pickle.load(f)

for ax_idx, axs_row_stimuli in enumerate(axs_stimuli[0].flatten()):
    stimulus_set = all_stimuli[ax_idx]
    for stimulus, ax in zip(stimulus_set, axs_row_stimuli.flatten()):
        ax.imshow(stimulus, cmap="gray")
        ax.axis("off")

    # %%
