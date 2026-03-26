import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import os
import re
import glob

def plot_and_average_csv(
    prefix,
    sub_title,
    xscale,
    total_title,
    xlabel,
    ylabel,
    folder=".",
    # --- styling knobs (defaults chosen for 4-up layout) ---
    linewidth=3,
    label_fontsize=18,
    tick_fontsize=18,
    spine_width=2.5,
):
    """
    Reads {prefix}*.csv files, plots each run, computes an averaged curve over x, plots mean,
    and saves mean to {prefix}_mean.csv.

    Styling is set for compact, succinct figures: thicker lines, no grid, minimal spines,
    larger text for 4-figure-per-page layouts.
    """

    pattern = os.path.join(folder, f"{prefix}*.csv")
    files = sorted(glob.glob(pattern))

    num_pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.csv$")

    all_dfs = []

    if isinstance(xscale, (int, float)):
        xscales = [xscale] * len(files)
    elif isinstance(xscale, list):
        xscales = xscale
    else:
        print("unrecognized xscale")
        return

    def _style_axes(ax):
        """Apply supervisor-requested succinct style to a single Matplotlib Axes."""
        # No gridlines
        ax.grid(False)

        # Keep only left & bottom spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Make remaining spines slightly thicker
        ax.spines["left"].set_linewidth(spine_width)
        ax.spines["bottom"].set_linewidth(spine_width)

        # Tick styling: larger text, thicker ticks
        ax.tick_params(axis="both", which="major",
                       labelsize=tick_fontsize, width=spine_width)

        # Labels: larger relative to small figure size
        ax.xaxis.label.set_size(label_fontsize)
        ax.yaxis.label.set_size(label_fontsize)

    # -----------------------
    # Plot each individual run
    # -----------------------
    for f, scale in zip(files, xscales):
        basename = os.path.basename(f)
        m = num_pattern.search(basename)
        if not m:
            continue

        suffix = m.group(1)
        df = pd.read_csv(f, header=None)

        x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        y = pd.to_numeric(df.iloc[:, 1], errors="coerce")

        mask = (~x.isna()) & (~y.isna())
        x = x[mask] * scale
        y = y[mask]

        # Use fig, ax instead of global plt state (cleaner for multi-fig workflows)
        fig, ax = plt.subplots()
        ax.plot(x, y, linewidth=linewidth)

        ax.set_title(f"{sub_title}{suffix}", fontsize=label_fontsize)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        _style_axes(ax)
        fig.tight_layout()
        plt.show()

        all_dfs.append(pd.DataFrame({"x": x, "y": y}))

    if not all_dfs:
        print("No found csv file, example:", f"{prefix}1.csv, {prefix}2.csv ...")
        return

    # -----------------------
    # Compute and plot the mean
    # -----------------------
    all_data = pd.concat(all_dfs, axis=0)
    mean_df = all_data.groupby("x", as_index=False)["y"].mean()

    fig, ax = plt.subplots()
    ax.plot(mean_df["x"], mean_df["y"], linewidth=linewidth)

    ax.set_title(total_title, fontsize=label_fontsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    _style_axes(ax)
    fig.tight_layout()

    # Save mean curve
    if len(files) > 1:
        mean_csv_path = os.path.join(folder, f"{prefix}_mean.csv")
        mean_df.to_csv(mean_csv_path, index=False)
        print(f"Average saved: {mean_csv_path}")

files = "grid_1_mi"
legends = ["MI"]
xlabel = "training episodes (for agents)"
ylabel = "Mutual information esitmation"
title = "MI estimate over different runs"
folder = r"D:\Personal folder\University\Projects\4th year\belief-rnn\report\3.19"
xscale = [5/400]
plot_and_average_csv(
    files,
    title,
    xscale,
    title,
    xlabel,
    ylabel,
    folder)