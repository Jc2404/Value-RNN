from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGSIZE = (6, 4)
LINEWIDTH = 2.0
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 10
SHADE_ALPHA = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read all CSV files in a folder, use the first column as x and the first y column "
            "(second column overall), then plot the across-file mean with a shaded uncertainty band."
        )
    )
    parser.add_argument("--input-dir", required=True, type=str, help="Folder containing CSV files.")
    parser.add_argument("--x-label", required=True, type=str, help="X-axis label.")
    parser.add_argument("--y-label", required=True, type=str, help="Y-axis label.")
    parser.add_argument(
        "--output-name",
        required=True,
        type=str,
        help="Output filename stem to use for both the PNG and aggregate CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output folder. Defaults to the input folder.",
    )
    parser.add_argument(
        "--file-glob",
        type=str,
        default="*.csv",
        help="Glob used to discover input files inside the folder.",
    )
    parser.add_argument(
        "--shade",
        choices=("sem", "std"),
        default="sem",
        help="Statistic used for the shaded band. Default: sem.",
    )
    parser.add_argument(
        "--x-multiplier",
        type=float,
        default=1.0,
        help="Multiply the x-axis values by this factor before aggregation and plotting. Default: 1.0.",
    )
    return parser.parse_args()


def discover_csv_paths(input_dir: Path, file_glob: str) -> list[Path]:
    csv_paths = sorted(path for path in input_dir.glob(file_glob) if path.is_file())
    if not csv_paths:
        raise FileNotFoundError(f"No files matching '{file_glob}' were found in {input_dir}.")
    return csv_paths


def read_xy_from_csv(csv_path: Path, x_multiplier: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"{csv_path} must contain at least two columns.")

    x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    frame = pd.DataFrame({"x": x, "y": y}).dropna()

    if frame.empty:
        raise ValueError(f"{csv_path} does not contain numeric data in its first two columns.")

    frame["x"] = frame["x"].astype(float) * float(x_multiplier)
    frame = frame.groupby("x", as_index=False)["y"].mean()
    frame = frame.sort_values("x").reset_index(drop=True)
    return frame


def aggregate_series(csv_paths: list[Path], shade_mode: str, x_multiplier: float) -> pd.DataFrame:
    series_list: list[pd.Series] = []
    for csv_path in csv_paths:
        frame = read_xy_from_csv(csv_path, x_multiplier)
        series = frame.set_index("x")["y"]
        series.name = csv_path.stem
        series_list.append(series)

    combined = pd.concat(series_list, axis=1, join="outer").sort_index()
    counts = combined.count(axis=1)
    mean = combined.mean(axis=1, skipna=True)
    std = combined.std(axis=1, ddof=1, skipna=True)
    sem = std / np.sqrt(counts.astype(float))
    sem = sem.where(counts >= 2, np.nan)
    std = std.where(counts >= 2, np.nan)

    if shade_mode == "sem":
        shade = sem
        shade_column = "sem"
    else:
        shade = std
        shade_column = "std"

    result = pd.DataFrame(
        {
            "x_value": combined.index.to_numpy(dtype=float),
            "mean": mean.to_numpy(dtype=float),
            "std": std.to_numpy(dtype=float),
            "sem": sem.to_numpy(dtype=float),
            "shade": shade.to_numpy(dtype=float),
            "n_files": counts.to_numpy(dtype=int),
            "shade_type": [shade_column] * len(combined.index),
        }
    )
    return result


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def render_plot(
    png_path: Path,
    aggregate_df: pd.DataFrame,
    x_label: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    x_values = aggregate_df["x_value"].to_numpy(dtype=float)
    mean_values = aggregate_df["mean"].to_numpy(dtype=float)
    shade_values = aggregate_df["shade"].to_numpy(dtype=float)
    finite_mask = np.isfinite(mean_values)
    x_values = x_values[finite_mask]
    mean_values = mean_values[finite_mask]
    shade_values = shade_values[finite_mask]

    line = ax.plot(x_values, mean_values, linewidth=LINEWIDTH)[0]
    shade_mask = np.isfinite(shade_values)
    if shade_mask.any():
        ax.fill_between(
            x_values[shade_mask],
            mean_values[shade_mask] - shade_values[shade_mask],
            mean_values[shade_mask] + shade_values[shade_mask],
            color=line.get_color(),
            alpha=SHADE_ALPHA,
            linewidth=0.0,
        )

    ax.set_xlabel(x_label, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_dir
    ensure_output_dir(output_dir)

    csv_paths = discover_csv_paths(input_dir, args.file_glob)
    aggregate_df = aggregate_series(csv_paths, args.shade, args.x_multiplier)

    output_stem = args.output_name
    png_path = output_dir / f"{output_stem}.png"
    csv_path = output_dir / f"{output_stem}.csv"

    render_plot(png_path, aggregate_df, args.x_label, args.y_label)
    aggregate_df.to_csv(csv_path, index=False)

    print(f"Saved plot: {png_path}")
    print(f"Saved aggregate CSV: {csv_path}")


if __name__ == "__main__":
    main()
