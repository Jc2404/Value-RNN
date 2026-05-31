from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS = ("MI", "softmax_linear_KL", "softmax_mlp_KL")
REQUIRED_COLUMNS = ("generator_episode", "evaluator_episode", "task_name", "task_value")

FIGSIZE = (6, 4)
LINEWIDTH = 2.0
FONT_SIZE = 14
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 10
LEGEND_FONT_SIZE = 10
BASE_MARKER = "o"
BASE_MARKER_SIZE = 70
BASE_LABEL = "Base environment"
BASE_MARKER_COLOR = "red"
BASE_REFERENCE_LINE_COLOR = "#808080"
TASK_VALUE_DECIMALS = 12

MI_LABEL = "Mutual Information"
KL_LABEL = "KL-Divergence"
EPISODE_LABEL = "Agent training episode"
MI_GAP_LABEL = "Mutual information gap"
EVOLUTION_COLORS = ("#f6ad55", "#e53e3e", "#63171b")


@dataclass(frozen=True)
class RunSummary:
    path: Path
    df: pd.DataFrame
    task_name: str
    shared_episodes: tuple[int, ...]
    first_nonzero_episode: int
    final_episode: int


@dataclass(frozen=True)
class PlotSpec:
    name: str
    x_label: str
    y_label: str
    x_kind: str
    series_order: tuple[tuple[str, str], ...]
    builder: Callable[[RunSummary], OrderedDict[str, pd.Series]] | None = None
    aggregate_fn: Callable[[list[RunSummary]], OrderedDict[str, pd.DataFrame]] | None = None
    base_marker_mode: str | None = None
    base_marker_series: str | None = None
    base_highlight_style: str = "marker"
    base_marker_color: str = BASE_MARKER_COLOR
    line_colors: tuple[str, ...] | None = None
    plot_style: str = "line"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate offline replay summary CSVs and reproduce notebook-style plots.",
    )
    parser.add_argument("--input-dir", required=True, type=str)
    parser.add_argument("--task-axis-label", required=True, type=str)
    parser.add_argument("--base-value", required=True, type=float)
    parser.add_argument("--output-dir", type=str, default=r"results\plots")
    parser.add_argument("--metrics", type=str, default=",".join(DEFAULT_METRICS))
    parser.add_argument(
        "--intermediate-episode",
        type=int,
        default=None,
        help=(
            "Optional checkpoint to use for 'Intermediate' in variant-evolution plots. "
            "If omitted, the script keeps the current default behavior."
        ),
    )
    parser.add_argument(
        "--file-glob",
        type=str,
        default="offline_replay_summary*.csv,counterfactual_replay_summary*.csv",
        help=(
            "Comma-separated glob(s) used to discover summary CSVs under --input-dir. "
            "Supports both legacy offline replay and counterfactual replay summaries."
        ),
    )
    parser.add_argument("--plot-sem", dest="plot_sem", action="store_true")
    parser.add_argument("--no-plot-sem", dest="plot_sem", action="store_false")
    parser.set_defaults(plot_sem=False)
    return parser.parse_args()


def parse_metrics_arg(raw_metrics: str) -> tuple[str, ...]:
    metrics = []
    for token in raw_metrics.split(","):
        metric = token.strip()
        if not metric:
            continue
        metrics.append(metric)

    if not metrics:
        raise ValueError("No metrics selected. Use a comma-separated subset of MI,softmax_linear_KL,softmax_mlp_KL.")

    unknown = sorted(set(metrics) - set(DEFAULT_METRICS))
    if unknown:
        raise ValueError(f"Unknown metrics requested: {unknown}")

    seen = set()
    ordered_metrics = []
    for metric in metrics:
        if metric not in seen:
            ordered_metrics.append(metric)
            seen.add(metric)

    has_linear = "softmax_linear_KL" in seen
    has_mlp = "softmax_mlp_KL" in seen
    if has_linear != has_mlp:
        raise ValueError(
            "Notebook-style KL analyses require both softmax_linear_KL and softmax_mlp_KL. "
            "Select both or neither."
        )

    return tuple(ordered_metrics)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_numeric_token(value: float) -> str:
    if np.isclose(value, round(value)):
        text = str(int(round(value)))
    else:
        text = np.format_float_positional(float(value), trim="-")
        text = text.rstrip("0").rstrip(".")
    text = text.replace("-", "m").replace(".", "p")
    return text


def base_token(base_value: float) -> str:
    return f"base_{format_numeric_token(base_value)}"


def read_numeric_column(df: pd.DataFrame, column: str, csv_path: Path) -> pd.Series:
    numeric = pd.to_numeric(df[column], errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{csv_path} has non-numeric values in '{column}'.")
    return numeric


def read_episode_column(df: pd.DataFrame, column: str, csv_path: Path) -> pd.Series:
    numeric = read_numeric_column(df, column, csv_path)
    values = numeric.to_numpy(dtype=float)
    if not np.all(np.isclose(values, np.round(values))):
        raise ValueError(f"{csv_path} has non-integer values in '{column}'.")
    return pd.Series(np.round(values).astype(int), index=df.index)


def read_task_value_column(df: pd.DataFrame, column: str, csv_path: Path) -> pd.Series:
    numeric = read_numeric_column(df, column, csv_path)
    return pd.Series(np.round(numeric.to_numpy(dtype=float), TASK_VALUE_DECIMALS), index=df.index)


def discover_csv_paths(input_dir: Path, file_glob: str) -> list[Path]:
    patterns = [token.strip() for token in str(file_glob).split(",") if token.strip()]
    csv_paths = []
    seen = set()
    for pattern in patterns:
        for path in sorted(input_dir.rglob(pattern)):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            csv_paths.append(path)
    if not csv_paths:
        raise FileNotFoundError(
            f"No files matching any of {patterns!r} were found under {input_dir}."
        )
    return csv_paths


def load_run_summaries(
    csv_paths: list[Path],
    selected_metrics: tuple[str, ...],
    base_value: float,
) -> tuple[list[RunSummary], str]:
    required_columns = set(REQUIRED_COLUMNS) | set(selected_metrics)
    run_summaries: list[RunSummary] = []
    folder_task_names: set[str] = set()

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        missing = sorted(required_columns - set(df.columns))
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        df = df.loc[:, list(REQUIRED_COLUMNS) + list(selected_metrics)].copy()
        df["generator_episode"] = read_episode_column(df, "generator_episode", csv_path)
        df["evaluator_episode"] = read_episode_column(df, "evaluator_episode", csv_path)
        df["task_value"] = read_task_value_column(df, "task_value", csv_path)

        task_names = {
            str(value).strip()
            for value in df["task_name"].dropna().tolist()
            if str(value).strip()
        }
        if len(task_names) != 1:
            raise ValueError(
                f"{csv_path} must contain exactly one non-null task_name, found {sorted(task_names)}."
            )
        task_name = next(iter(task_names))
        folder_task_names.add(task_name)

        generator_episodes = set(df["generator_episode"].tolist())
        evaluator_episodes = set(df["evaluator_episode"].tolist())
        if 0 not in generator_episodes:
            raise ValueError(f"{csv_path} is missing generator episode 0.")
        if 0 not in evaluator_episodes:
            raise ValueError(f"{csv_path} is missing evaluator episode 0.")

        shared_episodes = sorted(generator_episodes & evaluator_episodes)
        if not shared_episodes:
            raise ValueError(f"{csv_path} has no shared generator/evaluator episode values.")
        nonzero_shared_episodes = [episode for episode in shared_episodes if episode > 0]
        if not nonzero_shared_episodes:
            raise ValueError(f"{csv_path} has no non-zero shared generator/evaluator episode values.")
        first_nonzero_episode = int(nonzero_shared_episodes[0])
        final_episode = int(shared_episodes[-1])

        task_values = df["task_value"].to_numpy(dtype=float)
        if not np.isclose(task_values, base_value).any():
            raise ValueError(
                f"{csv_path} does not contain the requested base value {base_value} in task_value."
            )

        df = df.sort_values(["generator_episode", "evaluator_episode", "task_value"]).reset_index(drop=True)
        run_summaries.append(
            RunSummary(
                path=csv_path,
                df=df,
                task_name=task_name,
                shared_episodes=tuple(shared_episodes),
                first_nonzero_episode=first_nonzero_episode,
                final_episode=final_episode,
            )
        )

    if len(folder_task_names) != 1:
        raise ValueError(
            "Mixed task families detected in the input folder. "
            f"Found task_name values: {sorted(folder_task_names)}"
        )

    return run_summaries, next(iter(folder_task_names))


def task_value_mask(df: pd.DataFrame, task_value: float) -> pd.Series:
    return pd.Series(np.isclose(df["task_value"].to_numpy(dtype=float), task_value), index=df.index)


def extract_slice(
    df: pd.DataFrame,
    mask: pd.Series,
    x_col: str,
    metric_columns: tuple[str, ...],
    description: str,
    csv_path: Path,
) -> pd.DataFrame:
    subset = df.loc[mask, [x_col, *metric_columns]].copy()
    if subset.empty:
        raise ValueError(f"{csv_path} has no rows for analysis slice '{description}'.")
    subset = subset.sort_values(x_col).reset_index(drop=True)
    if subset[x_col].duplicated().any():
        raise ValueError(
            f"{csv_path} has duplicate '{x_col}' values for analysis slice '{description}'."
        )
    return subset


def search_generator_evaluator_across_task_values(
    run: RunSummary,
    metric_columns: tuple[str, ...],
    generator_episode: int,
    evaluator_episode: int,
) -> pd.DataFrame:
    mask = (
        (run.df["generator_episode"] == generator_episode)
        & (run.df["evaluator_episode"] == evaluator_episode)
    )
    description = (
        f"task sweep with generator_episode={generator_episode}, "
        f"evaluator_episode={evaluator_episode}"
    )
    return extract_slice(run.df, mask, "task_value", metric_columns, description, run.path)


def search_matched_generator_evaluator(
    run: RunSummary,
    metric_columns: tuple[str, ...],
    task_value: float,
) -> pd.DataFrame:
    mask = (
        (run.df["generator_episode"] == run.df["evaluator_episode"])
        & task_value_mask(run.df, task_value)
    )
    description = f"matched episode sweep at task_value={task_value}"
    return extract_slice(run.df, mask, "generator_episode", metric_columns, description, run.path)


def search_same_generator_different_evaluators(
    run: RunSummary,
    metric_columns: tuple[str, ...],
    generator_episode: int,
    task_value: float,
) -> pd.DataFrame:
    mask = (
        (run.df["generator_episode"] == generator_episode)
        & task_value_mask(run.df, task_value)
    )
    description = (
        f"fixed trajectory episode sweep with generator_episode={generator_episode}, "
        f"task_value={task_value}"
    )
    return extract_slice(run.df, mask, "evaluator_episode", metric_columns, description, run.path)


def search_same_evaluator_different_generators(
    run: RunSummary,
    metric_columns: tuple[str, ...],
    evaluator_episode: int,
    task_value: float,
) -> pd.DataFrame:
    mask = (
        (run.df["evaluator_episode"] == evaluator_episode)
        & task_value_mask(run.df, task_value)
    )
    description = (
        f"fixed agent episode sweep with evaluator_episode={evaluator_episode}, "
        f"task_value={task_value}"
    )
    return extract_slice(run.df, mask, "generator_episode", metric_columns, description, run.path)


def series_map_from_frame(
    frame: pd.DataFrame,
    x_col: str,
    series_order: tuple[tuple[str, str], ...],
) -> OrderedDict[str, pd.Series]:
    mapping: OrderedDict[str, pd.Series] = OrderedDict()
    x_values = frame[x_col].to_numpy()
    for series_label, metric_column in series_order:
        mapping[series_label] = pd.Series(
            frame[metric_column].to_numpy(dtype=float),
            index=x_values,
            name=series_label,
            dtype=float,
        )
    return mapping


def aggregate_series(series_list: list[pd.Series], x_kind: str) -> pd.DataFrame:
    frame = pd.concat(
        [series.rename(f"run_{idx}") for idx, series in enumerate(series_list)],
        axis=1,
        sort=True,
    ).sort_index()
    counts = frame.count(axis=1).astype(int)
    means = frame.mean(axis=1, skipna=True)
    sems = frame.std(axis=1, ddof=1, skipna=True) / np.sqrt(counts.astype(float))
    sems = sems.where(counts >= 2, np.nan)

    x_values = frame.index.to_numpy()
    if x_kind == "episode":
        x_values = np.round(x_values.astype(float)).astype(int)

    return pd.DataFrame(
        {
            "x_value": x_values,
            "mean": means.to_numpy(dtype=float),
            "sem": sems.to_numpy(dtype=float),
            "n_runs": counts.to_numpy(dtype=int),
        }
    )


def aggregate_plot_data(
    run_summaries: list[RunSummary],
    plot_spec: PlotSpec,
) -> OrderedDict[str, pd.DataFrame]:
    if plot_spec.aggregate_fn is not None:
        return plot_spec.aggregate_fn(run_summaries)

    if plot_spec.builder is None:
        raise ValueError(f"Plot specification '{plot_spec.name}' is missing both builder and aggregate_fn.")

    collected: OrderedDict[str, list[pd.Series]] = OrderedDict(
        (series_label, []) for series_label, _ in plot_spec.series_order
    )

    for run in run_summaries:
        run_series = plot_spec.builder(run)
        for series_label, _ in plot_spec.series_order:
            collected[series_label].append(run_series[series_label])

    aggregated: OrderedDict[str, pd.DataFrame] = OrderedDict()
    for series_label, _ in plot_spec.series_order:
        aggregated[series_label] = aggregate_series(collected[series_label], plot_spec.x_kind)

    return aggregated


def save_plot_csv(
    csv_path: Path,
    aggregated_series: OrderedDict[str, pd.DataFrame],
    plot_spec: PlotSpec,
) -> None:
    if plot_spec.plot_style == "raw_scatter":
        rows = []
        for series_label, _ in plot_spec.series_order:
            frame = aggregated_series[series_label]
            for row in frame.itertuples(index=False):
                rows.append(
                    {
                        "series_label": series_label,
                        "x_value": row.x_value,
                        "value": row.value,
                        "run_id": row.run_id,
                    }
                )
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return

    rows = []
    for series_label, _ in plot_spec.series_order:
        frame = aggregated_series[series_label]
        for row in frame.itertuples(index=False):
            rows.append(
                {
                    "series_label": series_label,
                    "x_value": row.x_value,
                    "mean": row.mean,
                    "sem": row.sem,
                    "n_runs": row.n_runs,
                }
            )

    pd.DataFrame(rows).to_csv(csv_path, index=False)


def add_base_markers(
    ax: plt.Axes,
    aggregated_series: OrderedDict[str, pd.DataFrame],
    plot_spec: PlotSpec,
    line_colors: dict[str, str],
    base_value: float,
) -> None:
    if plot_spec.base_marker_mode == "all":
        labels_to_mark = [series_label for series_label, _ in plot_spec.series_order]
    elif plot_spec.base_marker_mode == "series" and plot_spec.base_marker_series is not None:
        labels_to_mark = [plot_spec.base_marker_series]
    else:
        labels_to_mark = []

    base_points: list[tuple[float, float]] = []
    for series_label in labels_to_mark:
        frame = aggregated_series[series_label]
        x_values = frame["x_value"].to_numpy(dtype=float)
        mask = np.isclose(x_values, base_value)
        if not mask.any():
            raise ValueError(
                f"Aggregated plot '{plot_spec.name}' does not contain base value {base_value} "
                f"for series '{series_label}'."
            )
        base_x = frame.loc[mask, "x_value"].iloc[0]
        base_y = frame.loc[mask, "mean"].iloc[0]
        base_points.append((float(base_x), float(base_y)))

    if not base_points:
        return

    if plot_spec.base_highlight_style == "vline":
        base_x_values = [point[0] for point in base_points]
        if not np.allclose(base_x_values, base_x_values[0]):
            raise ValueError(
                f"Aggregated plot '{plot_spec.name}' has inconsistent base x values "
                f"across highlighted series: {base_x_values}"
            )
        ax.axvline(
            base_x_values[0],
            color=plot_spec.base_marker_color,
            linestyle="--",
            linewidth=1.6,
            zorder=4,
            label=BASE_LABEL,
        )
        return

    first_marker = True
    for base_x, base_y in base_points:
        ax.scatter(
            base_x,
            base_y,
            color=plot_spec.base_marker_color,
            s=BASE_MARKER_SIZE,
            marker=BASE_MARKER,
            zorder=5,
            label=BASE_LABEL if first_marker else None,
        )
        first_marker = False


def render_plot(
    png_path: Path,
    aggregated_series: OrderedDict[str, pd.DataFrame],
    plot_spec: PlotSpec,
    plot_sem: bool,
    base_value: float,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    show_line_legend = len(plot_spec.series_order) > 1
    line_colors: dict[str, str] = {}
    xticks: list[float] = []

    for index, (series_label, _) in enumerate(plot_spec.series_order):
        frame = aggregated_series[series_label]
        line_label = series_label if show_line_legend else None
        line_color = None
        if plot_spec.line_colors is not None:
            line_color = plot_spec.line_colors[index]

        if plot_spec.plot_style == "raw_scatter":
            x_values = frame["x_value"].to_numpy(dtype=float)
            scatter = ax.scatter(
                x_values,
                frame["value"].to_numpy(dtype=float),
                s=40,
                label=line_label,
                color=line_color,
                alpha=0.85,
            )
            plotted_color = scatter.get_facecolor()[0]
            plotted_color = tuple(plotted_color)
        elif plot_spec.plot_style == "scatter":
            x_values = frame["x_value"].to_numpy(dtype=float)
            mean_values = frame["mean"].to_numpy(dtype=float)
            sem_values = frame["sem"].to_numpy(dtype=float)
            if plot_sem:
                errorbar = ax.errorbar(
                    x_values,
                    mean_values,
                    yerr=sem_values,
                    fmt="o",
                    capsize=3,
                    markersize=6,
                    label=line_label,
                    color=line_color,
                )
                plotted_color = errorbar[0].get_color()
            else:
                scatter = ax.scatter(
                    x_values,
                    mean_values,
                    s=40,
                    label=line_label,
                    color=line_color,
                )
                plotted_color = scatter.get_facecolor()[0]
                plotted_color = tuple(plotted_color)
        else:
            x_values = frame["x_value"].to_numpy(dtype=float)
            mean_values = frame["mean"].to_numpy(dtype=float)
            sem_values = frame["sem"].to_numpy(dtype=float)
            (line,) = ax.plot(
                x_values,
                mean_values,
                linewidth=LINEWIDTH,
                label=line_label,
                color=line_color,
            )
            plotted_color = line.get_color()

            if plot_sem:
                ax.fill_between(
                    x_values,
                    mean_values - sem_values,
                    mean_values + sem_values,
                    color=plotted_color,
                    alpha=0.2,
                )

        line_colors[series_label] = plotted_color
        xticks.extend(x_values.tolist())

    if plot_spec.base_marker_mode is not None:
        add_base_markers(ax, aggregated_series, plot_spec, line_colors, base_value)

    ax.set_xlabel(plot_spec.x_label, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(plot_spec.y_label, fontsize=LABEL_FONT_SIZE)
    ax.set_xticks(sorted(set(xticks)))
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_line_legend or plot_spec.base_marker_mode is not None:
        ax.legend(fontsize=LEGEND_FONT_SIZE, frameon=False)

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def metric_columns(series_order: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(metric for _, metric in series_order))


def make_task_sweep_builder(
    series_order: tuple[tuple[str, str], ...],
    generator_episode_fn: Callable[[RunSummary], int],
    evaluator_episode_fn: Callable[[RunSummary], int],
) -> Callable[[RunSummary], OrderedDict[str, pd.Series]]:
    columns = metric_columns(series_order)

    def builder(run: RunSummary) -> OrderedDict[str, pd.Series]:
        frame = search_generator_evaluator_across_task_values(
            run,
            columns,
            generator_episode_fn(run),
            evaluator_episode_fn(run),
        )
        return series_map_from_frame(frame, "task_value", series_order)

    return builder


def make_matched_episode_builder(
    series_order: tuple[tuple[str, str], ...],
    base_value: float,
) -> Callable[[RunSummary], OrderedDict[str, pd.Series]]:
    columns = metric_columns(series_order)

    def builder(run: RunSummary) -> OrderedDict[str, pd.Series]:
        frame = search_matched_generator_evaluator(run, columns, base_value)
        return series_map_from_frame(frame, "generator_episode", series_order)

    return builder


def make_episode_decoupling_builder(
    series_order: tuple[tuple[str, str], ...],
    base_value: float,
) -> Callable[[RunSummary], OrderedDict[str, pd.Series]]:
    columns = metric_columns(series_order)

    def builder(run: RunSummary) -> OrderedDict[str, pd.Series]:
        matched = search_matched_generator_evaluator(run, columns, base_value)
        fixed_trajectory = search_same_generator_different_evaluators(
            run,
            columns,
            generator_episode=run.final_episode,
            task_value=base_value,
        )
        fixed_agent = search_same_evaluator_different_generators(
            run,
            columns,
            evaluator_episode=run.final_episode,
            task_value=base_value,
        )

        mapping: OrderedDict[str, pd.Series] = OrderedDict()
        mapping["Matched"] = pd.Series(
            matched[series_order[0][1]].to_numpy(dtype=float),
            index=matched["generator_episode"].to_numpy(dtype=int),
            name="Matched",
            dtype=float,
        )
        mapping["Fixed trajectory"] = pd.Series(
            fixed_trajectory[series_order[1][1]].to_numpy(dtype=float),
            index=fixed_trajectory["evaluator_episode"].to_numpy(dtype=int),
            name="Fixed trajectory",
            dtype=float,
        )
        mapping["Fixed agent"] = pd.Series(
            fixed_agent[series_order[2][1]].to_numpy(dtype=float),
            index=fixed_agent["generator_episode"].to_numpy(dtype=int),
            name="Fixed agent",
            dtype=float,
        )
        return mapping

    return builder


def make_task_decoupling_builder(
    series_order: tuple[tuple[str, str], ...],
) -> Callable[[RunSummary], OrderedDict[str, pd.Series]]:
    columns = metric_columns(series_order)

    def builder(run: RunSummary) -> OrderedDict[str, pd.Series]:
        matched = search_generator_evaluator_across_task_values(
            run,
            columns,
            generator_episode=run.final_episode,
            evaluator_episode=run.final_episode,
        )
        fixed_trajectory = search_generator_evaluator_across_task_values(
            run,
            columns,
            generator_episode=run.final_episode,
            evaluator_episode=0,
        )
        fixed_agent = search_generator_evaluator_across_task_values(
            run,
            columns,
            generator_episode=0,
            evaluator_episode=run.final_episode,
        )

        mapping: OrderedDict[str, pd.Series] = OrderedDict()
        mapping["Matched"] = pd.Series(
            matched[series_order[0][1]].to_numpy(dtype=float),
            index=matched["task_value"].to_numpy(dtype=float),
            name="Matched",
            dtype=float,
        )
        mapping["Fixed trajectory"] = pd.Series(
            fixed_trajectory[series_order[1][1]].to_numpy(dtype=float),
            index=fixed_trajectory["task_value"].to_numpy(dtype=float),
            name="Fixed trajectory",
            dtype=float,
        )
        mapping["Fixed agent"] = pd.Series(
            fixed_agent[series_order[2][1]].to_numpy(dtype=float),
            index=fixed_agent["task_value"].to_numpy(dtype=float),
            name="Fixed agent",
            dtype=float,
        )
        return mapping

    return builder


def make_variant_evolution_builder(
    metric_column: str,
    mode: str,
    intermediate_episode: int | None = None,
) -> Callable[[RunSummary], OrderedDict[str, pd.Series]]:
    def builder(run: RunSummary) -> OrderedDict[str, pd.Series]:
        chosen_intermediate = resolve_intermediate_episode(run, intermediate_episode)
        checkpoints = (
            ("Untrained", 0),
            ("Intermediate", chosen_intermediate),
            ("Trained", run.final_episode),
        )
        mapping: OrderedDict[str, pd.Series] = OrderedDict()

        for series_label, episode in checkpoints:
            if mode == "matched":
                frame = search_generator_evaluator_across_task_values(
                    run,
                    (metric_column,),
                    generator_episode=episode,
                    evaluator_episode=episode,
                )
            elif mode == "fixed_trajectory":
                frame = search_generator_evaluator_across_task_values(
                    run,
                    (metric_column,),
                    generator_episode=run.final_episode,
                    evaluator_episode=episode,
                )
            elif mode == "fixed_agent":
                frame = search_generator_evaluator_across_task_values(
                    run,
                    (metric_column,),
                    generator_episode=episode,
                    evaluator_episode=run.final_episode,
                )
            else:
                raise ValueError(f"Unknown variant evolution mode: {mode}")

            mapping[series_label] = pd.Series(
                frame[metric_column].to_numpy(dtype=float),
                index=frame["task_value"].to_numpy(dtype=float),
                name=series_label,
                dtype=float,
            )

        return mapping

    return builder


def resolve_intermediate_episode(run: RunSummary, intermediate_episode: int | None) -> int:
    if intermediate_episode is None:
        return run.first_nonzero_episode

    available = set(run.shared_episodes)
    if intermediate_episode not in available:
        raise ValueError(
            f"{run.path} does not contain shared generator/evaluator episode "
            f"{intermediate_episode}. Available shared episodes: {list(run.shared_episodes)}"
        )
    if intermediate_episode == 0:
        raise ValueError(
            f"{run.path} cannot use 0 as the intermediate episode. "
            "Choose a checkpoint strictly between untrained and trained."
        )
    if intermediate_episode == run.final_episode:
        raise ValueError(
            f"{run.path} cannot use the final shared episode {run.final_episode} as the "
            "intermediate episode."
        )
    return int(intermediate_episode)


def validate_unique_task_rows(
    frame: pd.DataFrame,
    x_col: str,
    description: str,
    csv_path: Path,
) -> None:
    duplicate_mask = frame.duplicated([x_col, "task_value"], keep=False)
    if duplicate_mask.any():
        raise ValueError(
            f"{csv_path} has duplicate ({x_col}, task_value) rows for '{description}'."
        )


def collect_mi_gap_frame(
    run: RunSummary,
    mode: str,
) -> tuple[pd.DataFrame, str, str]:
    metric_column = "MI"

    if mode == "matched":
        frame = run.df.loc[
            run.df["generator_episode"] == run.df["evaluator_episode"],
            ["generator_episode", "task_value", metric_column],
        ].copy()
        x_col = "generator_episode"
        description = "matched MI gap episode sweep"
    elif mode == "fixed_trajectory":
        frame = run.df.loc[
            run.df["generator_episode"] == run.final_episode,
            ["evaluator_episode", "task_value", metric_column],
        ].copy()
        x_col = "evaluator_episode"
        description = "fixed trajectory MI gap episode sweep"
    elif mode == "fixed_agent":
        frame = run.df.loc[
            run.df["evaluator_episode"] == run.final_episode,
            ["generator_episode", "task_value", metric_column],
        ].copy()
        x_col = "generator_episode"
        description = "fixed agent MI gap episode sweep"
    else:
        raise ValueError(f"Unknown MI gap mode: {mode}")

    validate_unique_task_rows(frame, x_col, description, run.path)
    return frame, x_col, description


def aggregate_mi_gap_from_mean_curves(
    run_summaries: list[RunSummary],
    mode: str,
    base_value: float,
) -> OrderedDict[str, pd.DataFrame]:
    frames = []
    x_col_name: str | None = None

    for run_index, run in enumerate(run_summaries):
        frame, x_col, _ = collect_mi_gap_frame(run, mode)
        if x_col_name is None:
            x_col_name = x_col
        elif x_col_name != x_col:
            raise ValueError(f"Inconsistent x column for MI gap mode '{mode}'.")

        tagged = frame.copy()
        tagged["run_id"] = run_index
        frames.append(tagged)

    if not frames or x_col_name is None:
        raise ValueError(f"No run data was available for MI gap mode '{mode}'.")

    combined = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, float | int]] = []

    for x_value, group in combined.groupby(x_col_name, sort=True):
        group = group.sort_values(["task_value", "run_id"]).reset_index(drop=True)
        base_mask = np.isclose(group["task_value"].to_numpy(dtype=float), base_value)
        base_per_run = group.loc[base_mask, [x_col_name, "run_id", "MI"]]
        if base_per_run.empty:
            raise ValueError(
                f"No base task value found for MI gap mode '{mode}' at {x_col_name}={x_value}."
            )

        if base_per_run["run_id"].duplicated().any():
            raise ValueError(
                f"Multiple base rows found for MI gap mode '{mode}' at {x_col_name}={x_value}."
            )

        non_base_group = group.loc[~base_mask, [x_col_name, "run_id", "task_value", "MI"]]
        if non_base_group.empty:
            raise ValueError(
                f"No non-base task values found for MI gap mode '{mode}' at {x_col_name}={x_value}."
            )

        mean_by_task = non_base_group.groupby("task_value", sort=True)["MI"].mean()
        best_nonbase_task = float(mean_by_task.idxmax())
        best_nonbase_rows = non_base_group.loc[
            np.isclose(non_base_group["task_value"].to_numpy(dtype=float), best_nonbase_task),
            ["run_id", "MI"],
        ].rename(columns={"MI": "nonbase_mi"})

        merged = base_per_run.rename(columns={"MI": "base_mi"}).merge(
            best_nonbase_rows,
            on="run_id",
            how="inner",
        )
        if merged.empty:
            raise ValueError(
                f"No overlapping runs for base and best non-base task in MI gap mode '{mode}' "
                f"at {x_col_name}={x_value}."
            )

        diff = merged["base_mi"] - merged["nonbase_mi"]
        n_runs = int(diff.count())
        sem = float(diff.std(ddof=1) / np.sqrt(n_runs)) if n_runs >= 2 else np.nan
        rows.append(
            {
                "x_value": int(round(float(x_value))) if x_col_name.endswith("episode") else float(x_value),
                "mean": float(diff.mean()),
                "sem": sem,
                "n_runs": n_runs,
            }
        )

    return OrderedDict((("MI gap", pd.DataFrame(rows)),))


def collect_mi_gap_raw_points(
    run_summaries: list[RunSummary],
    mode: str,
    base_value: float,
) -> OrderedDict[str, pd.DataFrame]:
    rows: list[dict[str, float | int]] = []

    for run_index, run in enumerate(run_summaries):
        frame, x_col, description = collect_mi_gap_frame(run, mode)

        for x_value, group in frame.groupby(x_col, sort=True):
            group = group.sort_values("task_value").reset_index(drop=True)
            base_mask = np.isclose(group["task_value"].to_numpy(dtype=float), base_value)
            if base_mask.sum() != 1:
                raise ValueError(
                    f"{run.path} expected exactly one base task value for '{description}' at "
                    f"{x_col}={x_value}, found {int(base_mask.sum())}."
                )

            non_base = group.loc[~base_mask, "MI"]
            if non_base.empty:
                raise ValueError(
                    f"{run.path} needs at least one non-base task value for '{description}' at "
                    f"{x_col}={x_value}."
                )

            base_mi = float(group.loc[base_mask, "MI"].iloc[0])
            value = base_mi - float(non_base.max())
            point_x = int(round(float(x_value))) if x_col.endswith("episode") else float(x_value)
            rows.append(
                {
                    "x_value": point_x,
                    "value": value,
                    "run_id": run_index,
                }
            )

    return OrderedDict((("MI gap", pd.DataFrame(rows)),))


def make_mi_gap_aggregate_fn(
    mode: str,
    base_value: float,
) -> Callable[[list[RunSummary]], OrderedDict[str, pd.DataFrame]]:
    def aggregate_fn(run_summaries: list[RunSummary]) -> OrderedDict[str, pd.DataFrame]:
        return aggregate_mi_gap_from_mean_curves(run_summaries, mode, base_value)

    return aggregate_fn


def make_mi_gap_raw_scatter_fn(
    mode: str,
    base_value: float,
) -> Callable[[list[RunSummary]], OrderedDict[str, pd.DataFrame]]:
    def aggregate_fn(run_summaries: list[RunSummary]) -> OrderedDict[str, pd.DataFrame]:
        return collect_mi_gap_raw_points(run_summaries, mode, base_value)

    return aggregate_fn


def build_plot_specs(
    selected_metrics: tuple[str, ...],
    task_axis_label: str,
    base_value: float,
    intermediate_episode: int | None = None,
) -> list[PlotSpec]:
    specs: list[PlotSpec] = []
    base_name = base_token(base_value)

    if "MI" in selected_metrics:
        specs.extend(
            [
                PlotSpec(
                    name="mi__task_sweep__matched_ep0",
                    x_label=task_axis_label,
                    y_label=MI_LABEL,
                    x_kind="task",
                    series_order=(("Matched", "MI"),),
                    builder=make_task_sweep_builder(
                        (("Matched", "MI"),),
                        generator_episode_fn=lambda run: 0,
                        evaluator_episode_fn=lambda run: 0,
                    ),
                    base_marker_mode="all",
                ),
                PlotSpec(
                    name="mi__task_sweep__matched_epfinal",
                    x_label=task_axis_label,
                    y_label=MI_LABEL,
                    x_kind="task",
                    series_order=(("Matched", "MI"),),
                    builder=make_task_sweep_builder(
                        (("Matched", "MI"),),
                        generator_episode_fn=lambda run: run.final_episode,
                        evaluator_episode_fn=lambda run: run.final_episode,
                    ),
                    base_marker_mode="all",
                ),
                PlotSpec(
                    name=f"mi__episode_sweep__matched__{base_name}",
                    x_label=EPISODE_LABEL,
                    y_label=MI_LABEL,
                    x_kind="episode",
                    series_order=(("Matched", "MI"),),
                    builder=make_matched_episode_builder((("Matched", "MI"),), base_value),
                ),
                PlotSpec(
                    name=f"mi__episode_decoupling__{base_name}",
                    x_label=EPISODE_LABEL,
                    y_label=MI_LABEL,
                    x_kind="episode",
                    series_order=(
                        ("Matched", "MI"),
                        ("Fixed trajectory", "MI"),
                        ("Fixed agent", "MI"),
                    ),
                    builder=make_episode_decoupling_builder(
                        (
                            ("Matched", "MI"),
                            ("Fixed trajectory", "MI"),
                            ("Fixed agent", "MI"),
                        ),
                        base_value,
                    ),
                ),
                PlotSpec(
                    name="mi__task_decoupling__matched_epfinal",
                    x_label=task_axis_label,
                    y_label=MI_LABEL,
                    x_kind="task",
                    series_order=(
                        ("Matched", "MI"),
                        ("Fixed trajectory", "MI"),
                        ("Fixed agent", "MI"),
                    ),
                    builder=make_task_decoupling_builder(
                        (
                            ("Matched", "MI"),
                            ("Fixed trajectory", "MI"),
                            ("Fixed agent", "MI"),
                        )
                    ),
                    base_marker_mode="all",
                ),
                PlotSpec(
                    name="mi__variant_evolution__matched",
                    x_label=task_axis_label,
                    y_label=MI_LABEL,
                    x_kind="task",
                    series_order=(
                        ("Untrained", "MI"),
                        ("Intermediate", "MI"),
                        ("Trained", "MI"),
                    ),
                    builder=make_variant_evolution_builder("MI", "matched", intermediate_episode),
                    base_marker_mode="all",
                    base_highlight_style="vline",
                    base_marker_color=BASE_REFERENCE_LINE_COLOR,
                    line_colors=EVOLUTION_COLORS,
                ),
                PlotSpec(
                    name="mi__variant_evolution__fixed_trajectory",
                    x_label=task_axis_label,
                    y_label=MI_LABEL,
                    x_kind="task",
                    series_order=(
                        ("Untrained", "MI"),
                        ("Intermediate", "MI"),
                        ("Trained", "MI"),
                    ),
                    builder=make_variant_evolution_builder("MI", "fixed_trajectory", intermediate_episode),
                    base_marker_mode="all",
                    base_highlight_style="vline",
                    base_marker_color=BASE_REFERENCE_LINE_COLOR,
                    line_colors=EVOLUTION_COLORS,
                ),
                PlotSpec(
                    name="mi__variant_evolution__fixed_agent",
                    x_label=task_axis_label,
                    y_label=MI_LABEL,
                    x_kind="task",
                    series_order=(
                        ("Untrained", "MI"),
                        ("Intermediate", "MI"),
                        ("Trained", "MI"),
                    ),
                    builder=make_variant_evolution_builder("MI", "fixed_agent", intermediate_episode),
                    base_marker_mode="all",
                    base_highlight_style="vline",
                    base_marker_color=BASE_REFERENCE_LINE_COLOR,
                    line_colors=EVOLUTION_COLORS,
                ),
                PlotSpec(
                    name=f"mi__base_minus_best_nonbase__matched__{base_name}",
                    x_label=EPISODE_LABEL,
                    y_label=MI_GAP_LABEL,
                    x_kind="episode",
                    series_order=(("MI gap", "MI"),),
                    aggregate_fn=make_mi_gap_aggregate_fn("matched", base_value),
                ),
                PlotSpec(
                    name=f"mi__base_minus_best_nonbase__matched__{base_name}__scatter",
                    x_label=EPISODE_LABEL,
                    y_label=MI_GAP_LABEL,
                    x_kind="episode",
                    series_order=(("MI gap", "MI"),),
                    aggregate_fn=make_mi_gap_raw_scatter_fn("matched", base_value),
                    plot_style="raw_scatter",
                ),
                PlotSpec(
                    name=f"mi__base_minus_best_nonbase__fixed_trajectory__{base_name}",
                    x_label=EPISODE_LABEL,
                    y_label=MI_GAP_LABEL,
                    x_kind="episode",
                    series_order=(("MI gap", "MI"),),
                    aggregate_fn=make_mi_gap_aggregate_fn("fixed_trajectory", base_value),
                ),
                PlotSpec(
                    name=f"mi__base_minus_best_nonbase__fixed_trajectory__{base_name}__scatter",
                    x_label=EPISODE_LABEL,
                    y_label=MI_GAP_LABEL,
                    x_kind="episode",
                    series_order=(("MI gap", "MI"),),
                    aggregate_fn=make_mi_gap_raw_scatter_fn("fixed_trajectory", base_value),
                    plot_style="raw_scatter",
                ),
                PlotSpec(
                    name=f"mi__base_minus_best_nonbase__fixed_agent__{base_name}",
                    x_label=EPISODE_LABEL,
                    y_label=MI_GAP_LABEL,
                    x_kind="episode",
                    series_order=(("MI gap", "MI"),),
                    aggregate_fn=make_mi_gap_aggregate_fn("fixed_agent", base_value),
                ),
                PlotSpec(
                    name=f"mi__base_minus_best_nonbase__fixed_agent__{base_name}__scatter",
                    x_label=EPISODE_LABEL,
                    y_label=MI_GAP_LABEL,
                    x_kind="episode",
                    series_order=(("MI gap", "MI"),),
                    aggregate_fn=make_mi_gap_raw_scatter_fn("fixed_agent", base_value),
                    plot_style="raw_scatter",
                ),
            ]
        )

    has_kl_pair = "softmax_linear_KL" in selected_metrics and "softmax_mlp_KL" in selected_metrics
    if has_kl_pair:
        decoder_series_order = (
            ("Linear decoder", "softmax_linear_KL"),
            ("Non-linear decoder", "softmax_mlp_KL"),
        )
        specs.extend(
            [
                PlotSpec(
                    name="kl_decoders__task_sweep__matched_ep0",
                    x_label=task_axis_label,
                    y_label=KL_LABEL,
                    x_kind="task",
                    series_order=decoder_series_order,
                    builder=make_task_sweep_builder(
                        decoder_series_order,
                        generator_episode_fn=lambda run: 0,
                        evaluator_episode_fn=lambda run: 0,
                    ),
                    base_marker_mode="all",
                ),
                PlotSpec(
                    name="kl_decoders__task_sweep__matched_epfinal",
                    x_label=task_axis_label,
                    y_label=KL_LABEL,
                    x_kind="task",
                    series_order=decoder_series_order,
                    builder=make_task_sweep_builder(
                        decoder_series_order,
                        generator_episode_fn=lambda run: run.final_episode,
                        evaluator_episode_fn=lambda run: run.final_episode,
                    ),
                    base_marker_mode="all",
                ),
                PlotSpec(
                    name="kl_linear__variant_evolution__matched",
                    x_label=task_axis_label,
                    y_label=KL_LABEL,
                    x_kind="task",
                    series_order=(
                        ("Untrained", "softmax_linear_KL"),
                        ("Intermediate", "softmax_linear_KL"),
                        ("Trained", "softmax_linear_KL"),
                    ),
                    builder=make_variant_evolution_builder(
                        "softmax_linear_KL",
                        "matched",
                        intermediate_episode,
                    ),
                    base_marker_mode="all",
                    base_highlight_style="vline",
                    base_marker_color=BASE_REFERENCE_LINE_COLOR,
                    line_colors=EVOLUTION_COLORS,
                ),
                PlotSpec(
                    name="kl_linear__variant_evolution__fixed_trajectory",
                    x_label=task_axis_label,
                    y_label=KL_LABEL,
                    x_kind="task",
                    series_order=(
                        ("Untrained", "softmax_linear_KL"),
                        ("Intermediate", "softmax_linear_KL"),
                        ("Trained", "softmax_linear_KL"),
                    ),
                    builder=make_variant_evolution_builder(
                        "softmax_linear_KL",
                        "fixed_trajectory",
                        intermediate_episode,
                    ),
                    base_marker_mode="all",
                    base_highlight_style="vline",
                    base_marker_color=BASE_REFERENCE_LINE_COLOR,
                    line_colors=EVOLUTION_COLORS,
                ),
                PlotSpec(
                    name="kl_linear__variant_evolution__fixed_agent",
                    x_label=task_axis_label,
                    y_label=KL_LABEL,
                    x_kind="task",
                    series_order=(
                        ("Untrained", "softmax_linear_KL"),
                        ("Intermediate", "softmax_linear_KL"),
                        ("Trained", "softmax_linear_KL"),
                    ),
                    builder=make_variant_evolution_builder(
                        "softmax_linear_KL",
                        "fixed_agent",
                        intermediate_episode,
                    ),
                    base_marker_mode="all",
                    base_highlight_style="vline",
                    base_marker_color=BASE_REFERENCE_LINE_COLOR,
                    line_colors=EVOLUTION_COLORS,
                ),
                PlotSpec(
                    name="kl_nonlinear__variant_evolution__matched",
                    x_label=task_axis_label,
                    y_label=KL_LABEL,
                    x_kind="task",
                    series_order=(
                        ("Untrained", "softmax_mlp_KL"),
                        ("Intermediate", "softmax_mlp_KL"),
                        ("Trained", "softmax_mlp_KL"),
                    ),
                    builder=make_variant_evolution_builder(
                        "softmax_mlp_KL",
                        "matched",
                        intermediate_episode,
                    ),
                    base_marker_mode="all",
                    base_highlight_style="vline",
                    base_marker_color=BASE_REFERENCE_LINE_COLOR,
                    line_colors=EVOLUTION_COLORS,
                ),
                PlotSpec(
                    name="kl_nonlinear__variant_evolution__fixed_trajectory",
                    x_label=task_axis_label,
                    y_label=KL_LABEL,
                    x_kind="task",
                    series_order=(
                        ("Untrained", "softmax_mlp_KL"),
                        ("Intermediate", "softmax_mlp_KL"),
                        ("Trained", "softmax_mlp_KL"),
                    ),
                    builder=make_variant_evolution_builder(
                        "softmax_mlp_KL",
                        "fixed_trajectory",
                        intermediate_episode,
                    ),
                    base_marker_mode="all",
                    base_highlight_style="vline",
                    base_marker_color=BASE_REFERENCE_LINE_COLOR,
                    line_colors=EVOLUTION_COLORS,
                ),
                PlotSpec(
                    name="kl_nonlinear__variant_evolution__fixed_agent",
                    x_label=task_axis_label,
                    y_label=KL_LABEL,
                    x_kind="task",
                    series_order=(
                        ("Untrained", "softmax_mlp_KL"),
                        ("Intermediate", "softmax_mlp_KL"),
                        ("Trained", "softmax_mlp_KL"),
                    ),
                    builder=make_variant_evolution_builder(
                        "softmax_mlp_KL",
                        "fixed_agent",
                        intermediate_episode,
                    ),
                    base_marker_mode="all",
                    base_highlight_style="vline",
                    base_marker_color=BASE_REFERENCE_LINE_COLOR,
                    line_colors=EVOLUTION_COLORS,
                ),
                PlotSpec(
                    name=f"kl_decoders__episode_sweep__{base_name}",
                    x_label=EPISODE_LABEL,
                    y_label=KL_LABEL,
                    x_kind="episode",
                    series_order=decoder_series_order,
                    builder=make_matched_episode_builder(decoder_series_order, base_value),
                ),
                PlotSpec(
                    name=f"kl_linear__episode_decoupling__{base_name}",
                    x_label=EPISODE_LABEL,
                    y_label=KL_LABEL,
                    x_kind="episode",
                    series_order=(
                        ("Matched", "softmax_linear_KL"),
                        ("Fixed trajectory", "softmax_linear_KL"),
                        ("Fixed agent", "softmax_linear_KL"),
                    ),
                    builder=make_episode_decoupling_builder(
                        (
                            ("Matched", "softmax_linear_KL"),
                            ("Fixed trajectory", "softmax_linear_KL"),
                            ("Fixed agent", "softmax_linear_KL"),
                        ),
                        base_value,
                    ),
                ),
                PlotSpec(
                    name=f"kl_nonlinear__episode_decoupling__{base_name}",
                    x_label=EPISODE_LABEL,
                    y_label=KL_LABEL,
                    x_kind="episode",
                    series_order=(
                        ("Matched", "softmax_mlp_KL"),
                        ("Fixed trajectory", "softmax_mlp_KL"),
                        ("Fixed agent", "softmax_mlp_KL"),
                    ),
                    builder=make_episode_decoupling_builder(
                        (
                            ("Matched", "softmax_mlp_KL"),
                            ("Fixed trajectory", "softmax_mlp_KL"),
                            ("Fixed agent", "softmax_mlp_KL"),
                        ),
                        base_value,
                    ),
                ),
                PlotSpec(
                    name="kl_linear__task_decoupling__matched_epfinal",
                    x_label=task_axis_label,
                    y_label=KL_LABEL,
                    x_kind="task",
                    series_order=(
                        ("Matched", "softmax_linear_KL"),
                        ("Fixed trajectory", "softmax_linear_KL"),
                        ("Fixed agent", "softmax_linear_KL"),
                    ),
                    builder=make_task_decoupling_builder(
                        (
                            ("Matched", "softmax_linear_KL"),
                            ("Fixed trajectory", "softmax_linear_KL"),
                            ("Fixed agent", "softmax_linear_KL"),
                        )
                    ),
                    base_marker_mode="all",
                ),
                PlotSpec(
                    name="kl_nonlinear__task_decoupling__matched_epfinal",
                    x_label=task_axis_label,
                    y_label=KL_LABEL,
                    x_kind="task",
                    series_order=(
                        ("Matched", "softmax_mlp_KL"),
                        ("Fixed trajectory", "softmax_mlp_KL"),
                        ("Fixed agent", "softmax_mlp_KL"),
                    ),
                    builder=make_task_decoupling_builder(
                        (
                            ("Matched", "softmax_mlp_KL"),
                            ("Fixed trajectory", "softmax_mlp_KL"),
                            ("Fixed agent", "softmax_mlp_KL"),
                        )
                    ),
                    base_marker_mode="all",
                ),
            ]
        )

    return specs


def run_plot_generation(
    run_summaries: list[RunSummary],
    plot_specs: list[PlotSpec],
    output_dir: Path,
    plot_sem: bool,
    base_value: float,
) -> None:
    ensure_output_dir(output_dir)

    for plot_spec in plot_specs:
        aggregated_series = aggregate_plot_data(run_summaries, plot_spec)
        png_path = output_dir / f"{plot_spec.name}.png"
        csv_path = output_dir / f"{plot_spec.name}.csv"
        save_plot_csv(csv_path, aggregated_series, plot_spec)
        render_plot(png_path, aggregated_series, plot_spec, plot_sem, base_value)
        print(f"Saved plot: {png_path}")
        print(f"Saved data: {csv_path}")


def main() -> None:
    args = parse_args()
    selected_metrics = parse_metrics_arg(args.metrics)
    input_dir = Path(args.input_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else input_dir / "aggregated_plots"
    )

    csv_paths = discover_csv_paths(input_dir, args.file_glob)
    run_summaries, task_name = load_run_summaries(csv_paths, selected_metrics, args.base_value)
    print(f"Loaded {len(run_summaries)} run summaries for task '{task_name}'.")

    plot_specs = build_plot_specs(
        selected_metrics,
        args.task_axis_label,
        args.base_value,
        intermediate_episode=args.intermediate_episode,
    )
    if not plot_specs:
        raise ValueError("No plot specifications were created for the requested metric selection.")

    run_plot_generation(run_summaries, plot_specs, output_dir, args.plot_sem, args.base_value)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
