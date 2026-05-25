import json
import re
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SETUP_ORDER = ["protocol_b_original", "compound_online", "offline_same"]
SETUP_LABELS = {
    "protocol_b_original": "Protocol B",
    "compound_online": "Compound online",
    "offline_same": "Offline same-agent",
    "offline_cross": "Offline cross-agent",
}
SETUP_COLORS = {
    "protocol_b_original": "#222222",
    "compound_online": "#D55E00",
    "offline_same": "#0072B2",
    "offline_cross": "#009E73",
}
SETUP_MARKERS = {
    "protocol_b_original": "o",
    "compound_online": "s",
    "offline_same": "^",
    "offline_cross": "D",
}
METRIC_LABELS = {
    "linreg_rsq-0": "Belief $R^2$",
    "normalized_MI": "Normalized MI",
    "MI": "MI",
    "comparison_rollout_mean_drqn_disc_return": "Discounted return",
}
TASK_LABELS = {
    "grid_size": "Grid size",
    "tprob": "Transition prob.",
    "reward_scheme": "Reward scheme",
    "reward_margin": "Reward margin",
    "listen_accuracy": "Listen accuracy",
    "reward_listen": "Listen reward",
    "tmaze_length": "Maze length",
    "tmaze_stochasticity": "Stochasticity",
    "starkweather_p_omission": "Omission prob.",
    "starkweather_bin_size": "Bin size",
    "starkweather_iti_hazard": "ITI hazard",
    "starkweather_iti_min": "ITI minimum",
    "starkweather_nITI_microstates": "ITI microstates",
    "crybaby_p_cry_if_hungry": "Cry hungry prob.",
    "crybaby_p_cry_if_full": "Cry full prob.",
    "base": "Matched",
}
HIGHER_IS_BETTER = {
    "linreg_rsq-0",
    "normalized_MI",
    "MI",
    "comparison_rollout_mean_drqn_disc_return",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "value"


def set_paper_style() -> None:
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "lines.linewidth": 2.4,
        "lines.markersize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def short_metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " "))


def short_task_label(task_name: str) -> str:
    return TASK_LABELS.get(task_name, task_name.replace("_", " ").title())


def stage_labels_for_count(count: int) -> list[str]:
    if count >= 3:
        return ["Untrained", "Mid", "Trained"][:count]
    if count == 2:
        return ["Early", "Late"]
    return ["Checkpoint"]


def coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip()
    text = text.replace({"": np.nan, "None": np.nan, "nan": np.nan})
    text = text.str.replace(
        r"^tensor\(([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\)$",
        r"\1",
        regex=True,
    )
    return pd.to_numeric(text, errors="coerce")


def add_normalized_mi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "normalized_MI" in df.columns:
        df["normalized_MI"] = coerce_numeric(df["normalized_MI"])
        return df
    if "MI" not in df.columns:
        return df

    entropy_candidates = [
        "softmax_mlp_H_true",
        "softmax_mlp_H_true-0",
        "softmax_linear_H_true",
        "softmax_linear_H_true-0",
        "H_true_b0",
    ]
    entropy_col = next((col for col in entropy_candidates if col in df.columns), None)
    if entropy_col is None:
        return df

    mi = coerce_numeric(df["MI"])
    entropy = coerce_numeric(df[entropy_col])
    valid = entropy.abs() > 1e-8
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out.loc[valid] = mi.loc[valid] / entropy.loc[valid]
    df["normalized_MI"] = out
    return df


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return add_normalized_mi(df)


def parse_episode_from_sheet(sheet_name: str) -> int | None:
    match = re.search(r"ep_(\d+)", str(sheet_name))
    if not match:
        return None
    return int(match.group(1))


def load_protocol_b_workbook(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    frames = []
    for sheet_name in xls.sheet_names:
        episode = parse_episode_from_sheet(sheet_name)
        if episode is None:
            continue
        frame = pd.read_excel(path, sheet_name=sheet_name)
        if frame.empty:
            continue
        frame["evaluator_episode"] = int(episode)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return add_normalized_mi(pd.concat(frames, ignore_index=True))


def find_summary(run_root: Path, relative_glob: str) -> Path | None:
    matches = sorted(run_root.glob(relative_glob))
    if not matches:
        return None
    return matches[0]


def find_protocol_b_files(report_root: Path, train_id: str) -> tuple[list[Path], list[Path]]:
    workbook_paths = []
    for path in report_root.rglob(f"protocolB_*_{train_id}.xlsx"):
        if "_avg" in path.stem:
            continue
        workbook_paths.append(path)
    belief_paths = list(report_root.rglob(f"protocolB_*_{train_id}_belief_eval_summary_table.csv"))
    workbook_paths.sort()
    belief_paths.sort()
    return workbook_paths, belief_paths


def choose_stage_map(episodes: list[int]) -> dict[int, str]:
    unique = sorted({int(ep) for ep in episodes})
    if not unique:
        return {}
    if len(unique) <= 3:
        selected = unique
    else:
        selected = [unique[0], unique[len(unique) // 2], unique[-1]]
        selected = sorted(dict.fromkeys(selected))
    labels = stage_labels_for_count(len(selected))
    return {ep: labels[idx] for idx, ep in enumerate(selected)}


def prepare_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = coerce_numeric(df[col])
    if "task_value" in df.columns:
        numeric_task = coerce_numeric(df["task_value"])
        df["task_value_num"] = numeric_task
        df["task_value_str"] = df["task_value"].astype(str).replace({"nan": "", "None": ""})
    return df


def load_run_data(run_root: Path, report_root: Path, protocol_b_xlsx: list[str] | None,
                  protocol_b_belief_csv: list[str] | None) -> dict:
    offline_path = find_summary(run_root, "artefacts/offline_replay/offline_replay_summary_*.csv")
    compound_path = find_summary(run_root, "artefacts/compound_online/compound_online_summary_*.csv")
    cache_manifest = run_root / "cache_manifest.json"

    offline_df = load_csv(offline_path) if offline_path else pd.DataFrame()
    compound_df = load_csv(compound_path) if compound_path else pd.DataFrame()

    train_id = None
    for df in [offline_df, compound_df]:
        if not df.empty and "train_id" in df.columns:
            train_id = str(df["train_id"].dropna().iloc[0])
            break
    if train_id is None and cache_manifest.exists():
        manifest = json.loads(cache_manifest.read_text(encoding="utf-8"))
        if manifest:
            train_id = str(manifest[0]["train_id"])
    if train_id is None:
        raise FileNotFoundError("Could not infer train_id from the offline run root.")

    if protocol_b_xlsx:
        workbook_paths = [Path(path).resolve() for path in protocol_b_xlsx]
    else:
        workbook_paths, auto_belief = find_protocol_b_files(report_root, train_id)
        if protocol_b_belief_csv:
            belief_paths = [Path(path).resolve() for path in protocol_b_belief_csv]
        else:
            belief_paths = auto_belief
    if protocol_b_xlsx and protocol_b_belief_csv:
        belief_paths = [Path(path).resolve() for path in protocol_b_belief_csv]
    elif protocol_b_xlsx and not protocol_b_belief_csv:
        _, belief_paths = find_protocol_b_files(report_root, train_id)

    protocol_b_repr_frames = [load_protocol_b_workbook(path) for path in workbook_paths]
    protocol_b_repr = pd.concat([df for df in protocol_b_repr_frames if not df.empty], ignore_index=True) if protocol_b_repr_frames else pd.DataFrame()

    belief_frames = [load_csv(path) for path in belief_paths]
    protocol_b_control = pd.concat([df for df in belief_frames if not df.empty], ignore_index=True) if belief_frames else pd.DataFrame()

    return {
        "train_id": train_id,
        "offline_path": offline_path,
        "compound_path": compound_path,
        "protocol_b_workbooks": workbook_paths,
        "protocol_b_belief_csvs": belief_paths,
        "offline": offline_df,
        "compound": compound_df,
        "protocol_b_repr": protocol_b_repr,
        "protocol_b_control": protocol_b_control,
    }


def filter_to_families(df: pd.DataFrame, families: set[str]) -> pd.DataFrame:
    if df.empty or "task_name" not in df.columns:
        return df.copy()
    keep = set(families) | {"base"}
    return df[df["task_name"].isin(keep)].copy()


def normalize_sources(data: dict) -> dict:
    offline_df = data["offline"].copy()
    compound_df = data["compound"].copy()
    protocol_b_repr = data["protocol_b_repr"].copy()
    protocol_b_control = data["protocol_b_control"].copy()

    if not offline_df.empty and "setup" not in offline_df.columns:
        offline_df["setup"] = "offline_replay"
    if not compound_df.empty and "setup" not in compound_df.columns:
        compound_df["setup"] = "compound_online"
    if not protocol_b_repr.empty:
        protocol_b_repr["setup"] = "protocol_b_original"
    if not protocol_b_control.empty:
        protocol_b_control["setup"] = "protocol_b_original"

    offline_same = pd.DataFrame()
    offline_cross = pd.DataFrame()
    if not offline_df.empty:
        offline_same = offline_df[coerce_numeric(offline_df["generator_episode"]) == coerce_numeric(offline_df["evaluator_episode"])].copy()
        offline_same["setup"] = "offline_same"
        offline_cross = offline_df.copy()
        offline_cross["setup"] = "offline_cross"

    preferred_family_sources = [offline_same, compound_df]
    families = set()
    for df in preferred_family_sources:
        if df.empty or "task_name" not in df.columns:
            continue
        families.update(name for name in df["task_name"].dropna().unique().tolist() if str(name) != "base")
    if not families:
        for df in [protocol_b_repr, protocol_b_control]:
            if df.empty or "task_name" not in df.columns:
                continue
            families.update(name for name in df["task_name"].dropna().unique().tolist() if str(name) != "base")
    if not families:
        raise ValueError("Could not find any non-base sweep families in the discovered results.")

    offline_same = filter_to_families(offline_same, families)
    offline_cross = filter_to_families(offline_cross, families)
    compound_df = filter_to_families(compound_df, families)
    protocol_b_repr = filter_to_families(protocol_b_repr, families)
    protocol_b_control = filter_to_families(protocol_b_control, families)

    prepare_cols = [
        "generator_episode",
        "evaluator_episode",
        "agent_episode",
        "task_value",
        "linreg_rsq-0",
        "normalized_MI",
        "MI",
        "comparison_rollout_mean_drqn_disc_return",
    ]
    offline_same = prepare_numeric_columns(offline_same, prepare_cols)
    offline_cross = prepare_numeric_columns(offline_cross, prepare_cols)
    compound_df = prepare_numeric_columns(compound_df, prepare_cols)
    protocol_b_repr = prepare_numeric_columns(protocol_b_repr, prepare_cols)
    protocol_b_control = prepare_numeric_columns(protocol_b_control, prepare_cols)

    all_episode_candidates = []
    for df, col in [
        (offline_same, "evaluator_episode"),
        (compound_df, "evaluator_episode"),
        (protocol_b_repr, "evaluator_episode"),
        (protocol_b_control, "agent_episode"),
    ]:
        if not df.empty and col in df.columns:
            all_episode_candidates.extend(coerce_numeric(df[col]).dropna().astype(int).tolist())
    stage_map = choose_stage_map(all_episode_candidates)
    if not stage_map:
        raise ValueError("Could not infer checkpoint stages from the available results.")

    selected = set(stage_map.keys())
    if not offline_same.empty:
        offline_same = offline_same[offline_same["evaluator_episode"].astype(int).isin(selected)].copy()
    if not offline_cross.empty:
        offline_cross = offline_cross[
            offline_cross["evaluator_episode"].astype(int).isin(selected)
            & offline_cross["generator_episode"].astype(int).isin(selected)
        ].copy()
    if not compound_df.empty:
        compound_df = compound_df[compound_df["evaluator_episode"].astype(int).isin(selected)].copy()
    if not protocol_b_repr.empty:
        protocol_b_repr = protocol_b_repr[protocol_b_repr["evaluator_episode"].astype(int).isin(selected)].copy()
    if not protocol_b_control.empty:
        protocol_b_control = protocol_b_control[protocol_b_control["agent_episode"].astype(int).isin(selected)].copy()
        protocol_b_control["evaluator_episode"] = protocol_b_control["agent_episode"].astype(int)

    for df in [offline_same, offline_cross, compound_df, protocol_b_repr, protocol_b_control]:
        if df.empty:
            continue
        if "evaluator_episode" in df.columns:
            df["stage_label"] = df["evaluator_episode"].astype(int).map(stage_map)
        if "generator_episode" in df.columns:
            df["generator_stage_label"] = df["generator_episode"].astype(int).map(stage_map)

    return {
        "families": sorted(families),
        "stage_map": stage_map,
        "offline_same": offline_same,
        "offline_cross": offline_cross,
        "compound": compound_df,
        "protocol_b_repr": protocol_b_repr,
        "protocol_b_control": protocol_b_control,
    }


def choose_representation_metrics(frames: list[pd.DataFrame]) -> list[str]:
    available = set()
    for df in frames:
        if df.empty:
            continue
        available.update(df.columns)
    metrics = []
    if "linreg_rsq-0" in available:
        metrics.append("linreg_rsq-0")
    if "normalized_MI" in available:
        metrics.append("normalized_MI")
    elif "MI" in available:
        metrics.append("MI")
    return metrics


def compute_degradation(df: pd.DataFrame, metric: str, *, group_cols: list[str]) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    base = df[df["task_name"] == "base"][group_cols + [metric]].copy()
    if base.empty:
        return pd.DataFrame()
    base = base.groupby(group_cols, as_index=False)[metric].mean()
    base = base.rename(columns={metric: "base_value"})

    var = df[df["task_name"] != "base"].copy()
    if var.empty:
        return pd.DataFrame()

    merged = var.merge(base, on=group_cols, how="left")
    merged[metric] = coerce_numeric(merged[metric])
    merged["base_value"] = coerce_numeric(merged["base_value"])
    if metric in HIGHER_IS_BETTER:
        merged["degradation_vs_base"] = merged["base_value"] - merged[metric]
    else:
        merged["degradation_vs_base"] = merged[metric] - merged["base_value"]
    merged["metric_name"] = metric
    return merged


def extract_base_rows(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    out = df[df["task_name"] == "base"].copy()
    out["metric_value"] = coerce_numeric(out[metric])
    out["metric_name"] = metric
    return out


def plot_base_performance(rep_sources: dict[str, pd.DataFrame], control_df: pd.DataFrame,
                          stage_map: dict[int, str], family: str, output_dir: Path,
                          rep_metrics: list[str], control_metric: str | None) -> None:
    panels = []
    for metric in rep_metrics:
        rows = []
        for setup_name in SETUP_ORDER:
            frame = rep_sources.get(setup_name, pd.DataFrame())
            if frame.empty:
                continue
            base_rows = extract_base_rows(frame[frame["task_name"].isin([family, "base"])], metric)
            if base_rows.empty:
                continue
            base_rows["setup"] = setup_name
            rows.append(base_rows)
        if rows:
            panels.append((metric, pd.concat(rows, ignore_index=True)))

    if control_metric is not None and not control_df.empty and control_metric in control_df.columns:
        control_rows = extract_base_rows(control_df[control_df["task_name"].isin([family, "base"])], control_metric)
        if not control_rows.empty:
            control_rows["setup"] = "protocol_b_original"
            panels.append((control_metric, control_rows))

    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.8), squeeze=False)
    stage_order = list(stage_map.keys())
    stage_names = [stage_map[ep] for ep in stage_order]

    for ax, (metric, data) in zip(axes[0], panels):
        for setup_name in SETUP_ORDER:
            if setup_name not in data["setup"].unique():
                continue
            subset = data[data["setup"] == setup_name].copy()
            subset = subset.dropna(subset=["metric_value"])
            if subset.empty:
                continue
            values = []
            for episode in stage_order:
                stage_subset = subset[subset["evaluator_episode"].astype(int) == int(episode)]
                values.append(stage_subset["metric_value"].mean() if not stage_subset.empty else np.nan)
            ax.plot(
                stage_names,
                values,
                marker=SETUP_MARKERS[setup_name],
                color=SETUP_COLORS[setup_name],
                label=SETUP_LABELS[setup_name],
            )
        ax.set_title(short_metric_label(metric))
        ax.set_xlabel("Training stage")
        ax.set_ylabel(short_metric_label(metric))
        ax.tick_params(axis="x", rotation=0)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3), bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(f"Matched performance: {short_task_label(family)}", y=1.08)
    fig.tight_layout()
    ensure_dir(output_dir)
    fig.savefig(output_dir / f"{sanitize(family)}_matched_base_performance.png")
    plt.close(fig)


def is_numeric_sweep(data: pd.DataFrame) -> bool:
    if "task_value_num" not in data.columns:
        return False
    values = data["task_value_num"].dropna().unique()
    return len(values) >= 2


def categorical_order(data: pd.DataFrame) -> list[str]:
    values = data["task_value_str"].replace({"": np.nan}).dropna().unique().tolist()
    return values


def plot_mismatch_penalty_curves(rep_sources: dict[str, pd.DataFrame], control_df: pd.DataFrame,
                                 stage_map: dict[int, str], family: str, output_dir: Path,
                                 rep_metrics: list[str], control_metric: str | None) -> None:
    metric_frames = []
    for metric in rep_metrics:
        rows = []
        for setup_name in SETUP_ORDER:
            frame = rep_sources.get(setup_name, pd.DataFrame())
            if frame.empty:
                continue
            deg = compute_degradation(
                frame[frame["task_name"].isin([family, "base"])],
                metric,
                group_cols=["setup", "evaluator_episode"],
            )
            if not deg.empty:
                rows.append(deg)
        if rows:
            metric_frames.append((metric, pd.concat(rows, ignore_index=True)))

    if control_metric is not None and not control_df.empty and control_metric in control_df.columns:
        control_deg = compute_degradation(
            control_df[control_df["task_name"].isin([family, "base"])],
            control_metric,
            group_cols=["setup", "evaluator_episode"],
        )
        if not control_deg.empty:
            metric_frames.append((control_metric, control_deg))

    if not metric_frames:
        return

    stage_order = list(stage_map.keys())
    stage_names = [stage_map[ep] for ep in stage_order]

    for metric, frame in metric_frames:
        fig, axes = plt.subplots(1, len(stage_order), figsize=(5.2 * len(stage_order), 4.5), squeeze=False, sharey=True)
        numeric = is_numeric_sweep(frame)
        x_label = short_task_label(family)

        for ax, episode, stage_label in zip(axes[0], stage_order, stage_names):
            stage_frame = frame[frame["evaluator_episode"].astype(int) == int(episode)].copy()
            if stage_frame.empty:
                continue

            for setup_name in SETUP_ORDER:
                subset = stage_frame[stage_frame["setup"] == setup_name].copy()
                if subset.empty:
                    continue
                if numeric:
                    subset = subset.groupby("task_value_num", as_index=False)["degradation_vs_base"].mean()
                    subset = subset.sort_values("task_value_num")
                    ax.plot(
                        subset["task_value_num"],
                        subset["degradation_vs_base"],
                        marker=SETUP_MARKERS[setup_name],
                        color=SETUP_COLORS[setup_name],
                        label=SETUP_LABELS[setup_name],
                    )
                else:
                    categories = categorical_order(stage_frame)
                    cat_to_pos = {cat: idx for idx, cat in enumerate(categories)}
                    offsets = {
                        "protocol_b_original": -0.22,
                        "compound_online": 0.0,
                        "offline_same": 0.22,
                    }
                    subset = subset.groupby("task_value_str", as_index=False)["degradation_vs_base"].mean()
                    positions = [cat_to_pos[val] + offsets.get(setup_name, 0.0) for val in subset["task_value_str"]]
                    ax.scatter(
                        positions,
                        subset["degradation_vs_base"],
                        marker=SETUP_MARKERS[setup_name],
                        color=SETUP_COLORS[setup_name],
                        s=56,
                        label=SETUP_LABELS[setup_name],
                    )
                    ax.set_xticks(range(len(categories)))
                    ax.set_xticklabels(categories, rotation=15)
            ax.axhline(0.0, color="#888888", linewidth=1.1)
            ax.set_title(stage_label)
            ax.set_xlabel(x_label)

        axes[0][0].set_ylabel(f"{short_metric_label(metric)} penalty")
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3), bbox_to_anchor=(0.5, 1.06))
        fig.suptitle(f"Mismatch penalty: {short_metric_label(metric)} in {short_task_label(family)}", y=1.10)
        fig.tight_layout()
        ensure_dir(output_dir)
        fig.savefig(output_dir / f"{sanitize(family)}_mismatch_penalty_{sanitize(metric)}.png")
        plt.close(fig)


def plot_training_effect(rep_sources: dict[str, pd.DataFrame], control_df: pd.DataFrame,
                         stage_map: dict[int, str], family: str, output_dir: Path,
                         rep_metrics: list[str], control_metric: str | None) -> None:
    panels = []
    for metric in rep_metrics:
        rows = []
        for setup_name in SETUP_ORDER:
            frame = rep_sources.get(setup_name, pd.DataFrame())
            if frame.empty:
                continue
            deg = compute_degradation(
                frame[frame["task_name"].isin([family, "base"])],
                metric,
                group_cols=["setup", "evaluator_episode"],
            )
            if deg.empty:
                continue
            summary = deg.groupby(["setup", "evaluator_episode"], as_index=False)["degradation_vs_base"].mean()
            summary["metric_name"] = metric
            rows.append(summary)
        if rows:
            panels.append((metric, pd.concat(rows, ignore_index=True)))

    if control_metric is not None and not control_df.empty and control_metric in control_df.columns:
        deg = compute_degradation(
            control_df[control_df["task_name"].isin([family, "base"])],
            control_metric,
            group_cols=["setup", "evaluator_episode"],
        )
        if not deg.empty:
            summary = deg.groupby(["setup", "evaluator_episode"], as_index=False)["degradation_vs_base"].mean()
            summary["metric_name"] = control_metric
            panels.append((control_metric, summary))

    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.8), squeeze=False)
    stage_order = list(stage_map.keys())
    stage_names = [stage_map[ep] for ep in stage_order]

    for ax, (metric, data) in zip(axes[0], panels):
        for setup_name in SETUP_ORDER:
            subset = data[data["setup"] == setup_name].copy()
            if subset.empty:
                continue
            values = []
            for episode in stage_order:
                stage_subset = subset[subset["evaluator_episode"].astype(int) == int(episode)]
                values.append(stage_subset["degradation_vs_base"].mean() if not stage_subset.empty else np.nan)
            ax.plot(
                stage_names,
                values,
                marker=SETUP_MARKERS[setup_name],
                color=SETUP_COLORS[setup_name],
                label=SETUP_LABELS[setup_name],
            )
        ax.axhline(0.0, color="#888888", linewidth=1.1)
        ax.set_title(short_metric_label(metric))
        ax.set_xlabel("Training stage")
        ax.set_ylabel(f"Mean {short_metric_label(metric)} penalty")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3), bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(f"Training effect on robustness: {short_task_label(family)}", y=1.08)
    fig.tight_layout()
    ensure_dir(output_dir)
    fig.savefig(output_dir / f"{sanitize(family)}_training_effect_on_robustness.png")
    plt.close(fig)


def plot_cross_replay_heatmaps(offline_cross: pd.DataFrame, stage_map: dict[int, str], family: str,
                               output_dir: Path, rep_metrics: list[str]) -> None:
    if offline_cross.empty:
        return

    panels = []
    for metric in rep_metrics:
        deg = compute_degradation(
            offline_cross[offline_cross["task_name"].isin([family, "base"])],
            metric,
            group_cols=["setup", "generator_episode", "evaluator_episode"],
        )
        if deg.empty:
            continue
        summary = deg.groupby(["generator_episode", "evaluator_episode"], as_index=False)["degradation_vs_base"].mean()
        summary["metric_name"] = metric
        panels.append((metric, summary))

    if not panels:
        return

    stage_order = list(stage_map.keys())
    stage_names = [stage_map[ep] for ep in stage_order]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.8), squeeze=False)

    for ax, (metric, summary) in zip(axes[0], panels):
        pivot = summary.pivot(index="evaluator_episode", columns="generator_episode", values="degradation_vs_base")
        pivot = pivot.reindex(index=stage_order, columns=stage_order)
        values = pivot.to_numpy(dtype=float)
        finite = np.isfinite(values)
        vmax = np.nanmax(np.abs(values[finite])) if finite.any() else 1.0
        vmax = max(vmax, 1e-6)
        image = ax.imshow(values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(stage_order)))
        ax.set_yticks(range(len(stage_order)))
        ax.set_xticklabels(stage_names, rotation=20)
        ax.set_yticklabels(stage_names)
        ax.set_xlabel("Generator stage")
        ax.set_ylabel("Evaluator stage")
        ax.set_title(short_metric_label(metric))

        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                value = values[row_idx, col_idx]
                if np.isnan(value):
                    continue
                color = "white" if abs(value) > vmax * 0.55 else "black"
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color=color, fontsize=10)

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Offline cross-replay penalty heatmap: {short_task_label(family)}", y=1.02)
    fig.tight_layout()
    ensure_dir(output_dir)
    fig.savefig(output_dir / f"{sanitize(family)}_offline_cross_replay_heatmaps.png")
    plt.close(fig)


def save_processed_tables(output_dir: Path, rep_sources: dict[str, pd.DataFrame], control_df: pd.DataFrame,
                          families: list[str], rep_metrics: list[str], control_metric: str | None) -> None:
    ensure_dir(output_dir)

    rep_frames = []
    for setup_name, df in rep_sources.items():
        if df.empty:
            continue
        keep_cols = [col for col in [
            "setup", "generator_episode", "evaluator_episode", "stage_label", "generator_stage_label",
            "variant", "task_name", "task_value", "task_value_num", "task_value_str",
            *rep_metrics,
        ] if col in df.columns]
        rep_frames.append(df[keep_cols].copy())
    if rep_frames:
        pd.concat(rep_frames, ignore_index=True).to_csv(output_dir / "representation_sources.csv", index=False)

    if not control_df.empty and control_metric is not None and control_metric in control_df.columns:
        keep_cols = [col for col in [
            "setup", "evaluator_episode", "stage_label", "variant", "task_name", "task_value",
            "task_value_num", "task_value_str", control_metric,
        ] if col in control_df.columns]
        control_df[keep_cols].copy().to_csv(output_dir / "control_source.csv", index=False)

    degradation_frames = []
    for family in families:
        for metric in rep_metrics:
            for setup_name, df in rep_sources.items():
                if setup_name == "offline_cross" or df.empty:
                    continue
                deg = compute_degradation(
                    df[df["task_name"].isin([family, "base"])],
                    metric,
                    group_cols=["setup", "evaluator_episode"],
                )
                if not deg.empty:
                    degradation_frames.append(deg)
        if control_metric is not None and not control_df.empty and control_metric in control_df.columns:
            deg = compute_degradation(
                control_df[control_df["task_name"].isin([family, "base"])],
                control_metric,
                group_cols=["setup", "evaluator_episode"],
            )
            if not deg.empty:
                degradation_frames.append(deg)
    if degradation_frames:
        pd.concat(degradation_frames, ignore_index=True).to_csv(output_dir / "degradation_long.csv", index=False)


def write_manifest(output_dir: Path, *, run_root: Path, discovered: dict, stage_map: dict[int, str], families: list[str]) -> None:
    payload = {
        "run_root": str(run_root),
        "train_id": discovered["train_id"],
        "offline_summary": str(discovered["offline_path"]) if discovered["offline_path"] else None,
        "compound_summary": str(discovered["compound_path"]) if discovered["compound_path"] else None,
        "protocol_b_workbooks": [str(path) for path in discovered["protocol_b_workbooks"]],
        "protocol_b_belief_csvs": [str(path) for path in discovered["protocol_b_belief_csvs"]],
        "stage_map": {str(k): v for k, v in stage_map.items()},
        "families": families,
    }
    ensure_dir(output_dir)
    (output_dir / "figure_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(args):
    set_paper_style()

    run_root = Path(args.run_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (run_root / "plots_comparison").resolve()
    report_root = Path(args.report_root).resolve() if args.report_root else (Path(__file__).resolve().parent / "report").resolve()

    discovered = load_run_data(
        run_root,
        report_root,
        args.protocol_b_xlsx,
        args.protocol_b_belief_csv,
    )
    normalized = normalize_sources(discovered)

    rep_sources = {
        "protocol_b_original": normalized["protocol_b_repr"],
        "compound_online": normalized["compound"],
        "offline_same": normalized["offline_same"],
        "offline_cross": normalized["offline_cross"],
    }
    control_df = normalized["protocol_b_control"]
    stage_map = normalized["stage_map"]
    families = normalized["families"]
    rep_metrics = choose_representation_metrics([
        rep_sources["protocol_b_original"],
        rep_sources["compound_online"],
        rep_sources["offline_same"],
        rep_sources["offline_cross"],
    ])
    control_metric = (
        "comparison_rollout_mean_drqn_disc_return"
        if not control_df.empty and "comparison_rollout_mean_drqn_disc_return" in control_df.columns
        else None
    )

    if not rep_metrics and control_metric is None:
        raise ValueError("Could not find any supported metrics to plot.")

    print(f"train_id={discovered['train_id']}", flush=True)
    print(f"families={families}", flush=True)
    print(f"stage_map={stage_map}", flush=True)

    figures_dir = output_dir / "figures"
    processed_dir = output_dir / "processed"
    ensure_dir(figures_dir)
    ensure_dir(processed_dir)

    for family in families:
        family_dir = figures_dir / sanitize(family)
        ensure_dir(family_dir)
        plot_base_performance(rep_sources, control_df, stage_map, family, family_dir, rep_metrics, control_metric)
        plot_mismatch_penalty_curves(rep_sources, control_df, stage_map, family, family_dir, rep_metrics, control_metric)
        plot_training_effect(rep_sources, control_df, stage_map, family, family_dir, rep_metrics, control_metric)
        plot_cross_replay_heatmaps(rep_sources["offline_cross"], stage_map, family, family_dir, rep_metrics)

    save_processed_tables(processed_dir, rep_sources, control_df, families, rep_metrics, control_metric)
    write_manifest(processed_dir, run_root=run_root, discovered=discovered, stage_map=stage_map, families=families)
    print(f"Saved comparison figures to: {figures_dir}", flush=True)
    print(f"Saved processed tables to: {processed_dir}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser("Create paper-quality comparison plots for offline replay, compound-online, and Protocol B results.")
    parser.add_argument("--run-root", type=str, required=True,
                        help="Offline run folder, typically cache/<name>.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Defaults to <run-root>/plots_comparison.")
    parser.add_argument("--report-root", type=str, default=None,
                        help="Root used to auto-discover matching Protocol B outputs. Defaults to ./report.")
    parser.add_argument("--protocol-b-xlsx", type=str, nargs="*", default=None,
                        help="Optional explicit Protocol B workbook path(s).")
    parser.add_argument("--protocol-b-belief-csv", type=str, nargs="*", default=None,
                        help="Optional explicit Protocol B belief summary CSV path(s).")
    args = parser.parse_args()
    main(args)
