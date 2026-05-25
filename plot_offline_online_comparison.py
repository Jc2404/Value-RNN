import json
import math
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
    "MI": "MI",
    "softmax_linear_KL": "Linear probe KL divergence",
    "softmax_linear_CE": "Linear probe cross-entropy",
    "softmax_linear_JS": "Linear probe JS divergence",
    "softmax_mlp_KL": "MLP probe KL divergence",
    "softmax_mlp_CE": "MLP probe cross-entropy",
    "softmax_mlp_JS": "MLP probe JS divergence",
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
        return ["Untrained", "Intermediate", "Trained"][:count]
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


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


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
    return pd.concat(frames, ignore_index=True)


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


def parse_base_task_value(raw_value: str | None):
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return text


def baseline_mask(df: pd.DataFrame, base_task_value) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)

    mask = pd.Series(False, index=df.index)
    if "task_name" in df.columns:
        mask = mask | (df["task_name"].astype(str) == "base")

    if base_task_value is None or "task_value" not in df.columns:
        return mask

    if isinstance(base_task_value, float):
        if "task_value_num" in df.columns:
            numeric = coerce_numeric(df["task_value_num"])
        else:
            numeric = coerce_numeric(df["task_value"])
        numeric_match = pd.Series(
            np.isclose(numeric, float(base_task_value), atol=1e-8, rtol=1e-6),
            index=df.index,
        )
        mask = mask | numeric_match
    else:
        string_values = df.get("task_value_str", df["task_value"].astype(str).str.strip())
        mask = mask | (string_values.astype(str).str.strip() == str(base_task_value))
    return mask


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
    if "MI" in available:
        metrics.append("MI")
    for metric in [
        "softmax_linear_KL",
        "softmax_linear_CE",
        "softmax_linear_JS",
        "softmax_mlp_KL",
        "softmax_mlp_CE",
        "softmax_mlp_JS",
    ]:
        if metric in available:
            metrics.append(metric)
    return metrics


def compute_degradation(df: pd.DataFrame, metric: str, *, group_cols: list[str], base_task_value=None) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    base_rows = baseline_mask(df, base_task_value)
    base = df[base_rows][group_cols + [metric]].copy()
    if base.empty:
        return pd.DataFrame()
    base = base.groupby(group_cols, as_index=False)[metric].mean()
    base = base.rename(columns={metric: "base_value"})

    var = df[~base_rows].copy()
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


def extract_base_rows(df: pd.DataFrame, metric: str, base_task_value=None) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    out = df[baseline_mask(df, base_task_value)].copy()
    out["metric_value"] = coerce_numeric(out[metric])
    out["metric_name"] = metric
    return out


def plot_base_performance(rep_sources: dict[str, pd.DataFrame], control_df: pd.DataFrame,
                          stage_map: dict[int, str], family: str, output_dir: Path,
                          rep_metrics: list[str], control_metric: str | None, base_task_value=None) -> None:
    for metric in rep_metrics:
        rows = []
        for setup_name in SETUP_ORDER:
            frame = rep_sources.get(setup_name, pd.DataFrame())
            if frame.empty:
                continue
            base_rows = extract_base_rows(frame[frame["task_name"].isin([family, "base"])], metric, base_task_value)
            if base_rows.empty:
                continue
            base_rows["setup"] = setup_name
            rows.append(base_rows)
        if not rows:
            continue

        data = pd.concat(rows, ignore_index=True)
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        stage_order = list(stage_map.keys())
        stage_names = [stage_map[ep] for ep in stage_order]

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

        ax.set_title(f"Matched performance: {short_metric_label(metric)}")
        ax.set_xlabel("Training stage")
        ax.set_ylabel(short_metric_label(metric))
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3), bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout()
        ensure_dir(output_dir)
        fig.savefig(output_dir / f"{sanitize(family)}_matched_base_performance_{sanitize(metric)}.png")
        plt.close(fig)

    if control_metric is not None and not control_df.empty and control_metric in control_df.columns:
        control_rows = extract_base_rows(
            control_df[control_df["task_name"].isin([family, "base"])],
            control_metric,
            base_task_value,
        )
        if not control_rows.empty:
            control_rows["setup"] = "protocol_b_original"
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            stage_order = list(stage_map.keys())
            stage_names = [stage_map[ep] for ep in stage_order]
            subset = control_rows.dropna(subset=["metric_value"]).copy()
            values = []
            for episode in stage_order:
                stage_subset = subset[subset["evaluator_episode"].astype(int) == int(episode)]
                values.append(stage_subset["metric_value"].mean() if not stage_subset.empty else np.nan)
            ax.plot(
                stage_names,
                values,
                marker=SETUP_MARKERS["protocol_b_original"],
                color=SETUP_COLORS["protocol_b_original"],
                label=SETUP_LABELS["protocol_b_original"],
            )
            ax.set_title(f"Matched performance: {short_metric_label(control_metric)}")
            ax.set_xlabel("Training stage")
            ax.set_ylabel(short_metric_label(control_metric))
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.02))
            fig.tight_layout()
            ensure_dir(output_dir)
            fig.savefig(output_dir / f"{sanitize(family)}_matched_base_performance_{sanitize(control_metric)}.png")
            plt.close(fig)


def is_numeric_sweep(data: pd.DataFrame) -> bool:
    if "task_value_num" not in data.columns:
        return False
    values = data["task_value_num"].dropna().unique()
    return len(values) >= 2


def categorical_order(data: pd.DataFrame) -> list[str]:
    values = data["task_value_str"].replace({"": np.nan}).dropna().unique().tolist()
    return values


def ordered_variant_rows(data: pd.DataFrame) -> list[tuple[str, str]]:
    if data.empty:
        return []
    numeric = data[["variant", "task_value_num"]].drop_duplicates().dropna(subset=["task_value_num"]).sort_values("task_value_num")
    ordered = [(str(row["variant"]), str(row["task_value_num"]).rstrip("0").rstrip(".") if "." in str(row["task_value_num"]) else str(row["task_value_num"])) for _, row in numeric.iterrows()]

    seen = {variant for variant, _ in ordered}
    categorical = data[["variant", "task_value_str"]].drop_duplicates()
    for _, row in categorical.iterrows():
        variant = str(row["variant"])
        if variant in seen:
            continue
        value = str(row["task_value_str"]).strip()
        if value:
            ordered.append((variant, value))
            seen.add(variant)
    return ordered


def draw_heatmap(ax, values: np.ndarray, stage_names: list[str], title: str, *,
                 cmap: str, symmetric: bool) -> None:
    finite = np.isfinite(values)
    if finite.any():
        if symmetric:
            vmax = np.nanmax(np.abs(values[finite]))
            vmax = max(vmax, 1e-6)
            vmin = -vmax
        else:
            vmin = float(np.nanmin(values[finite]))
            vmax = float(np.nanmax(values[finite]))
            if math.isclose(vmin, vmax):
                delta = max(abs(vmin) * 0.05, 1e-6)
                vmin -= delta
                vmax += delta
    else:
        vmin, vmax = (-1.0, 1.0) if symmetric else (0.0, 1.0)

    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(stage_names)))
    ax.set_yticks(range(len(stage_names)))
    ax.set_xticklabels(stage_names, rotation=20)
    ax.set_yticklabels(stage_names)
    ax.set_xlabel("Generator stage")
    ax.set_ylabel("Evaluator stage")
    ax.set_title(title)

    scale = max(abs(vmax), abs(vmin), 1e-9)
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isnan(value):
                continue
            color = "white" if abs(value) > scale * 0.55 else "black"
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color=color, fontsize=10)

    return image


def plot_mismatch_penalty_curves(rep_sources: dict[str, pd.DataFrame], control_df: pd.DataFrame,
                                 stage_map: dict[int, str], family: str, output_dir: Path,
                                 rep_metrics: list[str], control_metric: str | None, base_task_value=None) -> None:
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
                base_task_value=base_task_value,
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
            base_task_value=base_task_value,
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
                         rep_metrics: list[str], control_metric: str | None, base_task_value=None) -> None:
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
                base_task_value=base_task_value,
            )
            if deg.empty:
                continue
            summary = deg.groupby(["setup", "evaluator_episode"], as_index=False)["degradation_vs_base"].mean()
            summary["metric_name"] = metric
            rows.append(summary)
        if not rows:
            continue

        data = pd.concat(rows, ignore_index=True)
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        stage_order = list(stage_map.keys())
        stage_names = [stage_map[ep] for ep in stage_order]

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
        ax.set_title(f"Training effect: {short_metric_label(metric)}")
        ax.set_xlabel("Training stage")
        ax.set_ylabel(f"Mean {short_metric_label(metric)} penalty")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3), bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout()
        ensure_dir(output_dir)
        fig.savefig(output_dir / f"{sanitize(family)}_training_effect_on_robustness_{sanitize(metric)}.png")
        plt.close(fig)

    if control_metric is not None and not control_df.empty and control_metric in control_df.columns:
        deg = compute_degradation(
            control_df[control_df["task_name"].isin([family, "base"])],
            control_metric,
            group_cols=["setup", "evaluator_episode"],
            base_task_value=base_task_value,
        )
        if not deg.empty:
            summary = deg.groupby(["setup", "evaluator_episode"], as_index=False)["degradation_vs_base"].mean()
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            stage_order = list(stage_map.keys())
            stage_names = [stage_map[ep] for ep in stage_order]
            subset = summary.copy()
            values = []
            for episode in stage_order:
                stage_subset = subset[subset["evaluator_episode"].astype(int) == int(episode)]
                values.append(stage_subset["degradation_vs_base"].mean() if not stage_subset.empty else np.nan)
            ax.plot(
                stage_names,
                values,
                marker=SETUP_MARKERS["protocol_b_original"],
                color=SETUP_COLORS["protocol_b_original"],
                label=SETUP_LABELS["protocol_b_original"],
            )
            ax.axhline(0.0, color="#888888", linewidth=1.1)
            ax.set_title(f"Training effect: {short_metric_label(control_metric)}")
            ax.set_xlabel("Training stage")
            ax.set_ylabel(f"Mean {short_metric_label(control_metric)} penalty")
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.02))
            fig.tight_layout()
            ensure_dir(output_dir)
            fig.savefig(output_dir / f"{sanitize(family)}_training_effect_on_robustness_{sanitize(control_metric)}.png")
            plt.close(fig)


def plot_offline_vs_protocol_b_sanity(rep_sources: dict[str, pd.DataFrame], stage_map: dict[int, str],
                                      family: str, output_dir: Path, rep_metrics: list[str],
                                      base_task_value=None) -> None:
    offline_same = rep_sources.get("offline_same", pd.DataFrame())
    protocol_b = rep_sources.get("protocol_b_original", pd.DataFrame())
    if offline_same.empty or protocol_b.empty:
        return

    stage_order = list(stage_map.keys())
    stage_names = [stage_map[ep] for ep in stage_order]

    for metric in rep_metrics:
        if metric not in offline_same.columns or metric not in protocol_b.columns:
            continue

        offline_frame = offline_same[offline_same["task_name"].isin([family, "base"])].copy()
        protocol_frame = protocol_b[protocol_b["task_name"].isin([family, "base"])].copy()
        combined = pd.concat([offline_frame, protocol_frame], ignore_index=True)
        if combined.empty:
            continue

        numeric = is_numeric_sweep(combined)
        fig, axes = plt.subplots(1, len(stage_order), figsize=(5.2 * len(stage_order), 4.5), squeeze=False, sharey=True)

        for ax, episode, stage_label in zip(axes[0], stage_order, stage_names):
            stage_offline = offline_frame[offline_frame["evaluator_episode"].astype(int) == int(episode)].copy()
            stage_protocol = protocol_frame[protocol_frame["evaluator_episode"].astype(int) == int(episode)].copy()
            if stage_offline.empty and stage_protocol.empty:
                continue

            for setup_name, subset in [
                ("offline_same", stage_offline),
                ("protocol_b_original", stage_protocol),
            ]:
                if subset.empty:
                    continue
                subset = subset.copy()
                subset["metric_value"] = coerce_numeric(subset[metric])
                subset = subset.dropna(subset=["metric_value"])
                if subset.empty:
                    continue

                if numeric:
                    subset = subset.sort_values("task_value_num")
                    ax.plot(
                        subset["task_value_num"],
                        subset["metric_value"],
                        marker=SETUP_MARKERS[setup_name],
                        color=SETUP_COLORS[setup_name],
                        label=SETUP_LABELS[setup_name],
                    )
                    baseline_subset = subset[baseline_mask(subset, base_task_value)]
                    if not baseline_subset.empty:
                        ax.scatter(
                            baseline_subset["task_value_num"],
                            baseline_subset["metric_value"],
                            marker="x",
                            s=70,
                            color=SETUP_COLORS[setup_name],
                        )
                else:
                    categories = categorical_order(pd.concat([stage_offline, stage_protocol], ignore_index=True))
                    cat_to_pos = {cat: idx for idx, cat in enumerate(categories)}
                    offset = -0.12 if setup_name == "protocol_b_original" else 0.12
                    subset = subset.groupby("task_value_str", as_index=False)["metric_value"].mean()
                    positions = [cat_to_pos[val] + offset for val in subset["task_value_str"]]
                    ax.scatter(
                        positions,
                        subset["metric_value"],
                        marker=SETUP_MARKERS[setup_name],
                        color=SETUP_COLORS[setup_name],
                        s=60,
                        label=SETUP_LABELS[setup_name],
                    )
                    ax.plot(
                        positions,
                        subset["metric_value"],
                        color=SETUP_COLORS[setup_name],
                        linewidth=1.6,
                    )
                    ax.set_xticks(range(len(categories)))
                    ax.set_xticklabels(categories, rotation=15)

            ax.set_title(stage_label)
            ax.set_xlabel(short_task_label(family))

        axes[0][0].set_ylabel(short_metric_label(metric))
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 2), bbox_to_anchor=(0.5, 1.05))
        fig.suptitle(f"Sanity check: offline diagonal vs Protocol B for {short_metric_label(metric)}", y=1.10)
        fig.tight_layout()
        ensure_dir(output_dir)
        fig.savefig(output_dir / f"{sanitize(family)}_sanity_offline_vs_protocol_b_{sanitize(metric)}.png")
        plt.close(fig)


def plot_cross_replay_heatmaps(offline_cross: pd.DataFrame, stage_map: dict[int, str], family: str,
                               output_dir: Path, rep_metrics: list[str], base_task_value=None) -> None:
    if offline_cross.empty:
        return

    penalty_panels = []
    raw_panels = []
    for metric in rep_metrics:
        raw_source = offline_cross[
            offline_cross["task_name"].isin([family, "base"])
        ].copy()
        raw_source = raw_source[~baseline_mask(raw_source, base_task_value)].copy()
        if not raw_source.empty and metric in raw_source.columns:
            raw_summary = raw_source.groupby(
                ["generator_episode", "evaluator_episode"], as_index=False
            )[metric].mean()
            raw_summary["metric_name"] = metric
            raw_panels.append((metric, raw_summary))

        deg = compute_degradation(
            offline_cross[offline_cross["task_name"].isin([family, "base"])],
            metric,
            group_cols=["setup", "generator_episode", "evaluator_episode"],
            base_task_value=base_task_value,
        )
        if deg.empty:
            continue
        summary = deg.groupby(["generator_episode", "evaluator_episode"], as_index=False)["degradation_vs_base"].mean()
        summary["metric_name"] = metric
        penalty_panels.append((metric, summary))

    if not penalty_panels and not raw_panels:
        return

    stage_order = list(stage_map.keys())
    stage_names = [stage_map[ep] for ep in stage_order]
    ensure_dir(output_dir)

    if penalty_panels:
        for metric, summary in penalty_panels:
            fig, ax = plt.subplots(figsize=(5.4, 4.8))
            pivot = summary.pivot(index="evaluator_episode", columns="generator_episode", values="degradation_vs_base")
            pivot = pivot.reindex(index=stage_order, columns=stage_order)
            image = draw_heatmap(
                ax,
                pivot.to_numpy(dtype=float),
                stage_names,
                short_metric_label(metric),
                cmap="coolwarm",
                symmetric=True,
            )
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(f"Offline cross-replay penalty: {short_metric_label(metric)}", y=1.02)
            fig.tight_layout()
            fig.savefig(output_dir / f"{sanitize(family)}_offline_cross_replay_penalty_{sanitize(metric)}.png")
            plt.close(fig)

    if raw_panels:
        for metric, summary in raw_panels:
            fig, ax = plt.subplots(figsize=(5.4, 4.8))
            pivot = summary.pivot(index="evaluator_episode", columns="generator_episode", values=metric)
            pivot = pivot.reindex(index=stage_order, columns=stage_order)
            image = draw_heatmap(
                ax,
                pivot.to_numpy(dtype=float),
                stage_names,
                short_metric_label(metric),
                cmap="viridis",
                symmetric=False,
            )
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(f"Offline cross-replay raw value: {short_metric_label(metric)}", y=1.02)
            fig.tight_layout()
            fig.savefig(output_dir / f"{sanitize(family)}_offline_cross_replay_raw_{sanitize(metric)}.png")
            plt.close(fig)

    per_variant_dir = output_dir / "per_variant_heatmaps"
    ensure_dir(per_variant_dir)
    family_source = offline_cross[offline_cross["task_name"].isin([family, "base"])].copy()
    base_source = family_source[baseline_mask(family_source, base_task_value)].copy()
    variant_rows = ordered_variant_rows(family_source[~baseline_mask(family_source, base_task_value)].copy())

    if not base_source.empty:
        base_raw_panels = []
        for metric in rep_metrics:
            if metric not in base_source.columns:
                continue
            raw_summary = base_source.groupby(
                ["generator_episode", "evaluator_episode"], as_index=False
            )[metric].mean()
            if not raw_summary.empty:
                base_raw_panels.append((metric, raw_summary))

        if base_raw_panels:
            for metric, summary in base_raw_panels:
                fig, ax = plt.subplots(figsize=(5.4, 4.8))
                pivot = summary.pivot(index="evaluator_episode", columns="generator_episode", values=metric)
                pivot = pivot.reindex(index=stage_order, columns=stage_order)
                image = draw_heatmap(
                    ax,
                    pivot.to_numpy(dtype=float),
                    stage_names,
                    short_metric_label(metric),
                    cmap="viridis",
                    symmetric=False,
                )
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                fig.suptitle(
                    f"{short_task_label(family)} matched baseline: {short_metric_label(metric)}",
                    y=1.02,
                )
                fig.tight_layout()
                fig.savefig(per_variant_dir / f"{sanitize(family)}_base_raw_{sanitize(metric)}.png")
                plt.close(fig)

    for variant_name, variant_value_label in variant_rows:
        variant_source = family_source[family_source["variant"].astype(str) == str(variant_name)].copy()
        if variant_source.empty:
            continue

        variant_penalty_panels = []
        variant_raw_panels = []
        for metric in rep_metrics:
            if metric not in variant_source.columns:
                continue

            raw_summary = variant_source.groupby(
                ["generator_episode", "evaluator_episode"], as_index=False
            )[metric].mean()
            if not raw_summary.empty:
                variant_raw_panels.append((metric, raw_summary))

            deg = compute_degradation(
                pd.concat([variant_source, family_source[baseline_mask(family_source, base_task_value)]], ignore_index=True),
                metric,
                group_cols=["setup", "generator_episode", "evaluator_episode"],
                base_task_value=base_task_value,
            )
            deg = deg[deg["variant"].astype(str) == str(variant_name)].copy()
            if not deg.empty:
                penalty_summary = deg.groupby(
                    ["generator_episode", "evaluator_episode"], as_index=False
                )["degradation_vs_base"].mean()
                variant_penalty_panels.append((metric, penalty_summary))

        if variant_penalty_panels:
            for metric, summary in variant_penalty_panels:
                fig, ax = plt.subplots(figsize=(5.4, 4.8))
                pivot = summary.pivot(index="evaluator_episode", columns="generator_episode", values="degradation_vs_base")
                pivot = pivot.reindex(index=stage_order, columns=stage_order)
                image = draw_heatmap(
                    ax,
                    pivot.to_numpy(dtype=float),
                    stage_names,
                    short_metric_label(metric),
                    cmap="coolwarm",
                    symmetric=True,
                )
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                fig.suptitle(
                    f"{short_task_label(family)} variant {variant_value_label}: {short_metric_label(metric)} penalty",
                    y=1.02,
                )
                fig.tight_layout()
                fig.savefig(per_variant_dir / f"{sanitize(family)}_{sanitize(variant_value_label)}_penalty_{sanitize(metric)}.png")
                plt.close(fig)

        if variant_raw_panels:
            for metric, summary in variant_raw_panels:
                fig, ax = plt.subplots(figsize=(5.4, 4.8))
                pivot = summary.pivot(index="evaluator_episode", columns="generator_episode", values=metric)
                pivot = pivot.reindex(index=stage_order, columns=stage_order)
                image = draw_heatmap(
                    ax,
                    pivot.to_numpy(dtype=float),
                    stage_names,
                    short_metric_label(metric),
                    cmap="viridis",
                    symmetric=False,
                )
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                fig.suptitle(
                    f"{short_task_label(family)} variant {variant_value_label}: {short_metric_label(metric)}",
                    y=1.02,
                )
                fig.tight_layout()
                fig.savefig(per_variant_dir / f"{sanitize(family)}_{sanitize(variant_value_label)}_raw_{sanitize(metric)}.png")
                plt.close(fig)


def save_processed_tables(output_dir: Path, rep_sources: dict[str, pd.DataFrame], control_df: pd.DataFrame,
                          families: list[str], rep_metrics: list[str], control_metric: str | None,
                          base_task_value=None) -> None:
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
                    base_task_value=base_task_value,
                )
                if not deg.empty:
                    degradation_frames.append(deg)
        if control_metric is not None and not control_df.empty and control_metric in control_df.columns:
            deg = compute_degradation(
                control_df[control_df["task_name"].isin([family, "base"])],
                control_metric,
                group_cols=["setup", "evaluator_episode"],
                base_task_value=base_task_value,
            )
            if not deg.empty:
                degradation_frames.append(deg)
    if degradation_frames:
        pd.concat(degradation_frames, ignore_index=True).to_csv(output_dir / "degradation_long.csv", index=False)


def write_manifest(output_dir: Path, *, run_root: Path, discovered: dict, stage_map: dict[int, str],
                   families: list[str], base_task_value=None) -> None:
    payload = {
        "run_root": str(run_root),
        "train_id": discovered["train_id"],
        "offline_summary": str(discovered["offline_path"]) if discovered["offline_path"] else None,
        "compound_summary": str(discovered["compound_path"]) if discovered["compound_path"] else None,
        "protocol_b_workbooks": [str(path) for path in discovered["protocol_b_workbooks"]],
        "protocol_b_belief_csvs": [str(path) for path in discovered["protocol_b_belief_csvs"]],
        "stage_map": {str(k): v for k, v in stage_map.items()},
        "families": families,
        "base_task_value": base_task_value,
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
    base_task_value = parse_base_task_value(args.base_task_value)

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
    print(f"base_task_value={base_task_value}", flush=True)

    figures_dir = output_dir / "figures"
    base_dir = figures_dir / "base_performance"
    mismatch_dir = figures_dir / "mismatch_penalty"
    training_dir = figures_dir / "training_effect"
    heatmap_dir = figures_dir / "cross_replay_heatmaps"
    sanity_dir = figures_dir / "sanity_check"
    processed_dir = output_dir / "processed"
    for path in [figures_dir, base_dir, mismatch_dir, training_dir, heatmap_dir, sanity_dir]:
        ensure_dir(path)
    ensure_dir(processed_dir)

    for family in families:
        family_base_dir = base_dir / sanitize(family)
        family_mismatch_dir = mismatch_dir / sanitize(family)
        family_training_dir = training_dir / sanitize(family)
        family_heatmap_dir = heatmap_dir / sanitize(family)
        family_sanity_dir = sanity_dir / sanitize(family)
        for path in [family_base_dir, family_mismatch_dir, family_training_dir, family_heatmap_dir, family_sanity_dir]:
            ensure_dir(path)
        plot_base_performance(
            rep_sources, control_df, stage_map, family, family_base_dir, rep_metrics, control_metric, base_task_value
        )
        plot_mismatch_penalty_curves(
            rep_sources, control_df, stage_map, family, family_mismatch_dir, rep_metrics, control_metric, base_task_value
        )
        plot_training_effect(
            rep_sources, control_df, stage_map, family, family_training_dir, rep_metrics, control_metric, base_task_value
        )
        plot_cross_replay_heatmaps(rep_sources["offline_cross"], stage_map, family, family_heatmap_dir, rep_metrics, base_task_value)
        plot_offline_vs_protocol_b_sanity(
            rep_sources, stage_map, family, family_sanity_dir, rep_metrics, base_task_value
        )

    save_processed_tables(processed_dir, rep_sources, control_df, families, rep_metrics, control_metric, base_task_value)
    write_manifest(
        processed_dir,
        run_root=run_root,
        discovered=discovered,
        stage_map=stage_map,
        families=families,
        base_task_value=base_task_value,
    )
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
    parser.add_argument("--base-task-value", type=str, default=None,
                        help="Matched baseline task_value used when files do not contain an explicit base row.")
    args = parser.parse_args()
    main(args)
