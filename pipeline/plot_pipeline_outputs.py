import re
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METRIC_LABELS = {
    "MI": "Mutual information",
    "R2_b0": "Belief decoder R^2",
    "KL_b0": "Belief decoder KL divergence",
    "CE_b0": "Belief decoder cross-entropy",
    "H_true_b0": "True belief entropy",
    "H_pred_b0": "Decoded belief entropy",
    "JS_b0": "Jensen-Shannon divergence",
    "entropy_b0": "Belief entropy estimate",
    "normalized_CE_b0": "Normalized cross-entropy",
    "normalized_MI": "Normalized mutual information",
    "MI_normalized_by_mlp_entropy": "Normalized mutual information",
    "train/disc_return": "Discounted return",
    "train/return": "Return",
    "train/num_transitions": "Environment steps",
    "train/episode": "Training episode",
    "agent_episode": "Agent episode",
    "eval/agent_episode": "Agent episode",
    "task_value": "Task value",
    "epoch": "Epoch",
    "step": "Step",
    "metric_1_drqn_mean_return": "DRQN mean return",
    "metric_1_planner_mean_return": "Belief planner mean return",
    "metric_1_return_gap_planner_minus_drqn": "Gap in return between belief and DRQN planner",
    "metric_1_drqn_mean_disc_return": "DRQN mean discounted return",
    "metric_1_planner_mean_disc_return": "Belief planner mean discounted return",
    "metric_1_disc_return_gap_planner_minus_drqn": "Gap in discounted return between belief and DRQN planner",
    "metric_2_step_weighted_action_agreement_rate": "Step-weighted action agreement between belief and DRQN planner",
    "metric_3_step_weighted_mean_regret": "Step-weighted mean regret of DRQN relative to belief planner",
    "metric_3_step_weighted_mean_discounted_regret": "Step-weighted mean discounted regret of DRQN relative to belief planner",
    "metric_2_step_weighted_q_mse": "Step-weighted mean Q-value MSE between DRQN and planner",
    "metric_2_step_weighted_q_mae": "Step-weighted mean Q-value MAE between DRQN and planner",
    "metric_2_step_weighted_q_chosen_action_mse": "Step-weighted mean chosen-action Q-value MSE",
    "comparison_mean_episode_regret": "Mean episode regret of DRQN relative to belief planner",
    "comparison_mean_discounted_episode_regret": "Mean discounted episode regret of DRQN relative to belief planner",
    "comparison_rollout_mean_drqn_return": "DRQN rollout mean return",
    "comparison_rollout_mean_drqn_disc_return": "DRQN rollout mean discounted return",
}

NON_PLOTTED_COLUMNS = {
    "run_id",
    "environment",
    "planner_horizon",
    "belief_round_ndigits",
    "total_steps_budget",
}

DERIVED_METRIC_EPSILON = 1e-8


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "plot"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_token(text: str) -> str:
    text = text.replace("/", " ")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    words = []
    for word in text.split():
        lowered = word.lower()
        if lowered == "drqn":
            words.append("DRQN")
        elif lowered == "mi":
            words.append("MI")
        elif lowered == "nmi":
            words.append("NMI")
        elif lowered == "kl":
            words.append("KL")
        elif lowered == "ce":
            words.append("CE")
        elif lowered == "nce":
            words.append("NCE")
        elif lowered == "r2":
            words.append("R^2")
        elif lowered == "js":
            words.append("JS")
        elif lowered == "mlp":
            words.append("MLP")
        else:
            words.append(word.capitalize())
    return " ".join(words)


def metric_label(name: str) -> str:
    if name in METRIC_LABELS:
        return METRIC_LABELS[name]

    cleaned = re.sub(r"^metric_\d+_", "", name)
    cleaned = re.sub(r"^comparison_", "", cleaned)
    cleaned = cleaned.replace("H_true", "true_belief_entropy")
    cleaned = cleaned.replace("H_pred", "decoded_belief_entropy")
    cleaned = re.sub(r"(?<![A-Za-z0-9])JS(?![A-Za-z0-9])", "jensen_shannon", cleaned)
    return normalize_token(cleaned)


def coerce_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    text = series.astype(str).str.strip()
    text = text.str.replace(
        r"^tensor\(([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\)$",
        r"\1",
        regex=True,
    )
    valid_numeric = text.str.match(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")
    text = text.where(valid_numeric)
    return pd.to_numeric(text, errors="coerce")


METRIC_TOKEN_PATTERNS = {
    "KL": re.compile(r"(?<![A-Za-z0-9])KL(?![A-Za-z0-9])"),
    "CE": re.compile(r"(?<![A-Za-z0-9])CE(?![A-Za-z0-9])"),
    "kl": re.compile(r"(?<![A-Za-z0-9])kl(?![A-Za-z0-9])"),
    "ce": re.compile(r"(?<![A-Za-z0-9])ce(?![A-Za-z0-9])"),
}


def replace_metric_token(name: str, old_token: str, new_token: str) -> str | None:
    pattern = METRIC_TOKEN_PATTERNS.get(old_token)
    if pattern is None or not pattern.search(name):
        return None
    return pattern.sub(new_token, name, count=1)


def candidate_ce_columns(kl_col: str) -> list[str]:
    candidates = []
    for old, new in (("KL", "CE"), ("kl", "ce")):
        candidate = replace_metric_token(kl_col, old, new)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def metric_slot_key(column_name: str, *, tokens_to_remove: tuple[str, ...]) -> str:
    key = column_name.lower()
    for token in tokens_to_remove:
        key = re.sub(rf"(?<![a-z0-9]){re.escape(token.lower())}(?![a-z0-9])", "", key)
    key = re.sub(r"(?<![a-z0-9])(linear|mlp)(?![a-z0-9])", "", key)
    key = re.sub(r"[_/.-]+", "_", key).strip("_")
    return key or "default"


def entropy_slot_key(column_name: str) -> str:
    return metric_slot_key(column_name, tokens_to_remove=("KL", "CE", "kl", "ce"))


def true_entropy_slot_key(column_name: str) -> str:
    return metric_slot_key(column_name, tokens_to_remove=("h_true", "true_entropy"))


def metric_variant(column_name: str) -> str:
    lowered = column_name.lower()
    if "linear" in lowered:
        return "linear"
    if "mlp" in lowered:
        return "mlp"
    return "legacy"


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = coerce_numeric_series(numerator)
    denominator = coerce_numeric_series(denominator)
    out = numerator / denominator
    bad = denominator.abs() <= DERIVED_METRIC_EPSILON
    return out.mask(bad)


def add_derived_metric(df: pd.DataFrame, column_name: str, series: pd.Series) -> None:
    if column_name in df.columns:
        return
    df[column_name] = series


def report_entropy_differences(
    df: pd.DataFrame,
    entropy_by_variant: dict[str, dict[str, str]],
    source_name: str,
) -> None:
    linear = entropy_by_variant.get("linear", {})
    mlp = entropy_by_variant.get("mlp", {})
    shared_slots = sorted(set(linear) & set(mlp))
    for slot in shared_slots:
        linear_series = coerce_numeric_series(df[linear[slot]])
        mlp_series = coerce_numeric_series(df[mlp[slot]])
        diff = (linear_series - mlp_series).abs().dropna()
        if diff.empty:
            continue
        print(
            f"[derived-metrics] {source_name}: entropy difference "
            f"(linear vs mlp, slot={slot}) mean={diff.mean():.6g}, max={diff.max():.6g}",
            flush=True,
        )


def add_normalized_mi_columns(
    df: pd.DataFrame,
    mi_columns: list[str],
    entropy_by_variant: dict[str, dict[str, str]],
) -> None:
    mlp_entropy = entropy_by_variant.get("mlp", {})
    if not mi_columns or not mlp_entropy:
        return

    preferred_slots = [slot for slot in mlp_entropy if "train" not in slot]
    if not preferred_slots:
        preferred_slots = list(mlp_entropy.keys())

    for mi_col in mi_columns:
        if len(preferred_slots) == 1:
            slot = preferred_slots[0]
            entropy_col = mlp_entropy[slot]
            normalized_name = (
                "normalized_MI"
                if mi_col == "MI"
                else f"{mi_col}_normalized_by_mlp_entropy"
            )
            if mi_col == "MI":
                alias_name = "MI_normalized_by_mlp_entropy"
            else:
                alias_name = None
            normalized = safe_divide(df[mi_col], df[entropy_col])
            add_derived_metric(df, normalized_name, normalized)
            if alias_name is not None:
                add_derived_metric(df, alias_name, normalized)
            continue

        for slot in preferred_slots:
            entropy_col = mlp_entropy[slot]
            normalized_name = f"{mi_col}_normalized_by_mlp_entropy_{slot}"
            add_derived_metric(df, normalized_name, safe_divide(df[mi_col], df[entropy_col]))


def explicit_true_entropy_columns(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    entropy_by_variant: dict[str, dict[str, str]] = {
        "linear": {},
        "mlp": {},
        "legacy": {},
    }
    for col in df.columns:
        lowered = col.lower()
        if "h_true" not in lowered and "true_entropy" not in lowered:
            continue
        entropy_by_variant[metric_variant(col)][true_entropy_slot_key(col)] = col
    return entropy_by_variant


def fallback_entropy_column_name(kl_col: str) -> str | None:
    return (
        replace_metric_token(kl_col, "KL", "entropy")
        or replace_metric_token(kl_col, "kl", "entropy")
    )


def normalized_ce_column_name(ce_col: str) -> str | None:
    return (
        replace_metric_token(ce_col, "CE", "normalized_CE")
        or replace_metric_token(ce_col, "ce", "normalized_ce")
    )


def augment_derived_metrics(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()
    entropy_by_variant = explicit_true_entropy_columns(df)
    mi_columns = [col for col in df.columns if col in {"MI", "mi"}]

    for kl_col in list(df.columns):
        ce_col = next((candidate for candidate in candidate_ce_columns(kl_col) if candidate in df.columns), None)
        if ce_col is None:
            continue

        variant = metric_variant(kl_col)
        slot = entropy_slot_key(kl_col)
        entropy_col = entropy_by_variant.get(variant, {}).get(slot)
        normalized_ce_col = normalized_ce_column_name(ce_col)
        if normalized_ce_col is None:
            continue

        if entropy_col is None:
            entropy_col = fallback_entropy_column_name(kl_col)
            if entropy_col is None:
                continue
            entropy = coerce_numeric_series(df[ce_col]) - coerce_numeric_series(df[kl_col])
            entropy = entropy.clip(lower=DERIVED_METRIC_EPSILON)
            add_derived_metric(df, entropy_col, entropy)
            entropy_by_variant[variant].setdefault(slot, entropy_col)

        add_derived_metric(df, normalized_ce_col, safe_divide(df[ce_col], df[entropy_col]))

    report_entropy_differences(df, entropy_by_variant, source_name)
    add_normalized_mi_columns(df, mi_columns, entropy_by_variant)
    return df


def csv_context(csv_path: Path, run_root: Path) -> dict:
    rel = csv_path.relative_to(run_root)
    parts = rel.parts
    context = {
        "is_train_metrics": csv_path.name.startswith("train_metrics_"),
        "is_summary_table": csv_path.name.endswith("_summary_table.csv"),
        "is_decoder_summary": csv_path.name == "decoder_episode_summary.csv",
        "is_belief_eval": "belief_eval" in csv_path.stem,
        "sweep_label": None,
        "protocol_label": None,
    }
    if "protocol_a" in parts:
        idx = parts.index("protocol_a")
        context["protocol_label"] = "Protocol A"
        if idx + 1 < len(parts):
            context["sweep_label"] = normalize_token(parts[idx + 1])
    elif "protocol_b" in parts:
        idx = parts.index("protocol_b")
        context["protocol_label"] = "Protocol B"
        if idx + 1 < len(parts):
            context["sweep_label"] = normalize_token(parts[idx + 1])
    return context


def workbook_context(xlsx_path: Path, run_root: Path) -> dict:
    rel = xlsx_path.relative_to(run_root)
    parts = rel.parts
    context = {
        "protocol_label": None,
        "sweep_label": None,
    }
    if "protocol_a" in parts:
        idx = parts.index("protocol_a")
        context["protocol_label"] = "Protocol A"
        if idx + 1 < len(parts):
            context["sweep_label"] = normalize_token(parts[idx + 1])
    elif "protocol_b" in parts:
        idx = parts.index("protocol_b")
        context["protocol_label"] = "Protocol B"
        if idx + 1 < len(parts):
            context["sweep_label"] = normalize_token(parts[idx + 1])
    return context


def axis_label(name: str, context: dict | None = None) -> str:
    if name == "task_value" and context and context.get("sweep_label"):
        return context["sweep_label"]
    return metric_label(name)


def csv_plot_title(csv_path: Path, y_col: str, context: dict) -> str:
    if context["is_train_metrics"]:
        return "Discounted return during training"
    metric_name = metric_label(y_col)
    protocol_label = context.get("protocol_label")
    sweep_label = context.get("sweep_label")
    if context.get("is_belief_eval") and protocol_label and sweep_label:
        return f"{protocol_label} {sweep_label} belief comparison: {metric_name}"
    if context.get("is_belief_eval") and protocol_label:
        return f"{protocol_label} belief comparison: {metric_name}"
    if context.get("is_summary_table") and protocol_label and sweep_label:
        return f"{protocol_label} {sweep_label} sweep: {metric_name}"
    if context.get("is_summary_table") and protocol_label:
        return f"{protocol_label}: {metric_name}"
    return metric_name


def workbook_plot_title(metric: str, context: dict) -> str:
    metric_name = metric_label(metric)
    protocol_label = context.get("protocol_label")
    sweep_label = context.get("sweep_label")
    if protocol_label and sweep_label:
        return f"{protocol_label} {sweep_label} sweep: {metric_name}"
    if protocol_label:
        return f"{protocol_label}: {metric_name}"
    return metric_name


def save_line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: Path,
    title: str,
    *,
    x_label: str | None = None,
    y_label: str | None = None,
) -> None:
    data = df[[x_col, y_col]].copy()
    data[x_col] = coerce_numeric_series(data[x_col])
    data[y_col] = coerce_numeric_series(data[y_col])
    data = data.dropna()
    if data.empty:
        return

    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data[x_col], data[y_col], marker="o")
    ax.set_xlabel(x_label or axis_label(x_col))
    ax.set_ylabel(y_label or axis_label(y_col))
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_train_metrics(
    csv_path: Path,
    run_root: Path,
    output_root: Path,
    *,
    train_x_axis: str,
) -> None:
    df = pd.read_csv(csv_path)
    if df.empty or "train/disc_return" not in df.columns:
        return

    x_col = "train/episode" if train_x_axis == "episode" else "train/num_transitions"
    if x_col not in df.columns:
        return

    rel_parent = csv_path.relative_to(run_root).parent
    out_dir = output_root / rel_parent / sanitize(csv_path.stem)
    suffix = "epochs" if train_x_axis == "episode" else "steps"
    out_path = out_dir / f"discounted_return_vs_{suffix}.png"
    save_line_plot(
        df,
        x_col,
        "train/disc_return",
        out_path,
        "Discounted return during training",
        x_label="Training episode" if train_x_axis == "episode" else "Environment steps",
        y_label="Discounted return",
    )


def plot_csv_metrics(
    csv_path: Path,
    run_root: Path,
    output_root: Path,
    *,
    train_x_axis: str,
) -> None:
    context = csv_context(csv_path, run_root)
    if context["is_train_metrics"]:
        plot_train_metrics(
            csv_path,
            run_root,
            output_root,
            train_x_axis=train_x_axis,
        )
        return

    df = augment_derived_metrics(pd.read_csv(csv_path), str(csv_path.relative_to(run_root)))
    if df.empty:
        return

    x_candidates = [
        "agent_episode",
        "eval/agent_episode",
        "mine_optim/epoch",
        "epoch",
        "step",
        "task_value",
    ]
    x_col = next((name for name in x_candidates if name in df.columns), None)
    if x_col is None:
        return

    numeric_cols = []
    for col in df.columns:
        if col == x_col or col in NON_PLOTTED_COLUMNS:
            continue
        series = coerce_numeric_series(df[col])
        if series.notna().any():
            numeric_cols.append(col)

    if not numeric_cols:
        return

    rel_parent = csv_path.relative_to(run_root).parent
    out_dir = output_root / rel_parent / sanitize(csv_path.stem)
    for y_col in numeric_cols:
        out_path = out_dir / f"{sanitize(y_col)}.png"
        save_line_plot(
            df,
            x_col,
            y_col,
            out_path,
            csv_plot_title(csv_path, y_col, context),
            x_label=axis_label(x_col, context),
            y_label=axis_label(y_col, context),
        )


def is_decoder_checkpoint_metrics(path: Path) -> bool:
    if path.name != "metrics.csv":
        return False
    parts = path.parts
    return "decoders" in parts and any(part.startswith("ep_") for part in parts)


def plot_protocol_workbook(
    xlsx_path: Path,
    run_root: Path,
    output_root: Path,
    *,
    protocol_a_base_value: float | None = None,
) -> None:
    workbook = pd.ExcelFile(xlsx_path)
    if not workbook.sheet_names:
        return

    context = workbook_context(xlsx_path, run_root)
    workbook_rel = str(xlsx_path.relative_to(run_root))
    sheets = {
        sheet: augment_derived_metrics(
            pd.read_excel(workbook, sheet_name=sheet),
            f"{workbook_rel}::{sheet}",
        )
        for sheet in workbook.sheet_names
    }
    example = sheets[workbook.sheet_names[0]]
    id_cols = {"variant", "task_name", "task_value"}
    metric_cols = [c for c in example.columns if c not in id_cols]
    if not metric_cols:
        return

    rel_parent = xlsx_path.relative_to(run_root).parent
    out_dir = output_root / rel_parent / sanitize(xlsx_path.stem)
    ensure_dir(out_dir)

    for metric in metric_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False

        for sheet in workbook.sheet_names:
            df = sheets[sheet]
            if metric not in df.columns or "task_value" not in df.columns:
                continue

            base_df = df[df["task_name"] == "base"] if "task_name" in df.columns else pd.DataFrame()
            var_df = df[df["task_name"] != "base"].copy() if "task_name" in df.columns else df.copy()
            sheet_color = None
            if not var_df.empty and "task_value" in var_df.columns:
                var_df["task_value"] = coerce_numeric_series(var_df["task_value"])
                var_df[metric] = coerce_numeric_series(var_df[metric])
                var_df = var_df.dropna(subset=["task_value", metric]).sort_values("task_value")
                if not var_df.empty:
                    line = ax.plot(
                        var_df["task_value"],
                        var_df[metric],
                        marker="o",
                        label=f"Episode {sheet.split('_')[-1]}",
                    )
                    try:
                        sheet_color = line[0].get_color()
                    except Exception:
                        sheet_color = None
                    plotted = True

            if not base_df.empty:
                base_y = coerce_numeric_series(base_df[metric]).dropna()
                if not base_y.empty:
                    if context.get("protocol_label") == "Protocol A" and protocol_a_base_value is not None:
                        base_x = protocol_a_base_value
                    elif not var_df.empty:
                        min_x = float(var_df["task_value"].min())
                        max_x = float(var_df["task_value"].max())
                        gap = (max_x - min_x) * 0.08 if max_x > min_x else 0.1
                        base_x = min_x - gap
                    else:
                        base_x = 0.0
                    color = sheet_color
                    if color is None:
                        color = "black"
                    ax.scatter(base_x, float(base_y.iloc[0]), marker="x", s=80, color=color)
                    plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel(axis_label("task_value", context))
        ax.set_ylabel(metric_label(metric))
        ax.set_title(workbook_plot_title(metric, context))
        ax.grid(False)
        ax.legend(loc="best")
        if "r^2" in metric_label(metric).lower():
            ax.set_ylim(bottom=0.0, top=1.0)

        fig.tight_layout()
        fig.savefig(out_dir / f"{sanitize(metric)}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def should_plot_csv(path: Path) -> bool:
    if path.name == "decoder_episode_summary.csv":
        return True
    if is_decoder_checkpoint_metrics(path):
        return False
    if path.name == "metrics.csv":
        return True
    if path.name.startswith("train_metrics_"):
        return True
    if path.name.startswith("mine_optim_"):
        return True
    if path.name.startswith("mine_train_"):
        return True
    if path.name.startswith("mine_eval_"):
        return True
    if path.name.endswith("_summary_table.csv"):
        return True
    return False


def main(args):
    run_root = Path(args.run_root).resolve()
    output_root = Path(args.output_dir).resolve()
    ensure_dir(output_root)

    for csv_path in run_root.rglob("*.csv"):
        if should_plot_csv(csv_path):
            plot_csv_metrics(
                csv_path,
                run_root,
                output_root,
                train_x_axis=args.train_x_axis,
            )

    for xlsx_path in run_root.rglob("*.xlsx"):
        plot_protocol_workbook(
            xlsx_path,
            run_root,
            output_root,
            protocol_a_base_value=args.protocol_a_base_value,
        )


if __name__ == "__main__":
    parser = ArgumentParser("Plot pipeline outputs from saved CSV and XLSX files.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--protocol-a-base-value", type=float, default=None)
    parser.add_argument("--train-x-axis", choices=["episode", "steps"], default="steps")
    args = parser.parse_args()
    main(args)
