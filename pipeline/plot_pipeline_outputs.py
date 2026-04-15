import math
import re
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "plot"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_line_plot(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str) -> None:
    data = df[[x_col, y_col]].copy()
    data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna()
    if data.empty:
        return

    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data[x_col], data[y_col], marker="o")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_csv_metrics(csv_path: Path, run_root: Path, output_root: Path) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    x_candidates = [
        "train/episode",
        "agent_episode",
        "eval/agent_episode",
        "mine_optim/epoch",
        "epoch",
        "step",
        "agent_episode",
        "task_value",
    ]
    x_col = next((name for name in x_candidates if name in df.columns), None)
    if x_col is None:
        return

    numeric_cols = []
    for col in df.columns:
        if col == x_col:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            numeric_cols.append(col)

    if not numeric_cols:
        return

    rel_parent = csv_path.relative_to(run_root).parent
    out_dir = output_root / rel_parent / sanitize(csv_path.stem)
    for y_col in numeric_cols:
        out_path = out_dir / f"{sanitize(y_col)}.png"
        title = f"{csv_path.stem}: {y_col}"
        save_line_plot(df, x_col, y_col, out_path, title)


def is_decoder_checkpoint_metrics(path: Path) -> bool:
    if path.name != "metrics.csv":
        return False
    parts = path.parts
    return "decoders" in parts and any(part.startswith("ep_") for part in parts)


def plot_protocol_workbook(xlsx_path: Path, run_root: Path, output_root: Path) -> None:
    workbook = pd.ExcelFile(xlsx_path)
    if not workbook.sheet_names:
        return

    example = pd.read_excel(workbook, sheet_name=workbook.sheet_names[0])
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
            df = pd.read_excel(workbook, sheet_name=sheet)
            if metric not in df.columns or "task_value" not in df.columns:
                continue

            base_df = df[df["task_name"] == "base"] if "task_name" in df.columns else pd.DataFrame()
            var_df = df[df["task_name"] != "base"].copy() if "task_name" in df.columns else df.copy()
            if not var_df.empty and "task_value" in var_df.columns:
                var_df["task_value"] = pd.to_numeric(var_df["task_value"], errors="coerce")
                var_df[metric] = pd.to_numeric(var_df[metric], errors="coerce")
                var_df = var_df.dropna(subset=["task_value", metric]).sort_values("task_value")
                if not var_df.empty:
                    ax.plot(
                        var_df["task_value"],
                        var_df[metric],
                        marker="o",
                        label=f"Episode {sheet.split('_')[-1]}",
                    )
                    plotted = True

            if not base_df.empty:
                base_y = pd.to_numeric(base_df[metric], errors="coerce").dropna()
                if not base_y.empty:
                    if not var_df.empty:
                        min_x = float(var_df["task_value"].min())
                        max_x = float(var_df["task_value"].max())
                        gap = (max_x - min_x) * 0.08 if max_x > min_x else 0.1
                        base_x = min_x - gap
                    else:
                        base_x = 0.0
                    ax.scatter(base_x, float(base_y.iloc[0]), marker="x", s=80, color="black")
                    plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel("Task value")
        ax.set_ylabel(metric)
        ax.set_title(f"{xlsx_path.stem}: {metric}")
        ax.grid(False)
        ax.legend(loc="best")
        if "rsq" in metric.lower():
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
            plot_csv_metrics(csv_path, run_root, output_root)

    for xlsx_path in run_root.rglob("*.xlsx"):
        plot_protocol_workbook(xlsx_path, run_root, output_root)


if __name__ == "__main__":
    parser = ArgumentParser("Plot pipeline outputs from saved CSV and XLSX files.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
