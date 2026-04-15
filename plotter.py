import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import os


parser = ArgumentParser("File name")
parser.add_argument("name", type=str, nargs="?", default=None)
parser.add_argument("task", type=str, nargs="?", default=None)
parser.add_argument("base_x", type=float, nargs="?", default=-0.02)
args = parser.parse_args()
excel_path = r"D:\\Personal folder\\University\\Projects\\4th year\\belief-rnn\\report\\"
excel_path = os.path.join(excel_path, f"{args.name}.xlsx")
# Load workbook
xls = pd.ExcelFile(excel_path)

# -----------------------------------
# Discover metrics automatically
# -----------------------------------
example_df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

# Assume metric columns_toggle_task_N are all except identifiers
id_cols = {"variant", "task_name", "task_value"}
metric_cols = [c for c in example_df.columns if c not in id_cols]

print("Detected metrics:", metric_cols)

# -----------------------------------
# Create one plot per metric
# -----------------------------------
for metric in metric_cols:

    plt.figure(figsize=(8,5))

    for sheet in xls.sheet_names:

        df = pd.read_excel(xls, sheet_name=sheet)

        # ---- Split base vs variant ----
        base_df = df[df["task_name"] == "base"]
        var_df = df[df["task_name"] != "base"].copy()

        # Sort variants by x value
        var_df = var_df.sort_values("task_value")

        # ---------------------------
        # Plot line for variants
        # ---------------------------
        if not var_df.empty:
            print("Plotting variants for sheet:", sheet)
            print(var_df["task_value"], var_df[metric])
            plt.plot(
                var_df["task_value"],
                var_df[metric],
                marker="o",
                label=f"Episode {sheet.split('_')[-1]}"
            )

        # ---------------------------
        # Plot base as isolated point
        # ---------------------------
        if not base_df.empty:
            base_y = base_df[metric].values[0]

            print("Protocol A")

            # place base at NaN-safe x position slightly left of first point
            if not var_df.empty:
                base_x = args.base_x
            else:
                base_x = 0
            plt.scatter(
                base_x,
                base_y,
                marker="x",
                s=100,
            )

    # ---- formatting ----
    plt.xlabel("Task value")
    plt.ylabel(metric)
    plt.xticks(var_df["task_value"])
    plt.title(f"{metric} for {args.task}={args.base_x}")
    plt.legend()
    plt.grid(False)
    if metric == "linreg_rsq-0":
        plt.ylim(top= 1, bottom=0)

    plt.tight_layout()
    plt_path = os.path.join(excel_path.replace(".xlsx", f"_{metric}.png"))
    plt.savefig(plt_path)

    plt.show()

