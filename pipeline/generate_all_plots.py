from argparse import ArgumentParser, Namespace
from pathlib import Path

from plot_pipeline_outputs import main as plot_main


def infer_repo_root(explicit_repo_root):
    if explicit_repo_root:
        return Path(explicit_repo_root).resolve()

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent,
        script_dir,
        Path.cwd(),
    ]
    for candidate in candidates:
        if (candidate / "train.py").exists():
            return candidate.resolve()
    return Path.cwd().resolve()


def main(args):
    repo_root = infer_repo_root(args.repo_root)

    runs_root = Path(args.runs_root) if args.runs_root else (repo_root / "runs")
    if not runs_root.is_absolute():
        runs_root = (repo_root / runs_root).resolve()

    run_root = (runs_root / args.run_name).resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"Run folder not found: {run_root}")

    output_dir = Path(args.output_dir) if args.output_dir else (run_root / "plots")
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()

    plot_main(
        Namespace(
            run_root=str(run_root),
            output_dir=str(output_dir),
            protocol_a_base_value=args.protocol_a_base_value,
        )
    )
    print(f"Generated plots in: {output_dir}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser("Generate all plots for an existing pipeline run.")
    parser.add_argument("run_name", type=str, help="Run folder name under runs/")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--runs-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--protocol-a-base-value", type=float, default=None)
    args = parser.parse_args()
    main(args)
