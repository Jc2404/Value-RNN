import json
import os
import shlex
import subprocess
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path


def sanitize(name: str) -> str:
    out = []
    for char in name:
        if char.isalnum() or char in ("-", "_", "."):
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("_") or "run"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def stringify_args(values):
    return [str(v) for v in values]


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


def resolve_script_path(script: str, config_dir: Path, repo_root: Path) -> str:
    script_path = Path(script)
    if script_path.is_absolute():
        return str(script_path)

    config_relative = (config_dir / script_path).resolve()
    if config_relative.exists():
        return str(config_relative)

    repo_relative = (repo_root / script_path).resolve()
    if repo_relative.exists():
        return str(repo_relative)

    return str(repo_relative)


def stage_enabled(cfg: dict) -> bool:
    return bool(cfg.get("enabled", False))


def command_string(cmd):
    return " ".join(shlex.quote(part) for part in cmd)


def build_subprocess_env(repo_root: Path):
    env = os.environ.copy()
    repo_root_str = str(repo_root)
    existing = env.get("PYTHONPATH")
    if existing:
        if repo_root_str not in existing.split(os.pathsep):
            env["PYTHONPATH"] = os.pathsep.join([repo_root_str, existing])
    else:
        env["PYTHONPATH"] = repo_root_str
    return env


def run_command(cmd, cwd: Path, log_path: Path, dry_run: bool = False) -> None:
    ensure_dir(log_path.parent)
    if dry_run:
        print(f"[dry-run] {command_string(cmd)}", flush=True)
        return

    print(f"[run] {command_string(cmd)}", flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=build_subprocess_env(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    with open(log_path, "w", encoding="utf-8") as log_file:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

    code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def build_python_command(script: str, script_args):
    return [
        sys.executable,
        script,
        *stringify_args(script_args),
    ]


def normalize_tests(tests_cfg):
    tests = []
    for item in tests_cfg:
        if isinstance(item, str):
            tests.append({
                "label": item.lstrip("-"),
                "flags": [item],
                "args": [],
            })
        elif isinstance(item, dict):
            flags = []
            if "flag" in item:
                flags.append(item["flag"])
            flags.extend(item.get("flags", []))
            tests.append({
                "label": item.get("label") or (flags[0].lstrip("-") if flags else "test"),
                "flags": flags,
                "args": item.get("args", []),
            })
        else:
            raise ValueError(f"Unsupported test configuration: {item!r}")
    return tests


def main(args):
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    repo_root = infer_repo_root(args.repo_root)
    config = load_json(config_path)

    run_name = config.get("run_name", "full_pipeline")
    run_slug = config.get("run_slug")
    if not run_slug:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_slug = f"{timestamp}_{sanitize(run_name)}"

    output_root = Path(config.get("output_root", "runs"))
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()

    run_root = Path(config.get("run_root", output_root / run_slug))
    if not run_root.is_absolute():
        run_root = (repo_root / run_root).resolve()

    weights_dir = (run_root / "weights").resolve()
    results_dir = (run_root / "results").resolve()
    report_dir = (run_root / "report").resolve()
    plots_dir = (run_root / "plots").resolve()
    logs_dir = (run_root / "logs").resolve()
    agent_dir = (run_root / "agent").resolve()

    for path in [run_root, weights_dir, results_dir, report_dir, plots_dir, logs_dir, agent_dir]:
        ensure_dir(path)

    manifest = {
        "run_name": run_name,
        "run_slug": run_slug,
        "repo_root": str(repo_root),
        "run_root": str(run_root),
        "weights_dir": str(weights_dir),
        "results_dir": str(results_dir),
        "report_dir": str(report_dir),
        "plots_dir": str(plots_dir),
        "logs_dir": str(logs_dir),
        "commands": [],
        "config": config,
    }
    write_json(run_root / "resolved_config.json", manifest)

    train_id = config.get("existing_train_id")

    train_cfg = config.get("train", {})
    if stage_enabled(train_cfg):
        train_name = train_cfg.get("name", run_name)
        train_script = resolve_script_path(train_cfg.get("script", "train.py"), config_dir, repo_root)
        train_args = [
            "--name",
            train_name,
            "--weights-dir",
            str(weights_dir),
            "--results-dir",
            str(agent_dir),
            *stringify_args(train_cfg.get("args", [])),
        ]
        train_cmd = build_python_command(train_script, train_args)
        manifest["commands"].append({"stage": "train", "command": train_cmd})
        write_json(run_root / "resolved_config.json", manifest)
        run_command(train_cmd, repo_root, logs_dir / "train.log", dry_run=args.dry_run)

        if args.dry_run:
            train_id = train_cfg.get("dry_run_train_id", "DRY_RUN_TRAIN_ID")
            manifest["train_run_info"] = {
                "run_id": train_id,
                "run_name": train_name,
                "weights_dir": str(weights_dir),
                "results_dir": str(agent_dir),
                "args": train_cfg.get("args", []),
            }
            write_json(run_root / "resolved_config.json", manifest)
        else:
            run_info = load_json(agent_dir / "train_run_info.json")
            train_id = run_info["run_id"]
            manifest["train_run_info"] = run_info
            write_json(run_root / "resolved_config.json", manifest)

    if not train_id:
        raise ValueError("No train_id available. Enable the train stage or provide existing_train_id in the config.")

    decoder_cfg = config.get("decoder_train", {})
    if stage_enabled(decoder_cfg):
        decoder_name = decoder_cfg.get("name", "decoder_train")
        decoder_script = resolve_script_path(decoder_cfg.get("script", "train_decoder.py"), config_dir, repo_root)
        decoder_args = [
            decoder_name,
            train_id,
            "--weights_dir",
            str(weights_dir),
        ]
        decoder_subdir = decoder_cfg.get("decoder_subdir")
        if decoder_subdir:
            decoder_args.extend(["--decoder_subdir", str(decoder_subdir)])
        decoder_args.extend(stringify_args(decoder_cfg.get("args", [])))
        decoder_cmd = build_python_command(decoder_script, decoder_args)
        manifest["commands"].append({"stage": "decoder_train", "command": decoder_cmd})
        write_json(run_root / "resolved_config.json", manifest)
        run_command(decoder_cmd, repo_root, logs_dir / "decoder_train.log", dry_run=args.dry_run)

    mi_cfg = config.get("mi_train", {})
    mine_id = None
    if stage_enabled(mi_cfg):
        mi_name = mi_cfg.get("name", "mine_train")
        mine_id = mi_cfg.get("mine_id", f"{run_slug}_mine")
        mi_script = resolve_script_path(mi_cfg.get("script", "probe_mi/train_mine.py"), config_dir, repo_root)
        mi_results_dir = (results_dir / "mi").resolve()
        ensure_dir(mi_results_dir)
        mi_args = [
            train_id,
            "--name",
            mi_name,
            "--mine_id",
            mine_id,
            "--weights_dir",
            str(weights_dir),
            "--results_dir",
            str(mi_results_dir),
            *stringify_args(mi_cfg.get("args", [])),
        ]
        mi_cmd = build_python_command(mi_script, mi_args)
        manifest["commands"].append({"stage": "mi_train", "command": mi_cmd, "mine_id": mine_id})
        write_json(run_root / "resolved_config.json", manifest)
        run_command(mi_cmd, repo_root, logs_dir / "mi_train.log", dry_run=args.dry_run)
        manifest["mine_id"] = mine_id
        write_json(run_root / "resolved_config.json", manifest)

    protocol_a_cfg = config.get("protocol_a", {})
    if stage_enabled(protocol_a_cfg):
        tests = normalize_tests(protocol_a_cfg.get("tests", []))
        if not tests:
            raise ValueError("protocol_a.enabled is true but no tests were provided.")
        common_args = stringify_args(protocol_a_cfg.get("args", []))
        script = resolve_script_path(protocol_a_cfg.get("script", "fix_decode_eval.py"), config_dir, repo_root)
        base_name = protocol_a_cfg.get("name", "protocolA")
        for test in tests:
            label = sanitize(test["label"])
            test_report_dir = (report_dir / "protocol_a" / label).resolve()
            ensure_dir(test_report_dir)
            stage_name = f"{base_name}_{label}"
            cmd_args = [
                stage_name,
                train_id,
                "--weights_dir",
                str(weights_dir),
                "--report_dir",
                str(test_report_dir),
                *common_args,
                *stringify_args(test["args"]),
                *stringify_args(test["flags"]),
            ]
            cmd = build_python_command(script, cmd_args)
            manifest["commands"].append({"stage": f"protocol_a:{label}", "command": cmd})
            write_json(run_root / "resolved_config.json", manifest)
            run_command(cmd, repo_root, logs_dir / f"protocol_a_{label}.log", dry_run=args.dry_run)

    protocol_b_cfg = config.get("protocol_b", {})
    if stage_enabled(protocol_b_cfg):
        tests = normalize_tests(protocol_b_cfg.get("tests", []))
        if not tests:
            raise ValueError("protocol_b.enabled is true but no tests were provided.")
        common_args = stringify_args(protocol_b_cfg.get("args", []))
        script = resolve_script_path(protocol_b_cfg.get("script", "retrain_decode_eval.py"), config_dir, repo_root)
        base_name = protocol_b_cfg.get("name", "protocolB")
        for test in tests:
            label = sanitize(test["label"])
            test_report_dir = (report_dir / "protocol_b" / label).resolve()
            ensure_dir(test_report_dir)
            stage_name = f"{base_name}_{label}"
            cmd_args = [
                stage_name,
                train_id,
                "--weights_dir",
                str(weights_dir),
                "--report_dir",
                str(test_report_dir),
                *common_args,
                *stringify_args(test["args"]),
                *stringify_args(test["flags"]),
            ]
            cmd = build_python_command(script, cmd_args)
            manifest["commands"].append({"stage": f"protocol_b:{label}", "command": cmd})
            write_json(run_root / "resolved_config.json", manifest)
            run_command(cmd, repo_root, logs_dir / f"protocol_b_{label}.log", dry_run=args.dry_run)

    belief_eval_cfg = config.get("belief_eval", {})
    if stage_enabled(belief_eval_cfg):
        script = resolve_script_path(belief_eval_cfg.get("script", "eval_drqn_vs_belief.py"), config_dir, repo_root)
        out_dir = (results_dir / "drqn_vs_belief").resolve()
        ensure_dir(out_dir)
        cmd_args = [
            "--run-id",
            train_id,
            "--weights-dir",
            str(weights_dir),
            "--output-dir",
            str(out_dir),
            *stringify_args(belief_eval_cfg.get("args", [])),
        ]
        cmd = build_python_command(script, cmd_args)
        manifest["commands"].append({"stage": "belief_eval", "command": cmd})
        write_json(run_root / "resolved_config.json", manifest)
        run_command(cmd, repo_root, logs_dir / "belief_eval.log", dry_run=args.dry_run)

    manifest["train_id"] = train_id
    if mine_id is not None:
        manifest["mine_id"] = mine_id
    write_json(run_root / "resolved_config.json", manifest)
    print(f"Run root: {run_root}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser("Run the full agent/decoder/evaluation pipeline from a JSON config.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args)
