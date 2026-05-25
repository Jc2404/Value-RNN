import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

from offline_trajectory_utils import (
    add_offline_analysis_flags,
    add_variant_test_flags,
    ordered_unique_ints,
)


def build_parser():
    parser = ArgumentParser(
        "Run offline trajectory caching and replay evaluation with separate generator and evaluator checkpoint lists."
    )
    parser.add_argument("train_id", type=str)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument(
        "--sample_checkpoint_episodes",
        type=int,
        nargs="+",
        required=True,
        help="Checkpoint episodes used to generate cached trajectories (m in m x n x e).",
    )
    parser.add_argument(
        "--result_checkpoint_episodes",
        type=int,
        nargs="+",
        required=True,
        help="Checkpoint episodes used to replay trajectories and fit decoders (n in m x n x e).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Target number of cached decision timesteps per generator checkpoint / variant.",
    )
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--approximate", action="store_true")
    parser.add_argument("--sample_wandb_project", type=str, default="offline-trajectory-cache")
    parser.add_argument("--replay_wandb_project", type=str, default="offline-replay-decode")
    parser.add_argument("--run_compound_online", action="store_true")
    add_variant_test_flags(parser)
    add_offline_analysis_flags(parser)
    return parser


def _append_common_positionals(cmd, args):
    cmd.append(args.train_id)
    cmd.extend(["--name", args.name])


def _append_variant_flags(cmd, args):
    for flag in [
        "test_length",
        "test_stochasticity",
        "test_variations",
        "test_p_omission",
        "test_bin_size",
        "test_iti_hazard",
        "test_iti_min",
        "test_nITI_microstates",
        "test_listen_accuracy",
        "test_reward_listen",
        "test_grid_size",
        "test_tprob",
        "test_reward_scheme",
        "test_reward_margin",
        "test_p_cry_if_hungry",
        "test_p_cry_if_full",
    ]:
        if getattr(args, flag):
            cmd.append(f"--{flag}")


def _append_analysis_flags(cmd, args):
    for flag in [
        "run_mi",
        "run_regression",
        "run_softmax_linear_probe",
        "run_softmax_mlp_probe",
    ]:
        if getattr(args, flag):
            cmd.append(f"--{flag}")

    if args.no_float64:
        cmd.append("--no-float64")

    if not args.standardize:
        cmd.append("--no-standardize")

    for key in [
        "valid_size",
        "probe_epochs",
        "probe_lr",
        "probe_batch_size",
        "mlp_hidden_dim",
        "mlp_dropout",
        "belief_loss",
        "mine_num_layers",
        "mine_hidden_size",
        "mine_alpha",
        "mine_num_epochs",
        "mine_batch_size",
        "mine_learning_rate",
        "mine_lambda",
        "representation_size",
    ]:
        cmd.extend([f"--{key}", str(getattr(args, key))])

    if args.belief_part is not None:
        cmd.extend(["--belief_part", str(args.belief_part)])


def run_command(cmd, repo_root: Path):
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def main(args):
    sample_episodes = ordered_unique_ints(args.sample_checkpoint_episodes)
    result_episodes = ordered_unique_ints(args.result_checkpoint_episodes)
    repo_root = Path(__file__).resolve().parent

    sample_script = repo_root / "sample_offline_trajectories.py"
    replay_script = repo_root / "offline_replay_decode_eval.py"

    sample_cmd = [sys.executable, str(sample_script)]
    _append_common_positionals(sample_cmd, args)
    sample_cmd.extend(["--wandb_project", args.sample_wandb_project])
    sample_cmd.extend(["--weights_dir", args.weights_dir])
    sample_cmd.extend(["--checkpoint_episodes", *[str(ep) for ep in sample_episodes]])
    sample_cmd.extend(["--num_samples", str(args.num_samples)])
    sample_cmd.extend(["--epsilon", str(args.epsilon)])
    _append_variant_flags(sample_cmd, args)

    replay_cmd = [sys.executable, str(replay_script)]
    _append_common_positionals(replay_cmd, args)
    replay_cmd.extend(["--wandb_project", args.replay_wandb_project])
    replay_cmd.extend(["--weights_dir", args.weights_dir])
    replay_cmd.extend(["--checkpoint_episodes", *[str(ep) for ep in result_episodes]])
    replay_cmd.extend(["--generator_checkpoint_episodes", *[str(ep) for ep in sample_episodes]])
    replay_cmd.extend(["--num_samples", str(args.num_samples)])
    replay_cmd.extend(["--epsilon", str(args.epsilon)])
    if args.approximate:
        replay_cmd.append("--approximate")
    if args.run_compound_online:
        replay_cmd.append("--run_compound_online")
    _append_variant_flags(replay_cmd, args)
    _append_analysis_flags(replay_cmd, args)

    run_command(sample_cmd, repo_root)
    run_command(replay_cmd, repo_root)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    print("\n".join(f"{key}={value}" for key, value in vars(args).items()), flush=True)
    main(args)
