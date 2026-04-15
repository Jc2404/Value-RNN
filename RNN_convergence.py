# probe_q_fit.py
#
# Diagnose whether a trained DRQN has converged with respect to its TD objective.
# For each saved checkpoint, this script:
#   1) loads the agent
#   2) samples trajectories with the current policy
#   3) builds replay-style transitions
#   4) computes Bellman targets using Q_tar
#   5) measures TD fit: MSE, MAE, bias, histogram, and optional pred-vs-target scatter
#
# Example:
#   python probe_q_fit.py myprobe w0s6rvqt --end_episode 5000 --period 500
#
# Outputs:
#   report/qfit_<name>_<train_id>/summary.png
#   report/qfit_<name>_<train_id>/summary.csv
#   report/qfit_<name>_<train_id>/ep_<E>_td_hist.png
#   report/qfit_<name>_<train_id>/ep_<E>_pred_vs_target.png   (optional)

import os
import csv
import random
from argparse import ArgumentParser
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from agents.drqn import DRQN
from environments.tmaze import TMaze
from environments.hike import MountainHike
from environments.irrelevant import Irrelevant
from environments.starkweather import StarkweatherEnv
from environments.tiger import Tiger
from environments.gridworld import GridWorld
from environments.crybaby import CryingBaby
from utils import get_run_statistic


def select_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_env_from_config(cfg, bayes: bool = False):
    """
    Rebuild the environment from a config/namespace object.
    This is the same logic as train.py, with bayes=False by default to match training.
    """
    if cfg.environment == "tmaze":
        env = TMaze(
            length=cfg.length,
            stochasticity=cfg.stochasticity,
            bayes=bayes,
        )
    elif cfg.environment == "hike":
        env = MountainHike(
            variations=cfg.variations,
            bayes=bayes,
        )
    elif cfg.environment == "starkweather":
        env = StarkweatherEnv(
            p_omission=cfg.p_omission,
            bin_size=cfg.bin_size,
            iti_hazard=cfg.iti_hazard,
            iti_min=cfg.iti_min,
            nITI_microstates=cfg.nITI_microstates,
        )
    elif cfg.environment == "tiger":
        env = Tiger(
            listen_accuracy=cfg.listen_accuracy,
            reward_listen=cfg.reward_listen,
            reward_correct=cfg.reward_correct,
            reward_wrong=cfg.reward_wrong,
            horizon=cfg.horizon,
            bayes=bayes,
        )
    elif cfg.environment == "gridworld":
        env = GridWorld(
            size=cfg.size,
            tprob=cfg.tprob,
            reward_scheme=cfg.reward_scheme,
            reward_margin=cfg.reward_margin,
            step_cost=cfg.step_cost,
            bayes=bayes,
        )
    elif cfg.environment == "crybaby":
        env = CryingBaby(
            p_hungry_if_full_wait=cfg.p_hungry_if_full_wait,
            p_stay_hungry_wait=cfg.p_stay_hungry_wait,
            p_full_if_feed=cfg.p_full_if_feed,
            p_cry_if_hungry=cfg.p_cry_if_hungry,
            p_cry_if_full=cfg.p_cry_if_full,
            reward_cry=cfg.reward_cry,
            cost_feed=cfg.cost_feed,
            bayes=bayes,
        )
    else:
        raise NotImplementedError(f"Unknown environment {cfg.environment}")

    if getattr(cfg, "irrelevant", 0) != 0:
        env = Irrelevant(
            env,
            state_size=cfg.irrelevant,
            bayes=bayes,
        )

    return env


@torch.no_grad()
def sample_transitions(agent: DRQN, env, num_trajectories: int, epsilon: float):
    """
    Sample transitions in the exact replay-style format used by Trajectory.get_transitions().
    """
    transitions = []
    for _ in range(num_trajectories):
        trajectory, = agent.play(env, epsilon=epsilon)
        transitions.extend(trajectory.get_transitions())
    return transitions


@torch.no_grad()
def compute_td_diagnostics(agent: DRQN, transitions, gamma: float, device: torch.device) -> Dict[str, np.ndarray]:
    """
    For each transition (seq_bef, a, r, o, d, seq_aft):
      pred   = Q(seq_bef)[-1, a]
      target = r                               if done
             = r + gamma * max_a' Q_tar(seq_aft)[-1, a']   otherwise

    Returns arrays plus summary metrics.
    """
    preds: List[float] = []
    targets: List[float] = []
    td_errors: List[float] = []

    agent.Q.eval()
    agent.Q_tar.eval()

    for seq_bef, a, r, _o, d, seq_aft in transitions:
        if a is None:
            continue

        seq_bef = seq_bef.to(device)
        seq_aft = seq_aft.to(device)

        q_bef, _ = agent.Q(seq_bef.unsqueeze(1))
        pred = q_bef[-1, 0, a]

        target = torch.tensor(float(r), dtype=torch.float32, device=device)
        if not d:
            q_next, _ = agent.Q_tar(seq_aft.unsqueeze(1))
            target = target + gamma * q_next[-1, 0, :].max()

        err = pred - target

        preds.append(float(pred.item()))
        targets.append(float(target.item()))
        td_errors.append(float(err.item()))

    preds_arr = np.asarray(preds, dtype=np.float32)
    targets_arr = np.asarray(targets, dtype=np.float32)
    td_arr = np.asarray(td_errors, dtype=np.float32)

    if len(preds_arr) == 0:
        return {
            "preds": preds_arr,
            "targets": targets_arr,
            "td_errors": td_arr,
            "mse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "n": 0,
        }

    return {
        "preds": preds_arr,
        "targets": targets_arr,
        "td_errors": td_arr,
        "mse": float(np.mean((preds_arr - targets_arr) ** 2)),
        "mae": float(np.mean(np.abs(preds_arr - targets_arr))),
        "bias": float(np.mean(preds_arr - targets_arr)),
        "n": int(len(preds_arr)),
    }


def plot_summary(checkpoints, mses, maes, biases, save_path: str) -> None:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(checkpoints, mses, marker="o", label="TD MSE")
    ax.plot(checkpoints, maes, marker="o", label="TD MAE")
    ax.plot(checkpoints, biases, marker="o", label="TD bias")

    ax.set_xlabel("checkpoint episode")
    ax.set_ylabel("metric")
    ax.set_title("DRQN fit to Bellman targets across checkpoints")
    ax.grid(True, alpha=0.3)
    ax.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_td_hist(td_errors: np.ndarray, title: str, save_path: str) -> None:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)

    ax.hist(td_errors, bins=40)
    ax.set_xlabel("TD error = Q - target")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pred_vs_target(preds: np.ndarray, targets: np.ndarray, title: str, save_path: str) -> None:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(preds, targets, s=8, alpha=0.5)
    lo = float(min(preds.min(), targets.min()))
    hi = float(max(preds.max(), targets.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--")

    ax.set_xlabel("predicted Q(s,a)")
    ax.set_ylabel("Bellman target")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_summary_csv(rows: List[Dict], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "n", "mse", "mae", "bias"])
        writer.writeheader()
        writer.writerows(rows)


def maybe_override_train_args(train_args, args):
    """
    By default we evaluate on the exact training environment.
    If --override_env is passed, selected parser values overwrite train_args.
    """
    if not args.override_env:
        return train_args

    for k, v in vars(args).items():
        if hasattr(train_args, k):
            setattr(train_args, k, v)
    return train_args


def main(args):
    device = select_device()
    set_seed(args.seed)

    train_args = get_run_statistic(args.train_id)
    train_args = maybe_override_train_args(train_args, args)

    env = build_env_from_config(train_args, bayes=False)

    agent = DRQN(
        cell=train_args.cell,
        action_size=env.action_size,
        observation_size=env.observation_size,
        num_layers=train_args.num_layers,
        hidden_size=train_args.hidden_size,
    )
    agent.Q.to(device)
    agent.Q_tar.to(device)

    if args.end_episode < 0:
        if hasattr(train_args, "num_episodes"):
            args.end_episode = int(train_args.num_episodes)
        else:
            raise ValueError("Could not infer end_episode from train_args; please pass --end_episode explicitly.")

    checkpoints = list(range(0, args.end_episode + 1, args.period))
    if checkpoints[-1] != args.end_episode:
        checkpoints.append(args.end_episode)

    name = args.name if args.name is not None else "noname"
    out_dir = os.path.join(args.report_dir, f"qfit_{name}_{args.train_id}")
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []
    mses, maes, biases = [], [], []

    gamma = float(env.gamma)

    print(f"[env] environment={train_args.environment} gamma={gamma}", flush=True)
    print(f"[agent] cell={train_args.cell} hidden_size={train_args.hidden_size} num_layers={train_args.num_layers}", flush=True)

    for ep in checkpoints:
        agent.load(args.train_id, episode=ep, weights_dir=args.weights_dir)
        print(f"[loaded] train_id={args.train_id} episode={ep}", flush=True)

        transitions = sample_transitions(
            agent=agent,
            env=env,
            num_trajectories=args.num_trajectories,
            epsilon=args.epsilon,
        )

        diag = compute_td_diagnostics(
            agent=agent,
            transitions=transitions,
            gamma=gamma,
            device=device,
        )

        print(
            f"[ep {ep}] n={diag['n']} mse={diag['mse']:.6f} "
            f"mae={diag['mae']:.6f} bias={diag['bias']:.6f}",
            flush=True,
        )

        mses.append(diag["mse"])
        maes.append(diag["mae"])
        biases.append(diag["bias"])
        summary_rows.append({
            "episode": ep,
            "n": diag["n"],
            "mse": diag["mse"],
            "mae": diag["mae"],
            "bias": diag["bias"],
        })

        hist_path = os.path.join(out_dir, f"ep_{ep}_td_hist.png")
        plot_td_hist(
            diag["td_errors"],
            title=f"TD error histogram | ep={ep}",
            save_path=hist_path,
        )

        if args.plot_pred_vs_target and diag["n"] > 0:
            scatter_path = os.path.join(out_dir, f"ep_{ep}_pred_vs_target.png")
            plot_pred_vs_target(
                diag["preds"],
                diag["targets"],
                title=f"Predicted Q vs Bellman target | ep={ep}",
                save_path=scatter_path,
            )

    summary_png = os.path.join(out_dir, "summary.png")
    summary_csv = os.path.join(out_dir, "summary.csv")

    plot_summary(checkpoints, mses, maes, biases, summary_png)
    save_summary_csv(summary_rows, summary_csv)

    print(f"[saved] {summary_png}", flush=True)
    print(f"[saved] {summary_csv}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser("Probe DRQN Bellman fit across checkpoints.")

    parser.add_argument("name", type=str, nargs="?", default=None)
    parser.add_argument("train_id", type=str)

    parser.add_argument("--report_dir", type=str, default="report")
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--period", type=int, default=500)
    parser.add_argument("--end_episode", type=int, default=-1)
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="Exploration rate used when sampling trajectories for diagnostics.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_trajectories", type=int, default=200)
    parser.add_argument("--plot_pred_vs_target", action="store_true")
    parser.add_argument("--override_env", action="store_true",
                        help="Use parser env args instead of the training config for environment construction.")

    

    """
    sub = parser.add_subparsers(dest="environment", required=False)
    # Optional override parser blocks. These are only used if --override_env is passed.
    tmaze = sub.add_parser("tmaze")
    tmaze.add_argument("--length", type=int, default=20)
    tmaze.add_argument("--stochasticity", type=float, default=0.0)

    hike = sub.add_parser("hike")
    hike.add_argument("--variations", type=str, default=None)

    starkweather = sub.add_parser("starkweather")
    starkweather.add_argument("--p_omission", type=float, default=0.1)
    starkweather.add_argument("--bin_size", type=float, default=0.2)
    starkweather.add_argument("--iti_hazard", type=float, default=1 / 65)
    starkweather.add_argument("--iti_min", type=float, default=0.0)
    starkweather.add_argument("--nITI_microstates", type=int, default=10)

    tiger = sub.add_parser("tiger")
    tiger.add_argument("--listen_accuracy", type=float, default=0.85)
    tiger.add_argument("--reward_listen", type=float, default=-1.0)
    tiger.add_argument("--reward_correct", type=float, default=10.0)
    tiger.add_argument("--reward_wrong", type=float, default=-100.0)
    tiger.add_argument("--horizon", type=int, default=20)

    gridworld = sub.add_parser("gridworld")
    gridworld.add_argument("--size", type=int, default=10)
    gridworld.add_argument("--tprob", type=float, default=0.7)
    gridworld.add_argument("--reward_scheme", type=str, default="julia")
    gridworld.add_argument("--reward_margin", type=int, default=2)
    gridworld.add_argument("--step_cost", type=float, default=0.0)

    crybaby = sub.add_parser("crybaby")
    crybaby.add_argument("--p_hungry_if_full_wait", type=float, default=0.10)
    crybaby.add_argument("--p_stay_hungry_wait", type=float, default=0.90)
    crybaby.add_argument("--p_full_if_feed", type=float, default=0.95)
    crybaby.add_argument("--p_cry_if_hungry", type=float, default=0.90)
    crybaby.add_argument("--p_cry_if_full", type=float, default=0.10)
    crybaby.add_argument("--reward_cry", type=float, default=-1.0)
    crybaby.add_argument("--cost_feed", type=float, default=-0.2)
    """

    args = parser.parse_args()
    print("\n".join(f"{k}={v}" for k, v in vars(args).items()), flush=True)
    main(args)
