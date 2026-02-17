#!/usr/bin/env python3
"""
cross_eval_excel.py

Evaluate a trained DRQN agent across multiple saved checkpoints and export to Excel.

Checkpoints are expected at:
  weights/{agent_id}-{episode}-Q.pth
  weights/{agent_id}-{episode}-Q_tar.pth

"""

import argparse
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List

import pandas as pd
import torch

from agents.drqn import DRQN
from environments.tmaze import TMaze
from environments.hike import MountainHike
from environments.starkweather import StarkweatherEnv
from environments.tiger import Tiger
from environments.irrelevant import Irrelevant

def build_env(env_name: str, args: argparse.Namespace, prefix: str) -> Any:
    """
    Build an env instance from args. `prefix` is "train" or "test".
    """
    def get(field: str, default=None):
        return getattr(args, f"{prefix}_{field}", default)

    if env_name == "tmaze":
        env = TMaze(
            length=get("length"),
            stochasticity=get("stochasticity"),
            bayes=False,
        )

    elif env_name == "hike":
        env = MountainHike(
            variations=get("variations"),
            bayes=False,
        )

    elif env_name == "starkweather":
        env = StarkweatherEnv(
            p_omission=get("p_omission"),
            bin_size=get("bin_size"),
            iti_hazard=get("iti_hazard"),
            iti_min=get("iti_min"),
            nITI_microstates=get("nITI_microstates"),
        )

    elif env_name == "tiger":
        env = Tiger(
            listen_accuracy=get("listen_accuracy"),
            reward_listen=get("reward_listen"),
            reward_correct=get("reward_correct"),
            reward_wrong=get("reward_wrong"),
            horizon=get("horizon"),
            bayes=False,
        )

    else:
        raise NotImplementedError(f"Unknown environment '{env_name}'")

    irr = get("irrelevant", 0)
    if irr and irr != 0:
        env = Irrelevant(env, state_size=irr, bayes=False)

    return env


def assert_env_compatible(train_env: Any, test_env: Any) -> None:
    """
    Check action/obs sizes match. Required to reuse the same Q network.
    """
    if train_env.action_size != test_env.action_size:
        raise ValueError(
            f"action_size mismatch: train={train_env.action_size}, test={test_env.action_size}.\n"
            "You can only reuse weights if action_size is identical."
        )
    if train_env.observation_size != test_env.observation_size:
        raise ValueError(
            f"observation_size mismatch: train={train_env.observation_size}, test={test_env.observation_size}.\n"
            "You can only reuse weights if observation_size is identical."
        )


def checkpoint_exists(agent_id: str, episode: int) -> bool:
    q = f"weights/{agent_id}-{episode}-Q.pth"
    qt = f"weights/{agent_id}-{episode}-Q_tar.pth"
    return os.path.exists(q) and os.path.exists(qt)

@dataclass
class Row:
    agent_id: str
    episode: int
    cell: str
    hidden_size: int
    num_layers: int
    device: str
    num_rollouts: int
    test_env: str
    train_env: Optional[str]
    test_action_size: int
    test_observation_size: int
    mean_return: float
    mean_discounted_return: float


def evaluate_one_checkpoint(
    *,
    agent_id: str,
    episode: int,
    cell: str,
    hidden_size: int,
    num_layers: int,
    device: torch.device,
    num_rollouts: int,
    test_env: Any,
) -> Dict[str, float]:
    """
    Load checkpoint at given episode and evaluate greedy policy on test_env.
    """
    agent = DRQN(
        cell=cell,
        action_size=test_env.action_size,
        observation_size=test_env.observation_size,
        num_layers=num_layers,
        hidden_size=hidden_size,
    )

    agent.load(agent_id, episode=episode)

    agent.Q.to(device)
    agent.Q_tar.to(device)

    mean_return, mean_disc_return = agent.eval(test_env, num_rollouts)
    return {
        "mean_return": float(mean_return),
        "mean_discounted_return": float(mean_disc_return),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cross-environment eval of saved DRQN checkpoints -> Excel.")

    # --- Agent / network ---
    parser.add_argument("--agent-id", required=True, type=str,
                        help="run_id used in weights/{agent-id}-{episode}-Q.pth")
    parser.add_argument("--cell", default="gru", type=str, choices=["gru", "lstm", "rnn"])
    parser.add_argument("--hidden-size", default=32, type=int)
    parser.add_argument("--num-layers", default=2, type=int)

    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--num-rollouts", default=50, type=int)

    # --- Multi-checkpoint eval controls ---
    parser.add_argument("--eval-period", default=500, type=int,
                        help="Evaluate checkpoints at episodes 0, eval-period, 2*eval-period, ...")
    parser.add_argument("--max-episode", required=True, type=int,
                        help="Largest episode index to try (inclusive).")

    # --- Output ---
    parser.add_argument("--out-name", required=True, type=str,
                        help="Excel filename written under results/. e.g. my_eval.xlsx")
    parser.add_argument("--results-dir", default="results", type=str)

    # --- Environments ---
    parser.add_argument("--test-env", required=True, type=str,
                        choices=["tmaze", "hike", "starkweather", "tiger"])
    parser.add_argument("--train-env", default=None, type=str,
                        choices=["tmaze", "hike", "starkweather", "tiger"],
                        help="Optional reference env (original training config) for compatibility checks.")

    # irrelevant wrapper sizes
    parser.add_argument("--test-irrelevant", type=int, default=0)
    parser.add_argument("--train-irrelevant", type=int, default=0)

    # ---- TMaze params ----
    parser.add_argument("--test-length", type=int, default=20)
    parser.add_argument("--test-stochasticity", type=float, default=0.0)
    parser.add_argument("--train-length", type=int, default=20)
    parser.add_argument("--train-stochasticity", type=float, default=0.0)

    # ---- Hike params ----
    parser.add_argument("--test-variations", type=int, default=4)
    parser.add_argument("--train-variations", type=int, default=4)

    # ---- Starkweather params ----
    parser.add_argument("--test-p-omission", type=float, default=0.0)
    parser.add_argument("--test-bin-size", type=int, default=1)
    parser.add_argument("--test-iti-hazard", type=float, default=0.0)
    parser.add_argument("--test-iti-min", type=int, default=1)
    parser.add_argument("--test-nITI-microstates", type=int, default=1)

    parser.add_argument("--train-p-omission", type=float, default=0.0)
    parser.add_argument("--train-bin-size", type=int, default=1)
    parser.add_argument("--train-iti-hazard", type=float, default=0.0)
    parser.add_argument("--train-iti-min", type=int, default=1)
    parser.add_argument("--train-nITI-microstates", type=int, default=1)

    # ---- Tiger params ----
    parser.add_argument("--test-listen-accuracy", type=float, default=0.85)
    parser.add_argument("--test-reward-listen", type=float, default=-1.0)
    parser.add_argument("--test-reward-correct", type=float, default=10.0)
    parser.add_argument("--test-reward-wrong", type=float, default=-100.0)
    parser.add_argument("--test-horizon", type=int, default=10)

    parser.add_argument("--train-listen-accuracy", type=float, default=0.85)
    parser.add_argument("--train-reward-listen", type=float, default=-1.0)
    parser.add_argument("--train-reward-correct", type=float, default=10.0)
    parser.add_argument("--train-reward-wrong", type=float, default=-100.0)
    parser.add_argument("--train-horizon", type=int, default=10)

    args = parser.parse_args(argv)

    test_env = build_env(args.test_env, args, prefix="test")

    train_env = None
    if args.train_env is not None:
        train_env = build_env(args.train_env, args, prefix="train")
        assert_env_compatible(train_env, test_env)

    device = torch.device(args.device)

    rows: List[Row] = []
    missing: List[int] = []

    if args.eval_period <= 0:
        raise ValueError("--eval-period must be a positive integer.")

    for ep in range(0, args.max_episode + 1, args.eval_period):
        if not checkpoint_exists(args.agent_id, ep):
            missing.append(ep)
            continue

        metrics = evaluate_one_checkpoint(
            agent_id=args.agent_id,
            episode=ep,
            cell=args.cell,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            device=device,
            num_rollouts=args.num_rollouts,
            test_env=test_env,
        )

        rows.append(
            Row(
                agent_id=args.agent_id,
                episode=ep,
                cell=args.cell,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                device=str(device),
                num_rollouts=args.num_rollouts,
                test_env=args.test_env,
                train_env=args.train_env,
                test_action_size=int(test_env.action_size),
                test_observation_size=int(test_env.observation_size),
                mean_return=metrics["mean_return"],
                mean_discounted_return=metrics["mean_discounted_return"],
            )
        )

        print(f"[eval] episode={ep:>6d}  return={metrics['mean_return']:.4f}  disc_return={metrics['mean_discounted_return']:.4f}")

    # Export to Excel
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, args.out_name)

    df = pd.DataFrame([asdict(r) for r in rows]).sort_values("episode")

    # Add a small "meta" sheet + the main results sheet
    meta = {
        "agent_id": args.agent_id,
        "cell": args.cell,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "device": str(device),
        "num_rollouts": args.num_rollouts,
        "eval_period": args.eval_period,
        "max_episode": args.max_episode,
        "test_env": args.test_env,
        "train_env": args.train_env,
        "test_action_size": int(test_env.action_size),
        "test_observation_size": int(test_env.observation_size),
        "missing_episodes": ", ".join(map(str, missing)) if missing else "",
    }
    meta_df = pd.DataFrame(list(meta.items()), columns=["key", "value"])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="eval")
        meta_df.to_excel(writer, index=False, sheet_name="meta")

    print(f"[done] wrote {len(df)} rows to {out_path}")
    if missing:
        print(f"[warn] missing checkpoints for episodes: {missing}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
