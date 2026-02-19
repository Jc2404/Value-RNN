#!/usr/bin/env python3
"""
cross_eval_excel.py

Consistent with train.py:
- Uses environment subparsers: tmaze/hike/starkweather/tiger
- Uses --irrelevant wrapper like train.py
- Evaluates a saved agent across checkpoints: 0, eval_period, 2*eval_period, ... max_episode
- Exports an Excel file to results/{name}.xlsx (unless name already ends with .xlsx)

Expected checkpoint files (per DRQN.save/load):
  weights/{agent_id}-{episode}-Q.pth
  weights/{agent_id}-{episode}-Q_tar.pth
"""

import os
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import pandas as pd
import torch

from environments.tmaze import TMaze
from environments.hike import MountainHike
from environments.irrelevant import Irrelevant
from environments.starkweather import StarkweatherEnv
from environments.tiger import Tiger
from agents.drqn import DRQN

def build_environment(config) -> object:
    """Mirror train.py environment construction, but for evaluation."""
    if config.environment == "tmaze":
        env = TMaze(length=config.length, stochasticity=config.stochasticity, bayes=False)
    elif config.environment == "hike":
        env = MountainHike(variations=config.variations, bayes=False)
    elif config.environment == "starkweather":
        env = StarkweatherEnv(
            p_omission=config.p_omission,
            bin_size=config.bin_size,
            iti_hazard=config.iti_hazard,
            iti_min=config.iti_min,
            nITI_microstates=config.nITI_microstates,
        )
    elif config.environment == "tiger":
        env = Tiger(
            listen_accuracy=config.listen_accuracy,
            reward_listen=config.reward_listen,
            reward_correct=config.reward_correct,
            reward_wrong=config.reward_wrong,
            horizon=config.horizon,
            bayes=False,
        )
    else:
        raise NotImplementedError(f"Unknown environment {config.environment}")

    if config.irrelevant != 0:
        env = Irrelevant(env, state_size=config.irrelevant, bayes=False)

    return env


def checkpoint_paths(agent_id: str, episode: int) -> Tuple[str, str]:
    q = f"weights/{agent_id}-{episode}-Q.pth"
    qtar = f"weights/{agent_id}-{episode}-Q_tar.pth"
    return q, qtar


def checkpoint_exists(agent_id: str, episode: int) -> bool:
    q, qtar = checkpoint_paths(agent_id, episode)
    return os.path.exists(q) and os.path.exists(qtar)


@dataclass
class EvalRow:
    agent_id: str
    episode: int
    environment: str
    irrelevant: int
    cell: str
    hidden_size: int
    num_layers: int
    device: str
    num_rollouts: int
    action_size: int
    observation_size: int
    mean_return: float
    mean_discounted_return: float


def evaluate_checkpoint(
    *,
    agent_id: str,
    episode: int,
    env: object,
    cell: str,
    hidden_size: int,
    num_layers: int,
    device: torch.device,
    num_rollouts: int,
) -> Tuple[float, float]:
    """Load {agent_id, episode} and run greedy eval on env."""
    agent = DRQN(
        cell=cell,
        action_size=env.action_size,
        observation_size=env.observation_size,
        num_layers=num_layers,
        hidden_size=hidden_size,
    )
    agent.load(agent_id, episode=episode)
    agent.Q.to(device)
    agent.Q_tar.to(device)

    mean_return, mean_disc_return = agent.eval(env, num_rollouts)
    return float(mean_return), float(mean_disc_return)


def main(args):
    # Build env once (same env object used across episodes)
    env = build_environment(args)

    device = torch.device(args.device)
    if args.eval_period <= 0:
        raise ValueError("--eval-period must be > 0")
    if args.max_episode < 0:
        raise ValueError("--max-episode must be >= 0")

    rows: List[EvalRow] = []
    missing: List[int] = []

    for ep in range(0, args.max_episode + 1, args.eval_period):
        if not checkpoint_exists(args.agent_id, ep):
            missing.append(ep)
            continue

        mean_r, mean_dr = evaluate_checkpoint(
            agent_id=args.agent_id,
            episode=ep,
            env=env,
            cell=args.cell,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            device=device,
            num_rollouts=args.num_rollouts,
        )

        rows.append(
            EvalRow(
                agent_id=args.agent_id,
                episode=ep,
                environment=args.environment,
                irrelevant=int(args.irrelevant),
                cell=args.cell,
                hidden_size=int(args.hidden_size),
                num_layers=int(args.num_layers),
                device=str(device),
                num_rollouts=int(args.num_rollouts),
                action_size=int(env.action_size),
                observation_size=int(env.observation_size),
                mean_return=mean_r,
                mean_discounted_return=mean_dr,
            )
        )

        print(f"[eval] episode={ep:>6d}  return={mean_r:.4f}  disc_return={mean_dr:.4f}")

    # Output path
    os.makedirs(args.results_dir, exist_ok=True)
    out_name = args.name
    if not out_name.lower().endswith(".xlsx"):
        out_name += ".xlsx"
    out_path = os.path.join(args.results_dir, out_name)

    df = pd.DataFrame([asdict(r) for r in rows]).sort_values("episode")

    # Meta sheet: store full args + missing eps info
    meta_items = list(vars(args).items())
    meta_items.append(("action_size", int(env.action_size)))
    meta_items.append(("observation_size", int(env.observation_size)))
    meta_items.append(("missing_episodes", ", ".join(map(str, missing)) if missing else ""))
    meta_df = pd.DataFrame(meta_items, columns=["key", "value"])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="eval")
        meta_df.to_excel(writer, index=False, sheet_name="meta")

    print(f"[done] wrote {len(df)} rows -> {out_path}")
    if missing:
        print(f"[warn] missing checkpoints at episodes: {missing}")


if __name__ == "__main__":
    parser = ArgumentParser()

    # Output (keep "name" convention like train.py)
    parser.add_argument("--name", type=str, required=True,
                        help="Excel filename under results/ ('.xlsx' appended if missing).")
    parser.add_argument("--results-dir", type=str, default="results")

    # Which agent
    parser.add_argument("--agent-id", type=str, required=True,
                        help="Checkpoint id used by DRQN.save/load (weights/{agent-id}-{episode}-Q.pth).")

    # Network (must match training)
    parser.add_argument("--cell", type=str, default="gru", choices=["gru", "lstm", "rnn"])
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)

    # Eval schedule
    parser.add_argument("--eval-period", type=int, default=500,
                        help="Evaluate at episodes 0, eval_period, 2*eval_period, ...")
    parser.add_argument("--max-episode", type=int, required=True,
                        help="Max episode index to try (inclusive).")
    parser.add_argument("--num-rollouts", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")

    # Environment modification
    parser.add_argument("--irrelevant", type=int, default=0)

    # Environment (same style as train.py)
    environment_subparser = parser.add_subparsers(title="environment", dest="environment", required=True)

    # T-Maze
    tmaze = environment_subparser.add_parser("tmaze")
    tmaze.add_argument("--length", type=int, default=20)
    tmaze.add_argument("--stochasticity", type=float, default=0.0)

    # Mountain Hike
    hike = environment_subparser.add_parser("hike")
    hike.add_argument("--variations", type=str, default=None)

    # StarkWeather
    starkweather = environment_subparser.add_parser("starkweather")
    starkweather.add_argument("--p_omission", type=float, default=0.1)
    starkweather.add_argument("--bin_size", type=float, default=0.2)
    starkweather.add_argument("--iti_hazard", type=float, default=1/65)
    starkweather.add_argument("--iti_min", type=float, default=0)
    starkweather.add_argument("--nITI_microstates", type=int, default=10)

    # Tiger
    tiger = environment_subparser.add_parser("tiger")
    tiger.add_argument("--listen-accuracy", type=float, default=0.85)
    tiger.add_argument("--reward-listen", type=float, default=-1.0)
    tiger.add_argument("--reward-correct", type=float, default=10.0)
    tiger.add_argument("--reward-wrong", type=float, default=-100.0)
    tiger.add_argument("--horizon", type=int, default=20)

    args = parser.parse_args()
    print("\n".join(f"\033[90m{k}=\033[0m{v}" for k, v in vars(args).items()))
    main(args)
