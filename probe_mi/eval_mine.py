# eval_mine_estimator.py
import csv
import os
import sys
from pathlib import Path

import wandb
from argparse import ArgumentParser

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from mine.mine import MutualInformationNeuralEstimator
from utils import generate_hiddens_and_beliefs, get_run_statistic
from probe_mi.build_env import select_device, build_environment, build_agent
from probe_mi.mine_io import load_mine_config


def save_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_mine_from_cfg(cfg, device):
    return MutualInformationNeuralEstimator(
        hs_sizes=cfg.hs_sizes,
        belief_sizes=cfg.belief_sizes,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        alpha=cfg.alpha,
        representation_sizes=cfg.representation_sizes,
        belief_part=cfg.belief_part,
        device=device,
    )


def build_mine_manual(hiddens, beliefs, args, device):
    belief_sizes = []
    representation_sizes = []
    for b in beliefs:
        belief_sizes.append(b.size(-1))
        if b.ndim == 2:
            representation_sizes.append(None)
        elif b.ndim == 3:
            representation_sizes.append(args.representation_size)
        else:
            raise ValueError("Expected belief tensors to have 2 or 3 dims.")

    return MutualInformationNeuralEstimator(
        hs_sizes=hiddens.size(-1),
        belief_sizes=belief_sizes,
        hidden_size=args.mine_hidden_size,
        num_layers=args.mine_num_layers,
        alpha=args.mine_alpha,
        representation_sizes=representation_sizes,
        belief_part=args.belief_part,
        device=device,
    )


def main(args):
    device = select_device()
    print("Device:", device)

    # Build env/agent from the run you want to evaluate on
    eval_run_args = get_run_statistic(args.eval_train_id)
    env = build_environment(eval_run_args)
    agent = build_agent(eval_run_args, env)

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.name, config=vars(args))

    rows = []

    # Loop over episodes
    for agent_episode in range(args.episode_start, args.episode_end + 1, args.episode_step):
        estimator_episode = agent_episode if args.match_estimator_episode else args.estimator_episode

        # Load agent checkpoint
        agent.load(args.eval_train_id, episode=agent_episode, weights_dir=args.weights_dir)
        print(f"Loaded agent {args.eval_train_id} @ episode {agent_episode}")

        # Generate data for this checkpoint
        hiddens, beliefs = generate_hiddens_and_beliefs(
            agent, env,
            num_samples=args.num_samples,
            epsilon=args.epsilon,
            approximate=args.approximate,
        )

        # Build/load estimator for this episode
        if args.use_saved_cfg:
            cfg = load_mine_config(args.mine_id, estimator_episode, root=args.weights_dir)
            mine = build_mine_from_cfg(cfg, device)
        else:
            mine = build_mine_manual(hiddens, beliefs, args, device)

        mine.load(args.mine_id, episode=estimator_episode, weights_dir=args.weights_dir)

        mi = mine.estimate(hiddens, beliefs)
        print(f"Episode {agent_episode} (estimator {estimator_episode}): MI = {mi}")
        rows.append({
            "eval_train_id": args.eval_train_id,
            "agent_episode": agent_episode,
            "mine_id": args.mine_id,
            "estimator_episode": estimator_episode,
            "mi": float(mi),
        })

        if args.use_wandb:
            key = "mine_eval/mi" if mine.belief_part is None else f"mine_eval/mi-{mine.belief_part}"
            if args.epsilon != 0.0:
                key += f"-{args.epsilon}"
            wandb.log({
                "eval/agent_episode": agent_episode,
                "eval/train_id": args.eval_train_id,
                "estimator/id": args.mine_id,
                "estimator/episode": estimator_episode,
                key: mi,
            })

    if args.use_wandb:
        wandb.finish()

    if args.results_dir and rows:
        base_name = args.name if args.name is not None else args.mine_id
        results_path = os.path.join(args.results_dir, f"mine_eval_{base_name}_{args.eval_train_id}.csv")
        save_rows(results_path, rows)
        print(f"Saved evaluation summary: {results_path}")


if __name__ == "__main__":
    p = ArgumentParser("Evaluate saved MINE estimators across multiple agent checkpoints.")

    # What to evaluate on (agent/env)
    p.add_argument("eval_train_id", type=str)

    # Episode sweep
    p.add_argument("--episode_start", type=int, required=True)
    p.add_argument("--episode_end", type=int, required=True)
    p.add_argument("--episode_step", type=int, default=100)

    # Which estimator to load
    p.add_argument("--mine_id", type=str, required=True)
    p.add_argument("--weights_dir", type=str, default="weights")
    p.add_argument("--results_dir", type=str, default=None)

    # Matching mode
    p.add_argument(
        "--match_estimator_episode",
        action="store_true",
        help="If set, use estimator_episode = agent_episode for each point in the sweep."
    )
    p.add_argument(
        "--estimator_episode",
        type=int,
        default=0,
        help="Used only if --match_estimator_episode is NOT set."
    )

    # Evaluation sampling
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--approximate", action="store_true")

    # Recommended: load architecture from cfg saved during training
    p.add_argument("--use_saved_cfg", action="store_true")

    # Manual architecture (only needed if not using saved cfg)
    p.add_argument("--mine_num_layers", type=int, default=2)
    p.add_argument("--mine_hidden_size", type=int, default=256)
    p.add_argument("--mine_alpha", type=float, default=0.01)
    p.add_argument("--representation_size", type=int, default=16)
    p.add_argument("--belief_part", type=int, default=None)

    # Logging
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="belief-mi-ref")
    p.add_argument("--name", type=str, default=None)

    args = p.parse_args()
    main(args)
