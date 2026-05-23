import os
from argparse import ArgumentParser

import torch
import wandb

from agents.drqn import DRQN
from offline_trajectory_utils import (
    add_variant_test_flags,
    cache_file_path,
    collect_offline_trajectory_episode,
    default_cache_dir,
    ensure_dir,
    ordered_unique_ints,
    save_csv,
    save_json,
    write_cache_manifest,
)
from offline_replay_decode_eval import pick_variants
from retrain_decode_eval import build_environment, parse_variant
from utils import get_run_statistic


def build_parser():
    parser = ArgumentParser("Collect fixed offline trajectory caches per checkpoint and Protocol-B variant.")
    parser.add_argument("train_id", type=str)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="offline-trajectory-cache")
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--report_dir", type=str, default="report")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--checkpoint_episodes", type=int, nargs="+", required=True)
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Target number of decision timesteps to cache per checkpoint/variant.")
    parser.add_argument("--epsilon", type=float, default=0.0)
    add_variant_test_flags(parser)
    return parser


def main(args):
    train_args = get_run_statistic(args.train_id)
    checkpoint_episodes = ordered_unique_ints(args.checkpoint_episodes)
    variants = pick_variants(train_args, args)
    if not variants:
        variants = [("base", {})]

    cache_dir = os.path.abspath(args.cache_dir or default_cache_dir(args.report_dir, args.train_id))
    ensure_dir(cache_dir)

    cfg = vars(train_args) | vars(args)
    wandb.init(project=args.wandb_project, name=args.name, config=cfg, save_code=True)
    wandb.define_metric("generator/episode")
    wandb.define_metric("*", step_metric="generator/episode")
    wandb.save("*.py")
    wandb.save("agents/*.py")
    wandb.save("environments/*.py")

    env0 = build_environment(train_args, overrides=variants[0][1])
    agent = DRQN(
        cell=train_args.cell,
        action_size=env0.action_size,
        observation_size=env0.observation_size,
        num_layers=train_args.num_layers,
        hidden_size=train_args.hidden_size,
    )

    manifest_rows = []
    summary_rows = []

    for generator_episode in checkpoint_episodes:
        agent.load(args.train_id, episode=generator_episode, weights_dir=args.weights_dir)
        print(f"[generator episode {generator_episode}] agent loaded", flush=True)

        for variant_name, overrides in variants:
            env = build_environment(train_args, overrides=overrides)
            task_name, task_value = parse_variant(variant_name)
            cached_episodes = []
            collected_steps = 0

            while collected_steps < args.num_samples:
                episode_record = collect_offline_trajectory_episode(
                    agent,
                    env,
                    epsilon=args.epsilon,
                )
                episode_record["episode_id"] = len(cached_episodes)
                cached_episodes.append(episode_record)
                collected_steps += int(episode_record["length"])
                print(
                    f"[generator episode {generator_episode}] {variant_name} "
                    f"progress: steps={collected_steps}/{args.num_samples}, "
                    f"episodes={len(cached_episodes)}",
                    flush=True,
                )

            cache_path = os.path.abspath(cache_file_path(cache_dir, generator_episode, variant_name))
            ensure_dir(os.path.dirname(cache_path))
            payload = {
                "format_version": 1,
                "metadata": {
                    "train_id": args.train_id,
                    "generator_episode": int(generator_episode),
                    "variant": variant_name,
                    "task_name": task_name,
                    "task_value": task_value,
                    "environment": train_args.environment,
                    "env_overrides": overrides,
                    "epsilon": float(args.epsilon),
                    "num_samples_target": int(args.num_samples),
                    "num_samples_collected": int(collected_steps),
                    "num_episodes": int(len(cached_episodes)),
                    "checkpoint_episodes": checkpoint_episodes,
                },
                "episodes": cached_episodes,
            }
            torch.save(payload, cache_path)

            row = {
                "train_id": args.train_id,
                "generator_episode": int(generator_episode),
                "variant": variant_name,
                "task_name": task_name,
                "task_value": task_value,
                "num_samples_target": int(args.num_samples),
                "num_samples_collected": int(collected_steps),
                "num_episodes": int(len(cached_episodes)),
                "cache_path": cache_path,
            }
            manifest_rows.append(row)
            summary_rows.append(row)

            wandb.log({
                "generator/episode": int(generator_episode),
                "task/variant": variant_name,
                "cache/num_samples_collected": int(collected_steps),
                "cache/num_episodes": int(len(cached_episodes)),
            })

    manifest_path = write_cache_manifest(cache_dir, manifest_rows)
    save_csv(os.path.join(cache_dir, "cache_summary.csv"), summary_rows)
    save_json(os.path.join(cache_dir, "cache_summary.json"), summary_rows)
    print(f"Saved cache manifest: {manifest_path}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    print("\n".join(f"{key}={value}" for key, value in vars(args).items()), flush=True)
    main(args)
