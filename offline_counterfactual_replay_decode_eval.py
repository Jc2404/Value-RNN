import os
from argparse import ArgumentParser

import pandas as pd
import torch
import wandb

from agents.drqn import DRQN
from counterfactual_variant_utils import (
    add_counterfactual_variant_flags,
    parse_counterfactual_variant,
    pick_counterfactual_variants,
)
from offline_replay_decode_eval import (
    run_enabled_analyses,
    shuffle_split_pairs,
    write_summary_outputs,
)
from offline_trajectory_utils import (
    add_offline_analysis_flags,
    artefacts_dir,
    assert_counterfactual_belief_replay_supported,
    counterfactual_belief_cache_path,
    counterfactual_replay_root,
    ensure_dir,
    flatten_cached_replay,
    load_cache_manifest,
    matched_trajectory_cache_root,
    ordered_unique_ints,
    pair_artifact_dir,
    recompute_counterfactual_belief_episode,
    replay_agent_inputs,
    run_root_dir,
    save_csv,
    save_json,
    verify_hidden_preview,
)
from retrain_decode_eval import (
    build_environment,
    resolve_softmax_probe_specs,
    select_device,
)
from utils import get_run_statistic


def build_parser():
    parser = ArgumentParser(
        "Replay matched trajectory caches under counterfactual environment beliefs and fit offline decoders/MINE."
    )
    parser.add_argument("train_id", type=str)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="offline-counterfactual-replay-decode")
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--checkpoint_episodes", type=int, nargs="+", required=True)
    parser.add_argument(
        "--generator_checkpoint_episodes",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of cached generator checkpoints to replay. Defaults to all matched caches under the run root.",
    )
    add_counterfactual_variant_flags(parser)
    add_offline_analysis_flags(parser)
    return parser


def write_counterfactual_cache_outputs(summary_root: str, train_id: str, summary_rows, invalid_rows) -> None:
    ensure_dir(summary_root)

    summary_csv_path = os.path.join(summary_root, f"counterfactual_belief_cache_summary_{train_id}.csv")
    summary_json_path = os.path.join(summary_root, f"counterfactual_belief_cache_summary_{train_id}.json")
    summary_excel_path = os.path.join(summary_root, f"counterfactual_belief_cache_summary_{train_id}.xlsx")
    save_csv(summary_csv_path, summary_rows)
    save_json(summary_json_path, summary_rows)

    invalid_csv_path = os.path.join(summary_root, f"counterfactual_invalid_episodes_{train_id}.csv")
    invalid_json_path = os.path.join(summary_root, f"counterfactual_invalid_episodes_{train_id}.json")
    invalid_excel_path = os.path.join(summary_root, f"counterfactual_invalid_episodes_{train_id}.xlsx")
    save_csv(invalid_csv_path, invalid_rows)
    save_json(invalid_json_path, invalid_rows)

    with pd.ExcelWriter(summary_excel_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="belief_cache_summary", index=False)

    with pd.ExcelWriter(invalid_excel_path, engine="openpyxl") as writer:
        pd.DataFrame(invalid_rows).to_excel(writer, sheet_name="invalid_episodes", index=False)

    print(f"Saved belief cache summary CSV: {summary_csv_path}", flush=True)
    print(f"Saved invalid episode CSV: {invalid_csv_path}", flush=True)


def _build_belief_cache(
    train_args,
    cache_payload,
    source_cache_path: str,
    counterfactual_root_dir: str,
    generator_episode: int,
    variant_name: str,
    overrides,
):
    env = build_environment(train_args, overrides=overrides)
    assert_counterfactual_belief_replay_supported(env)

    task_name, task_value = parse_counterfactual_variant(variant_name)
    total_episodes = int(len(cache_payload["episodes"]))
    total_samples = int(sum(int(episode["length"]) for episode in cache_payload["episodes"]))

    valid_episodes = []
    invalid_rows = []
    invalid_episodes = []

    for episode_record in cache_payload["episodes"]:
        result = recompute_counterfactual_belief_episode(env, episode_record)
        if result["valid"]:
            valid_episodes.append(result["episode"])
            continue

        invalid_episodes.append(result)
        invalid_rows.append({
            "generator_episode": int(generator_episode),
            "variant": variant_name,
            "task_name": task_name,
            "task_value": task_value,
            "source_cache_path": source_cache_path,
            "episode_id": result["episode_id"],
            "episode_length": int(result["length"]),
            "first_invalid_step": int(result["first_invalid_step"]),
            "reason": result["reason"],
            "exception": result.get("exception"),
        })

    valid_samples = int(sum(int(episode["length"]) for episode in valid_episodes))
    discarded_episodes = int(total_episodes - len(valid_episodes))
    discarded_samples = int(total_samples - valid_samples)

    if discarded_episodes == 0:
        status = "ok"
    elif valid_episodes:
        status = "partial_invalid"
    else:
        status = "all_invalid"

    belief_cache_file = os.path.abspath(
        counterfactual_belief_cache_path(counterfactual_root_dir, generator_episode, variant_name)
    )
    ensure_dir(os.path.dirname(belief_cache_file))

    summary_row = {
        "generator_episode": int(generator_episode),
        "variant": variant_name,
        "task_name": task_name,
        "task_value": task_value,
        "trajectory_variant": "base",
        "source_cache_path": source_cache_path,
        "belief_cache_path": belief_cache_file,
        "status": status,
        "num_episodes_total": int(total_episodes),
        "num_episodes_valid": int(len(valid_episodes)),
        "num_episodes_discarded": int(discarded_episodes),
        "num_samples_total": int(total_samples),
        "num_samples_valid": int(valid_samples),
        "num_samples_discarded": int(discarded_samples),
    }

    payload = {
        "format_version": 1,
        "metadata": summary_row | {
            "train_id": cache_payload["metadata"]["train_id"],
            "environment": cache_payload["metadata"]["environment"],
            "env_overrides": overrides,
            "source_cache_metadata": cache_payload["metadata"],
        },
        "valid_episodes": valid_episodes,
        "invalid_episodes": invalid_episodes,
    }
    torch.save(payload, belief_cache_file)

    print(
        f"[generator episode {generator_episode}] {variant_name} counterfactual belief cache: "
        f"kept {len(valid_episodes)}/{total_episodes} episodes and "
        f"{valid_samples}/{total_samples} samples",
        flush=True,
    )
    return summary_row, invalid_rows


def _save_pair_stub(pair_dir: str, row: dict, pair_meta: dict) -> None:
    ensure_dir(pair_dir)
    save_csv(os.path.join(pair_dir, "metrics.csv"), [row])
    save_json(os.path.join(pair_dir, "metadata.json"), pair_meta)


def main(args):
    train_args = get_run_statistic(args.train_id)
    evaluator_episodes = ordered_unique_ints(args.checkpoint_episodes)
    generator_episode_filter = None
    if args.generator_checkpoint_episodes:
        generator_episode_filter = set(ordered_unique_ints(args.generator_checkpoint_episodes))

    run_root = os.path.abspath(run_root_dir(args.name))
    cache_dir = os.path.abspath(matched_trajectory_cache_root(run_root))
    artefact_root = os.path.abspath(artefacts_dir(run_root))
    counterfactual_root_dir = os.path.abspath(counterfactual_replay_root(artefact_root))

    cache_rows = []
    if os.path.isdir(cache_dir):
        cache_rows = load_cache_manifest(cache_dir)

    cache_rows = [row for row in cache_rows if str(row.get("variant")) == "base"]
    if generator_episode_filter is not None:
        cache_rows = [
            row for row in cache_rows
            if int(row["generator_episode"]) in generator_episode_filter
        ]

    if not cache_rows:
        raise FileNotFoundError(
            f"No matched trajectory caches found under {cache_dir}"
            + (
                f" for generator episodes {sorted(generator_episode_filter)}."
                if generator_episode_filter is not None else "."
            )
            + " Run sample_matched_trajectories.py first."
        )

    variants = pick_counterfactual_variants(train_args, args)
    if not variants:
        variants = [("base", {})]

    env0 = build_environment(train_args, overrides=None)
    assert_counterfactual_belief_replay_supported(env0)

    device = select_device()
    softmax_specs = resolve_softmax_probe_specs(args)
    if not (args.run_mi or args.run_regression or softmax_specs):
        raise ValueError(
            "No offline analyses enabled. Use at least one of "
            "--run_mi --run_regression --run_softmax_linear_probe --run_softmax_mlp_probe"
        )

    agent = DRQN(
        cell=train_args.cell,
        action_size=env0.action_size,
        observation_size=env0.observation_size,
        num_layers=train_args.num_layers,
        hidden_size=train_args.hidden_size,
    )

    cfg = vars(train_args) | vars(args)
    wandb.init(project=args.wandb_project, name=args.name, config=cfg, save_code=True)
    wandb.define_metric("train/episode")
    wandb.define_metric("mine_optim/global_step")
    wandb.define_metric("*", step_metric="train/episode")
    wandb.define_metric("mine_optim/*", step_metric="mine_optim/global_step")
    wandb.save("*.py")
    wandb.save("agents/*.py")
    wandb.save("environments/*.py")
    wandb.save("mine/*.py")

    belief_cache_summary_rows = []
    belief_cache_invalid_rows = []
    replay_specs = []

    for cache_row in cache_rows:
        source_cache_path = cache_row["cache_path"]
        cache_payload = torch.load(source_cache_path)
        generator_episode = int(cache_payload["metadata"]["generator_episode"])

        for variant_name, overrides in variants:
            summary_row, invalid_rows = _build_belief_cache(
                train_args=train_args,
                cache_payload=cache_payload,
                source_cache_path=source_cache_path,
                counterfactual_root_dir=counterfactual_root_dir,
                generator_episode=generator_episode,
                variant_name=variant_name,
                overrides=overrides,
            )
            belief_cache_summary_rows.append(summary_row)
            belief_cache_invalid_rows.extend(invalid_rows)
            replay_specs.append(summary_row)

    write_counterfactual_cache_outputs(
        counterfactual_root_dir,
        args.train_id,
        belief_cache_summary_rows,
        belief_cache_invalid_rows,
    )

    summary_rows = []
    per_eval_rows = {episode: [] for episode in evaluator_episodes}

    for evaluator_episode in evaluator_episodes:
        agent.load(args.train_id, episode=evaluator_episode, weights_dir=args.weights_dir)
        print(f"[evaluator episode {evaluator_episode}] agent loaded", flush=True)

        for replay_spec in replay_specs:
            generator_episode = int(replay_spec["generator_episode"])
            variant_name = replay_spec["variant"]
            task_name = replay_spec["task_name"]
            task_value = replay_spec["task_value"]
            source_cache_path = replay_spec["source_cache_path"]
            belief_cache_file = replay_spec["belief_cache_path"]

            source_cache_payload = torch.load(source_cache_path)
            source_episode_map = {
                int(episode["episode_id"]): episode
                for episode in source_cache_payload["episodes"]
            }
            belief_cache_payload = torch.load(belief_cache_file)
            valid_episodes = belief_cache_payload["valid_episodes"]

            pair_dir = pair_artifact_dir(
                counterfactual_root_dir,
                generator_episode,
                variant_name,
                evaluator_episode,
            )
            row = {
                "setup": "offline_counterfactual_replay",
                "run_name": args.name,
                "train_id": args.train_id,
                "generator_episode": int(generator_episode),
                "evaluator_episode": int(evaluator_episode),
                "variant": variant_name,
                "task_name": task_name,
                "task_value": task_value,
                "trajectory_variant": "base",
                "source_cache_path": source_cache_path,
                "belief_cache_path": belief_cache_file,
                "num_episodes_total": int(replay_spec["num_episodes_total"]),
                "num_episodes_valid": int(replay_spec["num_episodes_valid"]),
                "num_episodes_discarded": int(replay_spec["num_episodes_discarded"]),
                "num_samples_total": int(replay_spec["num_samples_total"]),
                "num_samples_valid": int(replay_spec["num_samples_valid"]),
                "num_samples_discarded": int(replay_spec["num_samples_discarded"]),
            }
            pair_meta = {
                "setup": "offline_counterfactual_replay",
                "run_name": args.name,
                "train_id": args.train_id,
                "generator_episode": int(generator_episode),
                "evaluator_episode": int(evaluator_episode),
                "variant": variant_name,
                "task_name": task_name,
                "task_value": task_value,
                "trajectory_variant": "base",
                "source_cache_path": source_cache_path,
                "belief_cache_path": belief_cache_file,
            }

            if not valid_episodes:
                row["status"] = "all_episodes_invalid"
                row["num_samples_train"] = 0
                row["num_samples_eval"] = 0
                _save_pair_stub(pair_dir, row, pair_meta)
                summary_rows.append(row)
                per_eval_rows[int(evaluator_episode)].append(row)
                print(
                    f"[evaluator episode {evaluator_episode}] {variant_name} "
                    "skipped: all episodes invalid under counterfactual replay",
                    flush=True,
                )
                continue

            replayed_sequences = []
            aligned_episodes = []
            for belief_episode in valid_episodes:
                episode_id = int(belief_episode["episode_id"])
                source_episode = source_episode_map[episode_id]
                replayed = replay_agent_inputs(agent, source_episode["agent_inputs"])
                if evaluator_episode == generator_episode:
                    verify_hidden_preview(source_episode["generator_hidden_preview"], replayed)
                replayed_sequences.append(replayed)
                aligned_episodes.append(belief_episode)

            X_all, B_all = flatten_cached_replay(aligned_episodes, replayed_sequences)
            row["num_samples_replayed"] = int(X_all.size(0))

            if int(X_all.size(0)) < 2:
                row["status"] = "insufficient_valid_samples"
                row["num_samples_train"] = 0
                row["num_samples_eval"] = int(X_all.size(0))
                _save_pair_stub(pair_dir, row, pair_meta)
                summary_rows.append(row)
                per_eval_rows[int(evaluator_episode)].append(row)
                print(
                    f"[evaluator episode {evaluator_episode}] {variant_name} "
                    f"skipped: only {int(X_all.size(0))} valid samples remain",
                    flush=True,
                )
                continue

            X_all = X_all.to(device)
            B_all = tuple(part.to(device) for part in B_all)
            Xtr, Xte, Btr, Bte = shuffle_split_pairs(X_all, B_all, args.valid_size, device)

            row["status"] = "ok"
            row["num_samples_train"] = int(Xtr.size(0))
            row["num_samples_eval"] = int(Xte.size(0))

            run_enabled_analyses(
                args,
                device,
                softmax_specs,
                setup="offline_counterfactual_replay",
                evaluator_episode=evaluator_episode,
                variant_name=variant_name,
                generator_episode=generator_episode,
                pair_dir=pair_dir,
                row=row,
                pair_meta=pair_meta,
                mi_train_inputs=Xtr,
                mi_train_beliefs=Btr,
                mi_eval_inputs=Xte,
                mi_eval_beliefs=Bte,
                probe_train_inputs=Xtr,
                probe_train_beliefs=Btr,
                probe_eval_inputs=Xte,
                probe_eval_beliefs=Bte,
            )
            summary_rows.append(row)
            per_eval_rows[int(evaluator_episode)].append(row)

    write_summary_outputs(
        counterfactual_root_dir,
        f"counterfactual_replay_summary_{args.train_id}",
        summary_rows,
        per_eval_rows,
        sort_columns=["generator_episode", "variant"],
    )

    wandb.finish()


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    print("\n".join(f"{key}={value}" for key, value in vars(args).items()), flush=True)
    main(args)
