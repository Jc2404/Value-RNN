import os
from argparse import ArgumentParser

import pandas as pd
import torch
import wandb

from agents.drqn import DRQN
from offline_trajectory_utils import (
    add_offline_analysis_flags,
    default_artifact_dir,
    default_cache_dir,
    ensure_dir,
    flatten_cached_replay,
    load_cache_manifest,
    ordered_unique_ints,
    pair_artifact_dir,
    replay_agent_inputs,
    save_csv,
    save_json,
    save_linreg_probe,
    save_mine_estimator,
    save_softmax_probe,
    verify_hidden_preview,
)
from retrain_decode_eval import (
    build_environment,
    build_mine,
    eval_linear_probe_torch,
    eval_softmax_probe_torch,
    fit_linear_probe_torch,
    fit_softmax_probe_torch,
    parse_variant,
    resolve_softmax_probe_specs,
    select_device,
)
from utils import get_run_statistic


def pick_variants(train_args, args):
    """
    Offline-workflow-local copy of the Protocol B variant grid.

    Edit this function if you want the offline trajectory workflow to use a
    different sweep without changing the original `retrain_decode_eval.py`
    behavior.
    """
    env_name = train_args.environment
    variants = []

    if env_name == "tmaze":
        if args.test_length:
            for length in [20, 30, 40]:
                variants.append((f"tmaze_length={length}", {"length": length}))
            return variants
        if args.test_stochasticity:
            for stochasticity in [0.0, 0.1, 0.4]:
                variants.append((f"tmaze_stochasticity={stochasticity}", {"stochasticity": stochasticity}))
            return variants

    if env_name == "hike":
        if args.test_variations:
            for variations in [1, 2, 4, 8]:
                variants.append((f"hike_variations={variations}", {"variations": variations}))
            return variants

    if env_name == "starkweather":
        if args.test_p_omission:
            for p_omission in [0.0, 0.2, 0.4]:
                variants.append((f"starkweather_p_omission={p_omission}", {"p_omission": p_omission}))
            return variants
        if args.test_bin_size:
            grid = [train_args.bin_size, max(1, train_args.bin_size // 2), train_args.bin_size * 2]
            seen = set()
            deduped = []
            for bin_size in grid:
                if bin_size not in seen:
                    deduped.append(bin_size)
                    seen.add(bin_size)
            for bin_size in deduped:
                variants.append((f"starkweather_bin_size={bin_size}", {"bin_size": bin_size}))
            return variants
        if args.test_iti_hazard:
            for iti_hazard in [0.01, 0.05, 0.1, 0.2]:
                variants.append((f"starkweather_iti_hazard={iti_hazard}", {"iti_hazard": iti_hazard}))
            return variants
        if args.test_iti_min:
            for iti_min in [0, 5, 10, 20]:
                variants.append((f"starkweather_iti_min={iti_min}", {"iti_min": iti_min}))
            return variants
        if args.test_nITI_microstates:
            for count in [1, 2, 4, 8]:
                variants.append((f"starkweather_nITI_microstates={count}", {"nITI_microstates": count}))
            return variants

    if env_name == "tiger":
        if args.test_listen_accuracy:
            for accuracy in [0.4, 0.75, 0.9]:
                variants.append((f"listen_accuracy={accuracy}", {"listen_accuracy": accuracy}))
            return variants
        if args.test_reward_listen:
            for reward_listen in [-3, -1, 1]:
                variants.append((f"reward_listen={reward_listen}", {"reward_listen": reward_listen}))
            return variants

    if env_name == "gridworld":
        if args.test_grid_size:
            for size in [6, 10, 14]:
                variants.append((f"grid_size={size}", {"size": size}))
            return variants
        if args.test_tprob:
            for tprob in [0.1, 0.7, 1.0]:
                variants.append((f"tprob={tprob}", {"tprob": tprob}))
            return variants
        if args.test_reward_scheme:
            for reward_scheme in ["symmetric", "center", "scaled"]:
                variants.append((f"reward_scheme={reward_scheme}", {"reward_scheme": reward_scheme}))
            return variants
        if args.test_reward_margin:
            for reward_margin in [0, 2, 4]:
                variants.append((f"reward_margin={reward_margin}", {"reward_margin": reward_margin}))
            return variants

    if env_name == "crybaby":
        if args.test_p_cry_if_hungry:
            for p_cry_if_hungry in [0.5, 0.75, 0.9]:
                variants.append((f"crybaby_p_cry_if_hungry={p_cry_if_hungry}", {"p_cry_if_hungry": p_cry_if_hungry}))
            return variants
        if args.test_p_cry_if_full:
            for p_cry_if_full in [0.0, 0.2, 0.4]:
                variants.append((f"crybaby_p_cry_if_full={p_cry_if_full}", {"p_cry_if_full": p_cry_if_full}))
            return variants

    return variants


def build_parser():
    parser = ArgumentParser("Replay cached trajectories through agent checkpoints and fit offline decoders/MINE.")
    parser.add_argument("train_id", type=str)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="offline-replay-decode")
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--report_dir", type=str, default="report")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--artifact_dir", type=str, default=None)
    parser.add_argument("--checkpoint_episodes", type=int, nargs="+", required=True)
    add_offline_analysis_flags(parser)
    return parser


def shuffle_split_pairs(X, beliefs, valid_size, device):
    if not (0.0 < valid_size < 1.0):
        raise ValueError(f"valid_size should be in (0, 1), got {valid_size}")

    num_samples = X.size(0)
    if num_samples < 2:
        raise ValueError("Need at least two replayed samples for train/eval splitting.")

    perm = torch.randperm(num_samples, device=device)
    X = X[perm]
    beliefs = tuple(part[perm] for part in beliefs)

    split = int(num_samples * (1.0 - valid_size))
    split = min(max(split, 1), num_samples - 1)
    return X[:split], X[split:], tuple(part[:split] for part in beliefs), tuple(part[split:] for part in beliefs)


def make_offline_mine_logger(evaluator_episode, generator_episode, variant_name, num_epochs):
    def _log(payload):
        payload = dict(payload)
        if "mine_optim/epoch" in payload:
            epoch = int(payload["mine_optim/epoch"])
            payload["mine_optim/global_step"] = int(evaluator_episode) * int(num_epochs) + epoch
        payload["train/episode"] = int(evaluator_episode)
        payload["task/variant"] = variant_name
        payload["task/generator_episode"] = int(generator_episode)
        wandb.log(payload)

    return _log


def add_softmax_metrics(row, res_train, res_eval, spec_name, part_idx):
    if part_idx == 0:
        suffix = ""
    else:
        suffix = f"-{part_idx}"

    row[f"softmax_{spec_name}_KL{suffix}"] = res_eval["kl"]
    row[f"softmax_{spec_name}_CE{suffix}"] = res_eval["ce"]
    row[f"softmax_{spec_name}_H_true{suffix}"] = res_eval["true_entropy"]
    row[f"softmax_{spec_name}_H_pred{suffix}"] = res_eval["pred_entropy"]
    row[f"softmax_{spec_name}_JS{suffix}"] = res_eval["js"]
    row[f"softmax_{spec_name}_KL_train{suffix}"] = res_train["kl"]
    row[f"softmax_{spec_name}_CE_train{suffix}"] = res_train["ce"]
    row[f"softmax_{spec_name}_H_true_train{suffix}"] = res_train["true_entropy"]
    row[f"softmax_{spec_name}_H_pred_train{suffix}"] = res_train["pred_entropy"]
    row[f"softmax_{spec_name}_JS_train{suffix}"] = res_train["js"]


def log_softmax_metrics(evaluator_episode, generator_episode, variant_name, spec_name, part_idx, res_train, res_eval):
    if part_idx == 0:
        suffix = ""
    else:
        suffix = f"-{part_idx}"

    wandb.log({
        "train/episode": int(evaluator_episode),
        "task/variant": variant_name,
        "task/generator_episode": int(generator_episode),
        f"softmax/{spec_name}/KL{suffix}": res_eval["kl"],
        f"softmax/{spec_name}/CE{suffix}": res_eval["ce"],
        f"softmax/{spec_name}/H_true{suffix}": res_eval["true_entropy"],
        f"softmax/{spec_name}/H_pred{suffix}": res_eval["pred_entropy"],
        f"softmax/{spec_name}/JS{suffix}": res_eval["js"],
        f"softmax/{spec_name}/KL_train{suffix}": res_train["kl"],
        f"softmax/{spec_name}/CE_train{suffix}": res_train["ce"],
        f"softmax/{spec_name}/H_true_train{suffix}": res_train["true_entropy"],
        f"softmax/{spec_name}/H_pred_train{suffix}": res_train["pred_entropy"],
        f"softmax/{spec_name}/JS_train{suffix}": res_train["js"],
    })


def main(args):
    train_args = get_run_statistic(args.train_id)
    evaluator_episodes = ordered_unique_ints(args.checkpoint_episodes)
    cache_dir = os.path.abspath(args.cache_dir or default_cache_dir(args.report_dir, args.train_id))
    artifact_root = os.path.abspath(args.artifact_dir or default_artifact_dir(args.report_dir, args.train_id))
    ensure_dir(artifact_root)

    cache_rows = load_cache_manifest(cache_dir)
    if not cache_rows:
        raise FileNotFoundError(f"No trajectory caches found under {cache_dir}")

    device = select_device()
    softmax_specs = resolve_softmax_probe_specs(args)
    if not (args.run_mi or args.run_regression or softmax_specs):
        raise ValueError(
            "No offline analyses enabled. Use at least one of "
            "--run_mi --run_regression --run_softmax_linear_probe --run_softmax_mlp_probe"
        )

    env0 = build_environment(train_args, overrides=None)
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

    summary_rows = []
    per_eval_rows = {episode: [] for episode in evaluator_episodes}

    for evaluator_episode in evaluator_episodes:
        agent.load(args.train_id, episode=evaluator_episode, weights_dir=args.weights_dir)
        print(f"[evaluator episode {evaluator_episode}] agent loaded", flush=True)

        for cache_row in cache_rows:
            cache_path = cache_row["cache_path"]
            cache_payload = torch.load(cache_path)
            meta = cache_payload["metadata"]
            generator_episode = int(meta["generator_episode"])
            variant_name = meta["variant"]
            task_name, task_value = parse_variant(variant_name)

            print(
                f"[evaluator episode {evaluator_episode}] replaying cache "
                f"generator={generator_episode}, variant={variant_name}",
                flush=True,
            )

            replayed_sequences = []
            for episode_record in cache_payload["episodes"]:
                replayed = replay_agent_inputs(agent, episode_record["agent_inputs"])
                if evaluator_episode == generator_episode:
                    verify_hidden_preview(episode_record["generator_hidden_preview"], replayed)
                replayed_sequences.append(replayed)

            X_all, B_all = flatten_cached_replay(cache_payload["episodes"], replayed_sequences)
            X_all = X_all.to(device)
            B_all = tuple(part.to(device) for part in B_all)
            Xtr, Xte, Btr, Bte = shuffle_split_pairs(X_all, B_all, args.valid_size, device)

            pair_dir = pair_artifact_dir(artifact_root, generator_episode, variant_name, evaluator_episode)
            ensure_dir(pair_dir)

            row = {
                "train_id": args.train_id,
                "generator_episode": generator_episode,
                "evaluator_episode": int(evaluator_episode),
                "variant": variant_name,
                "task_name": task_name,
                "task_value": task_value,
                "num_samples_total": int(X_all.size(0)),
                "num_samples_train": int(Xtr.size(0)),
                "num_samples_eval": int(Xte.size(0)),
                "num_episodes": int(meta["num_episodes"]),
                "cache_path": cache_path,
            }

            pair_meta = {
                "train_id": args.train_id,
                "generator_episode": generator_episode,
                "evaluator_episode": int(evaluator_episode),
                "variant": variant_name,
                "task_name": task_name,
                "task_value": task_value,
                "cache_path": cache_path,
                "artifact_dir": pair_dir,
            }

            if args.run_mi:
                print(
                    f"[evaluator episode {evaluator_episode}] {variant_name} "
                    f"MINE start (generator={generator_episode})",
                    flush=True,
                )
                mine = build_mine(Xtr, Btr, args, device)
                mine.optimize(
                    Xtr,
                    Btr,
                    num_epochs=args.mine_num_epochs,
                    logger=make_offline_mine_logger(
                        evaluator_episode,
                        generator_episode,
                        variant_name,
                        args.mine_num_epochs,
                    ),
                    learning_rate=args.mine_learning_rate,
                    batch_size=args.mine_batch_size,
                    lambd=args.mine_lambda,
                    valid_size=args.valid_size,
                )
                mi_train = mine.estimate(Xtr, Btr)
                mi_eval = mine.estimate(Xte, Bte)
                row["MI_train"] = mi_train
                row["MI"] = mi_eval
                wandb.log({
                    "train/episode": int(evaluator_episode),
                    "task/variant": variant_name,
                    "task/generator_episode": generator_episode,
                    "mi/offline_train": mi_train,
                    "mi/offline": mi_eval,
                })
                save_mine_estimator(os.path.join(pair_dir, "mine.pth"), mine, Btr, args)

            if args.run_regression:
                for part_idx, (Ytr, Yte) in enumerate(zip(Btr, Bte)):
                    probe = fit_linear_probe_torch(
                        Xtr,
                        Ytr,
                        add_bias=True,
                        standardize=args.standardize,
                        use_float64=not args.no_float64,
                    )
                    rsq_te, _ = eval_linear_probe_torch(Xte, Yte, probe)
                    rsq_tr, _ = eval_linear_probe_torch(Xtr, Ytr, probe)
                    row[f"linreg_rsq-{part_idx}"] = rsq_te
                    row[f"linreg_rsq_train-{part_idx}"] = rsq_tr
                    wandb.log({
                        "train/episode": int(evaluator_episode),
                        "task/variant": variant_name,
                        "task/generator_episode": generator_episode,
                        f"linreg/rsq-{part_idx}": rsq_te,
                        f"linreg/rsq_train-{part_idx}": rsq_tr,
                    })
                    save_linreg_probe(os.path.join(pair_dir, f"linreg_b{part_idx}.pth"), probe)

            for spec in softmax_specs:
                spec_name = spec["name"]
                for part_idx, (Ytr, Yte) in enumerate(zip(Btr, Bte)):
                    state = fit_softmax_probe_torch(Xtr, Ytr, args, use_mlp_probe=spec["use_mlp"])
                    res_te = eval_softmax_probe_torch(Xte, Yte, state, standardize=args.standardize)
                    res_tr = eval_softmax_probe_torch(Xtr, Ytr, state, standardize=args.standardize)
                    add_softmax_metrics(row, res_tr, res_te, spec_name, part_idx)
                    log_softmax_metrics(
                        evaluator_episode,
                        generator_episode,
                        variant_name,
                        spec_name,
                        part_idx,
                        res_tr,
                        res_te,
                    )
                    save_softmax_probe(
                        os.path.join(pair_dir, f"{spec_name}_softmax_b{part_idx}.pth"),
                        state,
                        in_dim=int(Xtr.size(1)),
                        out_dim=int(Ytr.size(1)),
                        use_mlp=bool(spec["use_mlp"]),
                    )

            save_csv(os.path.join(pair_dir, "metrics.csv"), [row])
            save_json(os.path.join(pair_dir, "metadata.json"), pair_meta)
            summary_rows.append(row)
            per_eval_rows[int(evaluator_episode)].append(row)

    summary_csv_path = os.path.join(artifact_root, f"offline_replay_summary_{args.train_id}.csv")
    summary_json_path = os.path.join(artifact_root, f"offline_replay_summary_{args.train_id}.json")
    summary_excel_path = os.path.join(artifact_root, f"offline_replay_summary_{args.train_id}.xlsx")

    save_csv(summary_csv_path, summary_rows)
    save_json(summary_json_path, summary_rows)
    with pd.ExcelWriter(summary_excel_path, engine="openpyxl") as writer:
        for evaluator_episode, rows in per_eval_rows.items():
            if not rows:
                continue
            frame = pd.DataFrame(rows)
            frame = frame.sort_values(by=["generator_episode", "variant"], na_position="first")
            frame.to_excel(writer, sheet_name=f"eval_ep_{evaluator_episode}"[:31], index=False)

    print(f"Saved offline replay summary CSV: {summary_csv_path}", flush=True)
    print(f"Saved offline replay summary JSON: {summary_json_path}", flush=True)
    print(f"Saved offline replay summary Excel: {summary_excel_path}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    print("\n".join(f"{key}={value}" for key, value in vars(args).items()), flush=True)
    main(args)
