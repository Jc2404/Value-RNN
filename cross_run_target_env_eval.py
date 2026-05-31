import json
import os
import random
from argparse import ArgumentParser

torch = None
BeliefPolicy = None
DRQN = None
assert_planner_supported_environment = None
eval_mean_returns_with_step_budget = None
evaluate_belief_comparison_metrics = None
rollout_drqn_episode = None
rollout_planner_episode = None
build_environment = None
build_mine = None
eval_softmax_probe_torch = None
fit_softmax_probe_torch = None
save_csv = None
select_device = None
shuffle_split_tensors = None
generate_hiddens_and_beliefs = None
get_run_statistic = None


def import_runtime_dependencies() -> None:
    global torch
    global BeliefPolicy
    global DRQN
    global assert_planner_supported_environment
    global eval_mean_returns_with_step_budget
    global evaluate_belief_comparison_metrics
    global rollout_drqn_episode
    global rollout_planner_episode
    global build_environment
    global build_mine
    global eval_softmax_probe_torch
    global fit_softmax_probe_torch
    global save_csv
    global select_device
    global shuffle_split_tensors
    global generate_hiddens_and_beliefs
    global get_run_statistic

    if torch is not None:
        return

    import torch as torch_module
    from agents.classic_belief import BeliefPolicy as belief_policy_cls
    from agents.drqn import DRQN as drqn_cls
    from belief_comparison import (
        assert_planner_supported_environment as assert_planner_supported_environment_fn,
        eval_mean_returns_with_step_budget as eval_mean_returns_with_step_budget_fn,
        evaluate_belief_comparison_metrics as evaluate_belief_comparison_metrics_fn,
        rollout_drqn_episode as rollout_drqn_episode_fn,
        rollout_planner_episode as rollout_planner_episode_fn,
    )
    from retrain_decode_eval import (
        build_environment as build_environment_fn,
        build_mine as build_mine_fn,
        eval_softmax_probe_torch as eval_softmax_probe_torch_fn,
        fit_softmax_probe_torch as fit_softmax_probe_torch_fn,
        save_csv as save_csv_fn,
        select_device as select_device_fn,
        shuffle_split_tensors as shuffle_split_tensors_fn,
    )
    from utils import (
        generate_hiddens_and_beliefs as generate_hiddens_and_beliefs_fn,
        get_run_statistic as get_run_statistic_fn,
    )

    torch = torch_module
    BeliefPolicy = belief_policy_cls
    DRQN = drqn_cls
    assert_planner_supported_environment = assert_planner_supported_environment_fn
    eval_mean_returns_with_step_budget = eval_mean_returns_with_step_budget_fn
    evaluate_belief_comparison_metrics = evaluate_belief_comparison_metrics_fn
    rollout_drqn_episode = rollout_drqn_episode_fn
    rollout_planner_episode = rollout_planner_episode_fn
    build_environment = build_environment_fn
    build_mine = build_mine_fn
    eval_softmax_probe_torch = eval_softmax_probe_torch_fn
    fit_softmax_probe_torch = fit_softmax_probe_torch_fn
    save_csv = save_csv_fn
    select_device = select_device_fn
    shuffle_split_tensors = shuffle_split_tensors_fn
    generate_hiddens_and_beliefs = generate_hiddens_and_beliefs_fn
    get_run_statistic = get_run_statistic_fn


def parse_episode_list(value: str) -> list[int]:
    if value is None:
        raise ValueError("--episodes-a is required.")

    episodes = []
    for chunk in value.split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        episodes.append(int(stripped))

    if not episodes:
        raise ValueError("--episodes-a must contain at least one episode.")
    if len(set(episodes)) != len(episodes):
        raise ValueError("--episodes-a must not contain duplicate episodes.")
    return episodes


def validate_args(args) -> None:
    args.episodes_a = parse_episode_list(args.episodes_a)

    if not (0.0 < args.valid_size < 1.0):
        raise ValueError("--valid-size must be strictly between 0 and 1.")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.return_total_steps <= 0:
        raise ValueError("--return-total-steps must be positive.")
    if args.probe_epochs <= 0:
        raise ValueError("--probe-epochs must be positive.")
    if args.mine_num_epochs <= 0:
        raise ValueError("--mine-num-epochs must be positive.")
    if args.probe_batch_size <= 0:
        raise ValueError("--probe-batch-size must be positive.")
    if args.mine_batch_size <= 0:
        raise ValueError("--mine-batch-size must be positive.")


def make_base_env_factory(train_args):
    return lambda: build_environment(train_args, overrides=None)


def build_agent(train_args, environment, device):
    if train_args.algorithm != "drqn":
        raise NotImplementedError(f"Unsupported algorithm for cross-run target-env eval: {train_args.algorithm}")

    agent = DRQN(
        cell=train_args.cell,
        action_size=environment.action_size,
        observation_size=environment.observation_size,
        num_layers=train_args.num_layers,
        hidden_size=train_args.hidden_size,
    )
    agent.Q.to(device)
    agent.Q_tar.to(device)
    return agent


def ensure_compatible_spaces(source_label: str, source_env, target_env) -> None:
    if source_env.action_size != target_env.action_size:
        raise ValueError(
            f"{source_label} action-size mismatch with target env: "
            f"{source_env.action_size} vs {target_env.action_size}."
        )
    if source_env.observation_size != target_env.observation_size:
        raise ValueError(
            f"{source_label} observation-size mismatch with target env: "
            f"{source_env.observation_size} vs {target_env.observation_size}."
        )


def to_device_beliefs(beliefs, device):
    return tuple(part.to(device) for part in beliefs)


def estimate_planner_returns(planner, env_factory, total_steps: int) -> dict:
    planner_mean_return, planner_mean_disc_return, planner_eval_episodes, planner_eval_steps = (
        eval_mean_returns_with_step_budget(
            rollout_fn=lambda env: rollout_planner_episode(planner, env, epsilon=0.0),
            env_factory=env_factory,
            total_steps=total_steps,
        )
    )
    return {
        "planner_mean_return": float(planner_mean_return),
        "planner_mean_disc_return": float(planner_mean_disc_return),
        "planner_eval_num_episodes": int(planner_eval_episodes),
        "planner_eval_total_steps": int(planner_eval_steps),
    }


def estimate_drqn_returns(agent, env_factory, total_steps: int) -> dict:
    drqn_mean_return, drqn_mean_disc_return, drqn_eval_episodes, drqn_eval_steps = (
        eval_mean_returns_with_step_budget(
            rollout_fn=lambda env: rollout_drqn_episode(agent, env, epsilon=0.0),
            env_factory=env_factory,
            total_steps=total_steps,
        )
    )
    return {
        "drqn_mean_return": float(drqn_mean_return),
        "drqn_mean_disc_return": float(drqn_mean_disc_return),
        "drqn_eval_num_episodes": int(drqn_eval_episodes),
        "drqn_eval_total_steps": int(drqn_eval_steps),
    }


def fit_mine_metrics(agent, env_factory, args, device) -> tuple[float, float]:
    train_env = env_factory()
    h_train, b_train = generate_hiddens_and_beliefs(
        agent,
        train_env,
        num_samples=args.num_samples,
        epsilon=args.epsilon,
        approximate=args.approximate,
    )
    h_train = h_train.to(device)
    b_train = to_device_beliefs(b_train, device)

    if len(b_train) == 1 and args.belief_part is not None:
        raise ValueError("belief_part was set, but the target environment exposes only one belief part.")

    eval_env = env_factory()
    h_eval, b_eval = generate_hiddens_and_beliefs(
        agent,
        eval_env,
        num_samples=args.num_samples,
        epsilon=args.epsilon,
        approximate=args.approximate,
    )
    h_eval = h_eval.to(device)
    b_eval = to_device_beliefs(b_eval, device)

    mine = build_mine(h_train, b_train, args, device)
    mine.optimize(
        h_train,
        b_train,
        num_epochs=args.mine_num_epochs,
        logger=lambda _: None,
        learning_rate=args.mine_learning_rate,
        batch_size=args.mine_batch_size,
        lambd=args.mine_lambda,
        valid_size=args.valid_size,
    )
    mi_train = float(mine.estimate(h_train, b_train))
    mi_eval = float(mine.estimate(h_eval, b_eval))
    return mi_train, mi_eval


def fit_softmax_metrics(agent, env_factory, args, device) -> dict:
    sample_env = env_factory()
    hiddens, beliefs = generate_hiddens_and_beliefs(
        agent,
        sample_env,
        num_samples=args.num_samples,
        epsilon=args.epsilon,
        approximate=args.approximate,
    )
    hiddens = hiddens.to(device)
    beliefs = to_device_beliefs(beliefs, device)

    x_train, x_valid, beliefs_train, beliefs_valid = shuffle_split_tensors(
        hiddens,
        beliefs,
        args.valid_size,
        device,
    )

    metrics = {}
    for part_idx, (y_train, y_valid) in enumerate(zip(beliefs_train, beliefs_valid)):
        linear_state = fit_softmax_probe_torch(x_train, y_train, args, use_mlp_probe=False)
        linear_train = eval_softmax_probe_torch(x_train, y_train, linear_state, standardize=args.standardize)
        linear_valid = eval_softmax_probe_torch(x_valid, y_valid, linear_state, standardize=args.standardize)

        metrics[f"softmax_linear_KL_b{part_idx}"] = float(linear_valid["kl"])
        metrics[f"softmax_linear_CE_b{part_idx}"] = float(linear_valid["ce"])
        metrics[f"softmax_linear_H_true_b{part_idx}"] = float(linear_valid["true_entropy"])
        metrics[f"softmax_linear_H_pred_b{part_idx}"] = float(linear_valid["pred_entropy"])
        metrics[f"softmax_linear_JS_b{part_idx}"] = float(linear_valid["js"])
        metrics[f"softmax_linear_KL_train_b{part_idx}"] = float(linear_train["kl"])
        metrics[f"softmax_linear_CE_train_b{part_idx}"] = float(linear_train["ce"])
        metrics[f"softmax_linear_H_true_train_b{part_idx}"] = float(linear_train["true_entropy"])
        metrics[f"softmax_linear_H_pred_train_b{part_idx}"] = float(linear_train["pred_entropy"])
        metrics[f"softmax_linear_JS_train_b{part_idx}"] = float(linear_train["js"])

        mlp_state = fit_softmax_probe_torch(x_train, y_train, args, use_mlp_probe=True)
        mlp_train = eval_softmax_probe_torch(x_train, y_train, mlp_state, standardize=args.standardize)
        mlp_valid = eval_softmax_probe_torch(x_valid, y_valid, mlp_state, standardize=args.standardize)

        metrics[f"softmax_mlp_KL_b{part_idx}"] = float(mlp_valid["kl"])
        metrics[f"softmax_mlp_CE_b{part_idx}"] = float(mlp_valid["ce"])
        metrics[f"softmax_mlp_H_true_b{part_idx}"] = float(mlp_valid["true_entropy"])
        metrics[f"softmax_mlp_H_pred_b{part_idx}"] = float(mlp_valid["pred_entropy"])
        metrics[f"softmax_mlp_JS_b{part_idx}"] = float(mlp_valid["js"])
        metrics[f"softmax_mlp_KL_train_b{part_idx}"] = float(mlp_train["kl"])
        metrics[f"softmax_mlp_CE_train_b{part_idx}"] = float(mlp_train["ce"])
        metrics[f"softmax_mlp_H_true_train_b{part_idx}"] = float(mlp_train["true_entropy"])
        metrics[f"softmax_mlp_H_pred_train_b{part_idx}"] = float(mlp_train["pred_entropy"])
        metrics[f"softmax_mlp_JS_train_b{part_idx}"] = float(mlp_train["js"])

    return metrics


def fit_optional_belief_comparison(agent, planner, env_factory, args) -> dict:
    compare_summary, _, _ = evaluate_belief_comparison_metrics(
        agent=agent,
        planner=planner,
        env_factory=env_factory,
        total_steps=args.return_total_steps,
        epsilon=args.belief_eval_epsilon,
    )
    return {
        "metric_2_step_weighted_action_agreement_rate": float(compare_summary["step_weighted_agreement_rate"]),
        "metric_3_step_weighted_mean_regret": float(compare_summary["step_weighted_mean_regret"]),
        "metric_3_step_weighted_mean_discounted_regret": float(
            compare_summary["step_weighted_mean_discounted_regret"]
        ),
        "metric_2_step_weighted_q_mse": float(compare_summary["metric_2_step_weighted_q_mse"]),
        "metric_2_step_weighted_q_mae": float(compare_summary["metric_2_step_weighted_q_mae"]),
        "metric_2_step_weighted_q_chosen_action_mse": float(
            compare_summary["metric_2_step_weighted_q_chosen_action_mse"]
        ),
        "comparison_num_episodes": int(compare_summary["num_episodes"]),
        "comparison_total_executed_steps": int(compare_summary["total_executed_steps"]),
        "comparison_mean_episode_regret": float(compare_summary["mean_episode_regret"]),
        "comparison_mean_discounted_episode_regret": float(compare_summary["mean_discounted_episode_regret"]),
    }


def evaluate_checkpoint(source_spec, target_train_args, planner, planner_metrics, args, device) -> dict:
    env_factory = make_base_env_factory(target_train_args)
    source_env = build_environment(source_spec["train_args"], overrides=None)

    agent = build_agent(source_spec["train_args"], source_env, device)
    agent.load(source_spec["run_id"], episode=source_spec["episode"], weights_dir=source_spec["weights_dir"])
    agent.Q.eval()
    agent.Q_tar.eval()

    return_metrics = estimate_drqn_returns(agent, env_factory, args.return_total_steps)
    mi_train, mi_eval = fit_mine_metrics(agent, env_factory, args, device)
    decoder_metrics = fit_softmax_metrics(agent, env_factory, args, device)

    summary = {
        "source_bucket": source_spec["source_bucket"],
        "source_run_id": source_spec["run_id"],
        "source_weights_dir": source_spec["weights_dir"],
        "source_environment": source_spec["train_args"].environment,
        "agent_episode": int(source_spec["episode"]),
        "target_run_id": args.run_id_b,
        "target_weights_dir": args.weights_dir_b,
        "target_environment": target_train_args.environment,
        "MI_train": float(mi_train),
        "MI": float(mi_eval),
        **return_metrics,
        **planner_metrics,
        "disc_return_gap_planner_minus_drqn": float(
            planner_metrics["planner_mean_disc_return"] - return_metrics["drqn_mean_disc_return"]
        ),
    }
    summary.update(decoder_metrics)

    if args.include_belief_comparison:
        summary.update(fit_optional_belief_comparison(agent, planner, env_factory, args))

    return summary


def build_source_specs(args, run_a_args, run_b_args) -> list[dict]:
    specs = []
    for episode in args.episodes_a:
        specs.append(
            {
                "source_bucket": "A",
                "run_id": args.run_id_a,
                "weights_dir": args.weights_dir_a,
                "episode": int(episode),
                "train_args": run_a_args,
            }
        )
    specs.append(
        {
            "source_bucket": "B",
            "run_id": args.run_id_b,
            "weights_dir": args.weights_dir_b,
            "episode": int(args.episode_b),
            "train_args": run_b_args,
        }
    )
    return specs


def main(args) -> None:
    validate_args(args)
    args.standardize = True
    import_runtime_dependencies()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device()
    print(f"Device: {device}", flush=True)

    print("Loading source run metadata...", flush=True)
    run_a_args = get_run_statistic(args.run_id_a)
    run_b_args = get_run_statistic(args.run_id_b)

    target_env = build_environment(run_b_args, overrides=None)
    source_env_a = build_environment(run_a_args, overrides=None)
    source_env_b = build_environment(run_b_args, overrides=None)

    assert_planner_supported_environment(target_env)
    ensure_compatible_spaces("run A", source_env_a, target_env)
    ensure_compatible_spaces("run B", source_env_b, target_env)

    env_factory = make_base_env_factory(run_b_args)
    planner = BeliefPolicy(
        planning_horizon=args.planning_horizon,
        belief_round_ndigits=args.belief_round_ndigits,
    )

    print("Evaluating shared belief-planner benchmark on target environment...", flush=True)
    planner_metrics = estimate_planner_returns(planner, env_factory, args.return_total_steps)

    source_specs = build_source_specs(args, run_a_args, run_b_args)
    summaries = []
    for source_spec in source_specs:
        print(
            f"Evaluating source bucket {source_spec['source_bucket']} "
            f"run={source_spec['run_id']} episode={source_spec['episode']} in target env {run_b_args.environment}",
            flush=True,
        )
        summary = evaluate_checkpoint(source_spec, run_b_args, planner, planner_metrics, args, device)
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    os.makedirs(args.report_dir, exist_ok=True)
    base_name = args.name if args.name is not None else "cross_run_target_env_eval"
    summary_json_path = os.path.join(args.report_dir, f"{base_name}_all_summaries.json")
    summary_csv_path = os.path.join(args.report_dir, f"{base_name}_summary_table.csv")

    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    save_csv(summary_csv_path, summaries)

    print(f"Saved summary JSON: {summary_json_path}", flush=True)
    print(f"Saved summary CSV: {summary_csv_path}", flush=True)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Compare checkpoints from two runs in run B's base training environment."
    )
    parser.add_argument("name", type=str, nargs="?", default=None)

    parser.add_argument("--run-id-a", type=str, required=True)
    parser.add_argument("--weights-dir-a", type=str, required=True)
    parser.add_argument("--episodes-a", type=str, required=True)

    parser.add_argument("--run-id-b", type=str, required=True)
    parser.add_argument("--weights-dir-b", type=str, required=True)
    parser.add_argument("--episode-b", type=int, required=True)

    parser.add_argument("--report-dir", type=str, required=True)

    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--approximate", action="store_true")

    parser.add_argument("--probe-epochs", type=int, default=300)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-batch-size", type=int, default=1024)
    parser.add_argument("--belief-loss", choices=["kl", "mse"], default="kl")
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--mlp-dropout", type=float, default=0.0)

    parser.add_argument("--mine-num-layers", type=int, default=2)
    parser.add_argument("--mine-hidden-size", type=int, default=256)
    parser.add_argument("--mine-alpha", type=float, default=0.01)
    parser.add_argument("--mine-num-epochs", type=int, default=400)
    parser.add_argument("--mine-batch-size", type=int, default=1024)
    parser.add_argument("--mine-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mine-lambda", type=float, default=0.0)
    parser.add_argument("--representation-size", type=int, default=16)
    parser.add_argument("--belief-part", type=int, default=None)

    parser.add_argument("--return-total-steps", type=int, default=5000)
    parser.add_argument("--planning-horizon", type=int, default=None)
    parser.add_argument("--belief-round-ndigits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--include-belief-comparison", action="store_true")
    parser.add_argument("--belief-eval-epsilon", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
