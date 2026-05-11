import random
from typing import Callable, Dict, List, Tuple

import torch

from agents.classic_belief import BeliefPolicy
from agents.memory import Trajectory
from environments.starkweather import StarkweatherEnv


def _move_hidden_to_device(hidden_states, device):
    if hidden_states is None:
        return None
    if isinstance(hidden_states, (tuple, list)):
        return tuple(h.to(device) for h in hidden_states)
    return hidden_states.to(device)


def _unwrap_environment(env):
    while hasattr(env, "environment"):
        env = env.environment
    return env


def planner_supported_environment(env) -> bool:
    base_env = _unwrap_environment(env)
    name = base_env.__class__.__name__.lower()
    return name in {"tiger", "cryingbaby", "tmaze", "starkweatherenv", "gridworld"}


def assert_planner_supported_environment(env):
    if planner_supported_environment(env):
        return

    base_env = _unwrap_environment(env)
    env_name = base_env.__class__.__name__
    raise NotImplementedError(
        "Belief comparison currently supports TMaze, Tiger, CryingBaby, "
        f"StarkweatherEnv, and GridWorld. Got {env_name}."
    )


def rollout_drqn_episode(agent, env, epsilon: float = 0.0) -> Tuple[Trajectory, int]:
    trajectory, = agent.play(env, epsilon=epsilon)
    return trajectory, int(trajectory.num_transitions)


def rollout_planner_episode(planner, env, epsilon: float = 0.0) -> Tuple[Trajectory, int]:
    trajectory, = planner.play(env, epsilon=epsilon)
    return trajectory, int(trajectory.num_transitions)


def eval_mean_returns_with_step_budget(rollout_fn, env_factory, total_steps: int) -> Tuple[float, float, int, int]:
    sum_returns = 0.0
    sum_disc_returns = 0.0
    episodes = 0
    steps = 0

    while steps < total_steps:
        env = env_factory()
        trajectory, ep_steps = rollout_fn(env)
        if ep_steps <= 0:
            raise RuntimeError("Encountered an episode with zero transitions; cannot use step budget reliably.")

        sum_returns += float(trajectory.get_cumulative_reward())
        sum_disc_returns += float(trajectory.get_cumulative_reward(env.gamma))
        episodes += 1
        steps += ep_steps

    mean_return = sum_returns / max(episodes, 1)
    mean_disc_return = sum_disc_returns / max(episodes, 1)
    return mean_return, mean_disc_return, episodes, steps


def evaluate_belief_comparison_metrics(agent, planner, env_factory, total_steps: int, epsilon: float = 0.0):
    device = next(agent.Q.parameters()).device

    per_episode: List[Dict] = []
    per_step: List[Dict] = []

    total_executed_steps = 0
    episode_idx = 0
    global_agreement_sum = 0
    global_regret_sum = 0.0
    global_discounted_regret_sum = 0.0
    global_q_mse_sum = 0.0
    global_q_mae_sum = 0.0
    global_q_chosen_action_mse_sum = 0.0

    while total_executed_steps < total_steps:
        env = env_factory()
        planner._prepare_environment(env)
        planner._reset_cache(env)

        obs = env.reset()
        trajectory = Trajectory(env.action_size, env.observation_size)
        trajectory.add(None, None, obs)

        hidden_states = None
        n_steps = 0
        agreement_sum = 0
        regret_sum = 0.0
        discounted_regret_sum = 0.0

        for t in range(env.horizon()):
            tau_t = trajectory.get_last_observed().view(1, 1, -1).to(device)
            hidden_states = _move_hidden_to_device(hidden_states, device)
            with torch.no_grad():
                drqn_values, hidden_states = agent.Q(tau_t, hidden_states)
            drqn_q = drqn_values.flatten().detach().cpu()

            if random.random() < epsilon:
                drqn_action = env.exploration()
            else:
                drqn_action = int(torch.argmax(drqn_q).item())

            belief = planner.extract_planning_belief(env).detach().clone().float()
            remaining_steps = planner._remaining_steps(env, t)
            planner_q = planner.q_values(env, belief, remaining_steps).detach().cpu()
            planner_action = int(torch.argmax(planner_q).item())

            agreement = int(drqn_action == planner_action)
            regret = float(torch.max(planner_q).item() - planner_q[drqn_action].item())

            q_error = drqn_q - planner_q
            q_mse = float((q_error ** 2).mean().item())
            q_mae = float(q_error.abs().mean().item())
            q_chosen_action_mse = float((drqn_q[drqn_action] - planner_q[drqn_action]) ** 2)

            agreement_sum += agreement
            regret_sum += regret
            discounted_regret_sum += (float(env.gamma) ** t) * regret
            n_steps += 1

            global_agreement_sum += agreement
            global_regret_sum += regret
            global_discounted_regret_sum += (float(env.gamma) ** t) * regret
            global_q_mse_sum += q_mse
            global_q_mae_sum += q_mae
            global_q_chosen_action_mse_sum += q_chosen_action_mse
            total_executed_steps += 1

            per_step.append({
                "episode": episode_idx,
                "timestep": t,
                "global_step": total_executed_steps,
                "remaining_steps": remaining_steps,
                "drqn_action": drqn_action,
                "planner_action": planner_action,
                "action_agreement": agreement,
                "step_regret": regret,
                "planner_q_max": float(torch.max(planner_q).item()),
                "planner_q_drqn_action": float(planner_q[drqn_action].item()),
                "drqn_q_max": float(torch.max(drqn_q).item()),
                "drqn_q_chosen_action": float(drqn_q[drqn_action].item()),
                "planner_q_mse": q_mse,
                "planner_q_mae": q_mae,
                "planner_q_chosen_action_mse": q_chosen_action_mse,
            })

            obs, reward, done = env.step(drqn_action)
            trajectory.add(drqn_action, reward, obs, terminal=done)

            if done:
                break

        ep_return = float(trajectory.get_cumulative_reward())
        ep_disc_return = float(trajectory.get_cumulative_reward(env.gamma))
        ep_agreement = agreement_sum / max(n_steps, 1)
        ep_mean_step_regret = regret_sum / max(n_steps, 1)

        per_episode.append({
            "episode": episode_idx,
            "num_steps": n_steps,
            "drqn_return": ep_return,
            "drqn_disc_return": ep_disc_return,
            "action_agreement_rate": ep_agreement,
            "episode_regret": regret_sum,
            "discounted_episode_regret": discounted_regret_sum,
            "mean_step_regret": ep_mean_step_regret,
        })
        episode_idx += 1

    num_episodes = len(per_episode)
    mean_episode_regret = sum(x["episode_regret"] for x in per_episode) / max(num_episodes, 1)
    mean_discounted_episode_regret = sum(x["discounted_episode_regret"] for x in per_episode) / max(num_episodes, 1)
    mean_drqn_return = sum(x["drqn_return"] for x in per_episode) / max(num_episodes, 1)
    mean_drqn_disc_return = sum(x["drqn_disc_return"] for x in per_episode) / max(num_episodes, 1)

    step_weighted_agreement = global_agreement_sum / max(total_executed_steps, 1)
    step_weighted_mean_regret = global_regret_sum / max(total_executed_steps, 1)
    step_weighted_mean_discounted_regret = global_discounted_regret_sum / max(total_executed_steps, 1)
    step_weighted_q_mse = global_q_mse_sum / max(total_executed_steps, 1)
    step_weighted_q_mae = global_q_mae_sum / max(total_executed_steps, 1)
    step_weighted_q_chosen_action_mse = global_q_chosen_action_mse_sum / max(total_executed_steps, 1)

    return {
        "total_executed_steps": total_executed_steps,
        "num_episodes": num_episodes,
        "step_weighted_agreement_rate": step_weighted_agreement,
        "step_weighted_mean_regret": step_weighted_mean_regret,
        "step_weighted_mean_discounted_regret": step_weighted_mean_discounted_regret,
        "metric_2_step_weighted_q_mse": step_weighted_q_mse,
        "metric_2_step_weighted_q_mae": step_weighted_q_mae,
        "metric_2_step_weighted_q_chosen_action_mse": step_weighted_q_chosen_action_mse,
        "mean_episode_regret": mean_episode_regret,
        "mean_discounted_episode_regret": mean_discounted_episode_regret,
        "mean_drqn_return_from_comparison_rollouts": mean_drqn_return,
        "mean_drqn_disc_return_from_comparison_rollouts": mean_drqn_disc_return,
    }, per_episode, per_step


def evaluate_agent_against_belief(
    agent,
    env_factory: Callable[[], object],
    total_steps: int,
    *,
    epsilon: float = 0.0,
    planning_horizon=None,
    belief_round_ndigits: int = 10,
    progress_prefix: str | None = None,
):
    def progress(message):
        if progress_prefix:
            print(f"{progress_prefix} {message}", flush=True)

    env_for_checks = env_factory()
    assert_planner_supported_environment(env_for_checks)

    planner = BeliefPolicy(
        planning_horizon=planning_horizon,
        belief_round_ndigits=belief_round_ndigits,
    )

    progress(f"belief eval phase 1/2: planner return eval start (step_budget={total_steps})")
    planner_mean_return, planner_mean_disc_return, planner_eval_episodes, planner_eval_steps = eval_mean_returns_with_step_budget(
        rollout_fn=lambda env: rollout_planner_episode(planner, env, epsilon=0.0),
        env_factory=env_factory,
        total_steps=total_steps,
    )
    progress(
        "belief eval phase 1/2 done: "
        f"episodes={planner_eval_episodes}, total_steps={planner_eval_steps}, "
        f"disc_return={planner_mean_disc_return:.6f}"
    )

    progress(
        "belief eval phase 2/2: comparison metrics start "
        f"(step_budget={total_steps}, epsilon={epsilon})"
    )
    compare_summary, per_episode, per_step = evaluate_belief_comparison_metrics(
        agent=agent,
        planner=planner,
        env_factory=env_factory,
        total_steps=total_steps,
        epsilon=epsilon,
    )
    progress(
        "belief eval phase 2/2 done: "
        f"episodes={compare_summary['num_episodes']}, total_steps={compare_summary['total_executed_steps']}, "
        f"agreement={compare_summary['step_weighted_agreement_rate']:.6f}"
    )

    summary = {
        "total_steps_budget": int(total_steps),
        "planner_horizon": planning_horizon,
        "belief_round_ndigits": int(belief_round_ndigits),
        "metric_1_planner_mean_return": float(planner_mean_return),
        "metric_1_planner_mean_disc_return": float(planner_mean_disc_return),
        "metric_1_planner_eval_num_episodes": int(planner_eval_episodes),
        "metric_1_planner_eval_total_steps": int(planner_eval_steps),
        "metric_1_return_gap_planner_minus_drqn": float(planner_mean_return - compare_summary["mean_drqn_return_from_comparison_rollouts"]),
        "metric_1_disc_return_gap_planner_minus_drqn": float(planner_mean_disc_return - compare_summary["mean_drqn_disc_return_from_comparison_rollouts"]),
        "metric_2_step_weighted_action_agreement_rate": float(compare_summary["step_weighted_agreement_rate"]),
        "metric_3_step_weighted_mean_regret": float(compare_summary["step_weighted_mean_regret"]),
        "metric_3_step_weighted_mean_discounted_regret": float(compare_summary["step_weighted_mean_discounted_regret"]),
        "metric_2_step_weighted_q_mse": float(compare_summary["metric_2_step_weighted_q_mse"]),
        "metric_2_step_weighted_q_mae": float(compare_summary["metric_2_step_weighted_q_mae"]),
        "metric_2_step_weighted_q_chosen_action_mse": float(compare_summary["metric_2_step_weighted_q_chosen_action_mse"]),
        "comparison_num_episodes": int(compare_summary["num_episodes"]),
        "comparison_total_executed_steps": int(compare_summary["total_executed_steps"]),
        "comparison_mean_episode_regret": float(compare_summary["mean_episode_regret"]),
        "comparison_mean_discounted_episode_regret": float(compare_summary["mean_discounted_episode_regret"]),
        "comparison_rollout_mean_drqn_return": float(compare_summary["mean_drqn_return_from_comparison_rollouts"]),
        "comparison_rollout_mean_drqn_disc_return": float(compare_summary["mean_drqn_disc_return_from_comparison_rollouts"]),
    }

    return summary, per_episode, per_step
