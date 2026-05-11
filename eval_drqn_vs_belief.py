import os
import json
import csv
import random
from argparse import ArgumentParser
from typing import Callable, Dict, List, Tuple

import torch

from agents.drqn import DRQN
try:
    from agents.classic_belief import BeliefPolicy
except ImportError:
    from classic_belief import BeliefPolicy
from agents.memory import Trajectory
from belief_comparison import evaluate_agent_against_belief

from environments.tmaze import TMaze
from environments.tiger import Tiger
from environments.crybaby import CryingBaby
from environments.irrelevant import Irrelevant
from environments.starkweather import StarkweatherEnv
from environments.gridworld import GridWorld
from utils import get_run_statistic


def _resolve_arg(args, train_args, name: str, fallback=None):
    value = getattr(args, name, None)
    if value is not None:
        return value
    if hasattr(train_args, name):
        return getattr(train_args, name)
    return fallback


def make_environment_factory(args, train_args) -> Callable[[], object]:
    environment_name = args.environment or train_args.environment

    if environment_name == 'tmaze':
        return lambda: TMaze(
            length=_resolve_arg(args, train_args, 'length', 20),
            stochasticity=_resolve_arg(args, train_args, 'stochasticity', 0.0),
            bayes=True,
        )
    if environment_name == 'tiger':
        return lambda: Tiger(
            listen_accuracy=_resolve_arg(args, train_args, 'listen_accuracy', 0.85),
            reward_listen=_resolve_arg(args, train_args, 'reward_listen', -1.0),
            reward_correct=_resolve_arg(args, train_args, 'reward_correct', 10.0),
            reward_wrong=_resolve_arg(args, train_args, 'reward_wrong', -100.0),
            horizon=_resolve_arg(args, train_args, 'horizon', 20),
            bayes=True,
        )
    if environment_name == 'crybaby':
        return lambda: CryingBaby(
            p_hungry_if_full_wait=_resolve_arg(args, train_args, 'p_hungry_if_full_wait', 0.10),
            p_stay_hungry_wait=_resolve_arg(args, train_args, 'p_stay_hungry_wait', 0.90),
            p_full_if_feed=_resolve_arg(args, train_args, 'p_full_if_feed', 0.95),
            p_cry_if_hungry=_resolve_arg(args, train_args, 'p_cry_if_hungry', 0.90),
            p_cry_if_full=_resolve_arg(args, train_args, 'p_cry_if_full', 0.10),
            p0_hungry=_resolve_arg(args, train_args, 'p0_hungry', 0.50),
            reward_cry=_resolve_arg(args, train_args, 'reward_cry', -1.0),
            cost_feed=_resolve_arg(args, train_args, 'cost_feed', -0.2),
            reward_quiet=_resolve_arg(args, train_args, 'reward_quiet', 0.0),
            horizon=_resolve_arg(args, train_args, 'horizon', 50),
            bayes=True,
        )
    if environment_name == 'starkweather':
        return lambda: StarkweatherEnv(
            bayes=True,
            p_omission=_resolve_arg(args, train_args, 'p_omission', 0.1),
            bin_size=_resolve_arg(args, train_args, 'bin_size', 0.2),
            iti_hazard=_resolve_arg(args, train_args, 'iti_hazard', 1 / 65.0),
            iti_min=_resolve_arg(args, train_args, 'iti_min', 0),
            nITI_microstates=_resolve_arg(args, train_args, 'nITI_microstates', 10),
            max_steps=_resolve_arg(args, train_args, 'max_steps', 200),
        )
    if environment_name == 'gridworld':
        return lambda: GridWorld(
            bayes=True,
            size=_resolve_arg(args, train_args, 'size', 10),
            tprob=_resolve_arg(args, train_args, 'tprob', 0.7),
            discount=_resolve_arg(args, train_args, 'discount', 0.95),
            max_steps=_resolve_arg(args, train_args, 'max_steps', 200),
            reward_scheme=_resolve_arg(args, train_args, 'reward_scheme', 'julia'),
            reward_margin=_resolve_arg(args, train_args, 'reward_margin', 2),
            step_cost=_resolve_arg(args, train_args, 'step_cost', 0.0),
        )
    raise ValueError(f"Unsupported environment for belief planner: {environment_name}")


def make_wrapped_environment_factory(args, train_args) -> Callable[[], object]:
    base_factory = make_environment_factory(args, train_args)

    def _factory():
        env = base_factory()
        irrelevant_size = _resolve_arg(args, train_args, 'irrelevant', 0)
        if irrelevant_size:
            env = Irrelevant(env, state_size=irrelevant_size, bayes=True)
        return env

    return _factory


def build_agent_for_episode(args, train_args, env, episode: int):
    algorithm = _resolve_arg(args, train_args, 'algorithm', 'drqn')
    if algorithm != 'drqn':
        raise NotImplementedError(f"Unsupported algorithm for belief comparison: {algorithm}")

    agent = DRQN(
        cell=_resolve_arg(args, train_args, 'cell', 'gru'),
        action_size=env.action_size,
        observation_size=env.observation_size,
        hidden_size=_resolve_arg(args, train_args, 'hidden_size', 32),
        num_layers=_resolve_arg(args, train_args, 'num_layers', 2),
    )
    agent.load(args.run_id, episode=episode, weights_dir=args.weights_dir)
    agent.Q.eval()
    agent.Q_tar.eval()
    return agent


def _move_hidden_to_device(hidden_states, device):
    if hidden_states is None:
        return None
    if isinstance(hidden_states, (tuple, list)):
        return tuple(h.to(device) for h in hidden_states)
    return hidden_states.to(device)


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




def save_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            f.write('')
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_eval_episodes(args) -> List[int]:
    if args.episodes_list:
        vals = [int(x.strip()) for x in args.episodes_list.split(',') if x.strip()]
        if not vals:
            raise ValueError("--episodes-list was provided but no valid episodes were parsed.")
        return vals

    if args.max_episode is None:
        return [int(args.episode)]

    start = int(args.start_episode)
    end = int(args.max_episode)
    period = int(args.period)
    if period <= 0:
        raise ValueError("--period must be positive.")
    if end < start:
        raise ValueError("--max-episode must be >= --start-episode.")
    return list(range(start, end + 1, period))


def evaluate_single_checkpoint(args, train_args, env_factory, episode: int) -> Tuple[Dict, List[Dict], List[Dict]]:
    env_for_shapes = env_factory()
    agent = build_agent_for_episode(args, train_args, env_for_shapes, episode)
    summary, per_episode, per_step = evaluate_agent_against_belief(
        agent,
        env_factory,
        args.total_steps,
        epsilon=args.epsilon,
        planning_horizon=args.planning_horizon,
        belief_round_ndigits=args.belief_round_ndigits,
    )
    summary.update({
        'run_id': args.run_id,
        'agent_episode': int(episode),
        'environment': args.environment,
    })

    for row in per_episode:
        row['agent_episode'] = int(episode)
    for row in per_step:
        row['agent_episode'] = int(episode)

    return summary, per_episode, per_step


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_args = get_run_statistic(args.run_id)
    train_environment = getattr(train_args, 'environment', None)
    if args.environment is not None and train_environment is not None and args.environment != train_environment:
        raise ValueError(
            f"Environment mismatch: checkpoint run {args.run_id} was trained on "
            f"{train_environment}, but evaluator was asked to use {args.environment}."
        )

    if args.environment is None:
        args.environment = train_environment

    env_factory = make_wrapped_environment_factory(args, train_args)
    eval_episodes = get_eval_episodes(args)

    all_summaries: List[Dict] = []
    all_per_episode: List[Dict] = []
    all_per_step: List[Dict] = []

    os.makedirs(args.output_dir, exist_ok=True)

    for ckpt_episode in eval_episodes:
        summary, per_episode, per_step = evaluate_single_checkpoint(args, train_args, env_factory, ckpt_episode)
        all_summaries.append(summary)
        all_per_episode.extend(per_episode)
        all_per_step.extend(per_step)
        print(json.dumps(summary, indent=2))

    summary_json_path = os.path.join(args.output_dir, f'{args.run_id}_{args.output_prefix}_all_summaries.json')
    summary_csv_path = os.path.join(args.output_dir, f'{args.run_id}_{args.output_prefix}_summary_table.csv')
    per_episode_path = os.path.join(args.output_dir, f'{args.run_id}_{args.output_prefix}_per_episode.csv')
    per_step_path = os.path.join(args.output_dir, f'{args.run_id}_{args.output_prefix}_per_step.csv')

    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2)
    save_csv(summary_csv_path, all_summaries)
    save_csv(per_episode_path, all_per_episode)
    save_csv(per_step_path, all_per_step)

    print(f"Saved checkpoint summaries to: {summary_json_path}")
    print(f"Saved checkpoint summary table to: {summary_csv_path}")
    print(f"Saved per-episode metrics to: {per_episode_path}")
    print(f"Saved per-step metrics to: {per_step_path}")


if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluate DRQN checkpoints against belief forward search planner using a total-step budget.')

    parser.add_argument('--run-id', type=str, required=True)
    parser.add_argument('--episode', type=int, default=None,
                        help='Single checkpoint episode to evaluate when --max-episode is not provided.')
    parser.add_argument('--start-episode', type=int, default=0)
    parser.add_argument('--max-episode', type=int, default=None)
    parser.add_argument('--period', type=int, default=500)
    parser.add_argument('--episodes-list', type=str, default=None,
                        help='Optional comma-separated explicit checkpoint list, e.g. 0,500,1000')

    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--output-prefix', type=str, default='drqn_vs_belief_period')
    parser.add_argument('--weights-dir', dest='weights_dir', type=str, default='weights')
    parser.add_argument('--total-steps', type=int, default=5000)
    parser.add_argument('--epsilon', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--cell', type=str, default=None,
                        help='Optional architecture override. Defaults to the saved training config.')
    parser.add_argument('--hidden-size', type=int, default=None,
                        help='Optional architecture override. Defaults to the saved training config.')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Optional architecture override. Defaults to the saved training config.')
    parser.add_argument('--irrelevant', type=int, default=None,
                        help='Optional irrelevant-wrapper override. Defaults to the saved training config.')

    parser.add_argument('--planning-horizon', type=int, default=None)
    parser.add_argument('--belief-round-ndigits', type=int, default=10)

    subparsers = parser.add_subparsers(dest='environment', required=False)

    tmaze = subparsers.add_parser('tmaze')
    tmaze.add_argument('--length', type=int, default=None)
    tmaze.add_argument('--stochasticity', type=float, default=None)

    tiger = subparsers.add_parser('tiger')
    tiger.add_argument('--listen-accuracy', type=float, default=None)
    tiger.add_argument('--reward-listen', type=float, default=None)
    tiger.add_argument('--reward-correct', type=float, default=None)
    tiger.add_argument('--reward-wrong', type=float, default=None)
    tiger.add_argument('--horizon', type=int, default=None)

    crybaby = subparsers.add_parser('crybaby')
    crybaby.add_argument('--p_hungry_if_full_wait', type=float, default=None)
    crybaby.add_argument('--p_stay_hungry_wait', type=float, default=None)
    crybaby.add_argument('--p_full_if_feed', type=float, default=None)
    crybaby.add_argument('--p_cry_if_hungry', type=float, default=None)
    crybaby.add_argument('--p_cry_if_full', type=float, default=None)
    crybaby.add_argument('--p0_hungry', type=float, default=None)
    crybaby.add_argument('--reward_cry', type=float, default=None)
    crybaby.add_argument('--reward_quiet', type=float, default=None)
    crybaby.add_argument('--cost_feed', type=float, default=None)
    crybaby.add_argument('--horizon', type=int, default=None)

    stark = subparsers.add_parser('starkweather')
    stark.add_argument('--p_omission', type=float, default=None)
    stark.add_argument('--bin_size', type=float, default=None)
    stark.add_argument('--iti_hazard', type=float, default=None)
    stark.add_argument('--iti_min', type=float, default=None)
    stark.add_argument('--nITI_microstates', type=int, default=None)
    stark.add_argument('--max_steps', type=int, default=None)

    grid = subparsers.add_parser('gridworld')
    grid.add_argument('--size', type=int, default=None)
    grid.add_argument('--tprob', type=float, default=None)
    grid.add_argument('--discount', type=float, default=None)
    grid.add_argument('--max_steps', type=int, default=None)
    grid.add_argument('--reward_scheme', type=str, default=None)
    grid.add_argument('--reward_margin', type=int, default=None)
    grid.add_argument('--step_cost', type=float, default=None)

    args = parser.parse_args()
    if args.max_episode is None and args.episode is None and args.episodes_list is None:
        parser.error('Provide either --episode, --max-episode/--period, or --episodes-list.')
    main(args)
