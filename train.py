import csv
import json
import os

import wandb

from argparse import ArgumentParser

from environments.tmaze import TMaze
from environments.hike import MountainHike
from environments.irrelevant import Irrelevant
from environments.starkweather import StarkweatherEnv
from environments.tiger import Tiger
from environments.gridworld import GridWorld
from environments.crybaby import CryingBaby
from agents.drqn import DRQN


def save_rows(path, rows):
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(args):

    # Initialize logging
    wandb.init(
        project='belief-train_reproduction',
        name=args.name,
        config=args,
        save_code=True)
    config = wandb.config
    metrics_rows = []

    # Save all packages
    wandb.save('agents/*.py')
    wandb.save('environments/*.py')
    wandb.save('mine/*.py')

    # Initialize environment
    if config.environment == 'tmaze':
        environment = TMaze(
            length=config.length,
            stochasticity=config.stochasticity,
            bayes=False,
        )
    elif config.environment == 'hike':
        environment = MountainHike(
            variations=config.variations,
            bayes=False,
        )
    elif config.environment == 'starkweather':
        environment = StarkweatherEnv(
            p_omission=config.p_omission,
            bin_size = config.bin_size,
            iti_hazard = config.iti_hazard,
            iti_min = config.iti_min,
            nITI_microstates = config.nITI_microstates,
        )
    elif config.environment == 'tiger':
        environment = Tiger(
            listen_accuracy=config.listen_accuracy,
            reward_listen=config.reward_listen,
            reward_correct=config.reward_correct,
            reward_wrong=config.reward_wrong,
            horizon=config.horizon,
            bayes=False,
        )
    elif config.environment == 'gridworld':
        environment = GridWorld(
            size=config.size,
            tprob=config.tprob,
            reward_scheme=config.reward_scheme,
            reward_margin=config.reward_margin,
            step_cost=config.step_cost,
            bayes=False,
        )
    elif config.environment == 'crybaby':
        environment = CryingBaby(
            p_hungry_if_full_wait=config.p_hungry_if_full_wait,
            p_stay_hungry_wait=config.p_stay_hungry_wait,
            p_full_if_feed=config.p_full_if_feed,
            p_cry_if_hungry=config.p_cry_if_hungry,
            p_cry_if_full=config.p_cry_if_full,
            reward_cry=config.reward_cry,
            cost_feed=config.cost_feed,
            bayes=False,
        )
    else:
        raise NotImplementedError(f'Unknown environment {config.environment}')

    # Add irrelevant variables
    if config.irrelevant != 0:
        environment = Irrelevant(
            environment,
            state_size=config.irrelevant,
            bayes=False,
        )

    # Initialise agent
    if config.algorithm == 'drqn':
        network_kwargs = {
            'num_layers': config.num_layers,
            'hidden_size': config.hidden_size}

        agent = DRQN(
            cell=config.cell,
            action_size=environment.action_size,
            observation_size=environment.observation_size,
            **network_kwargs)
    else:
        raise NotImplementedError(f'Unknown algorithm {config.algorithm}')

    # Load weights
    if config.load is not None:
        agent.load(config.load, episode=config.load_at, weights_dir=config.weights_dir)

    def logger(payload):
        wandb.log(payload)
        metrics_rows.append(dict(payload))

    # Train agent
    agent.train(
        environment,
        wandb.run.id,
        logger,
        num_episodes=config.num_episodes,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_gradient_steps=config.num_gradient_steps,
        target_period=config.target_period,
        eval_period=config.eval_period,
        num_rollouts=config.num_rollouts,
        epsilon=config.epsilon,
        buffer_capacity=config.buffer_capacity,
        weights_dir=config.weights_dir,
    )

    if config.results_dir:
        os.makedirs(config.results_dir, exist_ok=True)
        run_info_path = os.path.join(config.results_dir, 'train_run_info.json')
        with open(run_info_path, 'w', encoding='utf-8') as f:
            json.dump({
                'run_id': wandb.run.id,
                'run_name': wandb.run.name,
                'weights_dir': config.weights_dir,
                'results_dir': config.results_dir,
                'args': vars(args),
            }, f, indent=2)

        if metrics_rows:
            metrics_path = os.path.join(config.results_dir, f'train_metrics_{wandb.run.id}.csv')
            save_rows(metrics_path, metrics_rows)


if __name__ == '__main__':

    parser = ArgumentParser(
        description='Experiments hidden states and beliefs',
    )
    parser.add_argument('positional_name', type=str, nargs='?', default=None)
    parser.add_argument('--name', dest='name', type=str, default=None)

    # Architecture
    parser.add_argument('-C', '--cell', type=str, default='gru')
    parser.add_argument('-H', '--hidden_size', type=int, default=32)
    parser.add_argument('-S', '--num_layers', type=int, default=2)

    # Retrain
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--load-at', type=int, default=None)
    parser.add_argument('--weights-dir', dest='weights_dir', type=str, default='weights')
    parser.add_argument('--results-dir', dest='results_dir', type=str, default=None)

    # Evaluation
    parser.add_argument('--eval-period', type=int, default=5)
    parser.add_argument('--num-rollouts', type=int, default=50)

    # Algorithm
    parser.add_argument('--algorithm', type=str, default='drqn')
    parser.add_argument('-E', '--num-episodes', type=int, default=5000)
    parser.add_argument('-B', '--batch-size', type=int, default=32)
    parser.add_argument('-a', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('-I', '--num-gradient-steps', type=int, default=10)
    parser.add_argument('-U', '--target-period', type=int, default=10)
    parser.add_argument('-e', '--epsilon', type=float, default=0.2)
    parser.add_argument('-R', '--buffer-capacity', type=int, default=8192)

    # Environment modification
    parser.add_argument('--irrelevant', type=int, default=0)

    # Environment
    environment_subparser = parser.add_subparsers(
        title='environment', dest='environment', required=True)

    # Environment: T-Maze
    tmaze = environment_subparser.add_parser('tmaze')
    tmaze.add_argument('--length', type=int, default=20)
    tmaze.add_argument('--stochasticity', type=float, default=0.0)

    # Environment: Mountain Hike
    hike = environment_subparser.add_parser('hike')
    hike.add_argument('--variations', type=str, default=None)

    # Environment: StarkWeather
    starkweather = environment_subparser.add_parser('starkweather')
    starkweather.add_argument('--p_omission', type = float, default = 0.1)
    starkweather.add_argument('--bin_size', type = float, default = 0.2)
    starkweather.add_argument('--iti_hazard', type = float, default = 1/65)
    starkweather.add_argument('--iti_min', type = float, default = 0)
    starkweather.add_argument('--nITI_microstates', type = int, default = 10)

    # Environment: Tiger
    tiger = environment_subparser.add_parser('tiger')
    tiger.add_argument('--listen-accuracy', type=float, default=0.85)
    tiger.add_argument('--reward-listen', type=float, default=-1.0)
    tiger.add_argument('--reward-correct', type=float, default=10.0)
    tiger.add_argument('--reward-wrong', type=float, default=-100.0)
    tiger.add_argument('--horizon', type=int, default=20)

    # Environment: GridWorld
    gridworld = environment_subparser.add_parser('gridworld')
    gridworld.add_argument('--size', type=int, default=10)
    gridworld.add_argument('--tprob', type=float, default=0.7)
    gridworld.add_argument('--reward-scheme', type=str, default='julia')
    gridworld.add_argument('--reward-margin', type=int, default=2)
    gridworld.add_argument('--step-cost', type=float, default=0.0)

    # Environment: Crying Baby
    crybaby = environment_subparser.add_parser('crybaby')
    crybaby.add_argument('--p_hungry_if_full_wait', type=float, default=0.10)
    crybaby.add_argument('--p_stay_hungry_wait', type=float, default=0.90)
    crybaby.add_argument('--p_full_if_feed', type=float, default=0.95)
    crybaby.add_argument('--p_cry_if_hungry', type=float, default=0.90)
    crybaby.add_argument('--p_cry_if_full', type=float, default=0.10)
    crybaby.add_argument('--reward_cry', type=float, default=-1.0)
    crybaby.add_argument('--cost_feed', type=float, default=-0.2)

    # Parse command line arguments
    args = parser.parse_args()
    if args.name is None:
        args.name = args.positional_name
    del args.positional_name
    print('\n'.join(f'\033[90m{k}=\033[0m{v}' for k, v in vars(args).items()))

    main(args)
