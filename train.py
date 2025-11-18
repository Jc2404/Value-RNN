import wandb

from argparse import ArgumentParser

from environments.tmaze import TMaze
from environments.hike import MountainHike
from environments.irrelevant import Irrelevant
from environments.starkweather import StarkweatherEnv

from agents.drqn import DRQN


def main(args):

    # Initialize logging
    wandb.init(
        project='belief-train_reproduction',
        name=args.name,
        config=args,
        save_code=True)
    config = wandb.config

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
        agent.load(config.load, episode=config.load_at)

    # Train agent
    agent.train(
        environment,
        wandb.run.id,
        wandb.log,
        num_episodes=config.num_episodes,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_gradient_steps=config.num_gradient_steps,
        target_period=config.target_period,
        eval_period=config.eval_period,
        num_rollouts=config.num_rollouts,
        epsilon=config.epsilon,
        buffer_capacity=config.buffer_capacity,
    )


if __name__ == '__main__':

    parser = ArgumentParser(
        description='Experiments hidden states and beliefs',
    )
    parser.add_argument('name', type=str, nargs='?', default=None)

    # Architecture
    parser.add_argument('-C', '--cell', type=str, default='gru')
    parser.add_argument('-H', '--hidden_size', type=int, default=32)
    parser.add_argument('-S', '--num_layers', type=int, default=2)

    # Retrain
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--load-at', type=int, default=None)

    # Evaluation
    parser.add_argument('--eval-period', type=int, default=10)
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

    # Parse command line arguments
    args = parser.parse_args()
    print('\n'.join(f'\033[90m{k}=\033[0m{v}' for k, v in vars(args).items()))

    main(args)
