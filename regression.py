import wandb
import torch

from environments.tmaze import TMaze
from environments.hike import MountainHike
from environments.irrelevant import Irrelevant
from environments.starkweather import StarkweatherEnv

from agents.drqn import DRQN

from utils import generate_hiddens_and_beliefs, get_run_statistic

from argparse import ArgumentParser


def fit_linear_probe(X, Y, add_bias=True, standardize=True):
    """
    X: [N, H] hidden
    Y: [N, K] belief
    """
    device = X.device
    if standardize:
        mean = X.mean(0, keepdim=True)
        std = X.std(0, keepdim=True) + 1e-6
        Xn = (X - mean) / std
    else:
        mean, std = None, None
        Xn = X

    if add_bias:
        ones = torch.ones(Xn.size(0), 1, device=device)
        Xn = torch.cat([Xn, ones], dim=1)

    # W = (X^T X)^(-1) X^T Y
    res = torch.linalg.lstsq(Xn, Y)
    W = res.solution
    print(f"X: {Xn.shape}, Y: {Y.shape}, W: {W.shape}", flush = True)
    return {
        'W': W,
        'mean': mean,
        'std': std,
        'add_bias': add_bias,
    }

def eval_linear_probe(X, Y, probe):
    device = X.device
    if probe['mean'] is not None:
        Xn = (X - probe['mean']) / probe['std']
    else:
        Xn = X
    if probe['add_bias']:
        ones = torch.ones(Xn.size(0), 1, device=device)
        Xn = torch.cat([Xn, ones], dim=1)
    print(f"X: {Xn.shape}, Y: {Y.shape}", flush = True)
    Yhat = Xn @ probe['W']

    # R^2
    num = ((Y - Yhat) ** 2).sum()
    den = ((Y - Y.mean(0, keepdim=True)) ** 2).sum()
    rsq = 1 - (num / den)
    return rsq.item(), Yhat

def main(args):

    # Retrieve training arguments
    train_args = get_run_statistic(args.train_id)

    # Merge configurations
    config = vars(train_args) | vars(args)

    # Initialize logging
    wandb.init(
        project='belief-regression',
        name=args.name,
        config=config,
        save_code=True)
    config = wandb.config

    # Save all packages
    wandb.save('*.py')
    wandb.save('agents/*.py')
    wandb.save('environments/*.py')

    # Initialize environment
    if train_args.environment == 'tmaze':
        environment = TMaze(
            bayes=True,
            length=train_args.length,
            stochasticity=train_args.stochasticity)
    elif train_args.environment == 'hike':
        environment = MountainHike(
            bayes=True,
            variations=train_args.variations)
    elif config.environment == 'starkweather':
        environment = StarkweatherEnv(
            p_omission= train_args.p_omission,
            bin_size = train_args.bin_size,
            iti_hazard = train_args.iti_hazard,
            iti_min = train_args.iti_min,
            nITI_microstates = train_args.nITI_microstates,
        )
    else:
        environment = train_args.environment
        raise NotImplementedError(f'Unknown environment {environment}')

    # Add irrelevant variables
    if train_args.irrelevant != 0:
        environment = Irrelevant(environment, state_size=train_args.irrelevant,
                                 bayes=True)

    # Initialize agent
    if train_args.algorithm == 'drqn':
        network_kwargs = {
            'num_layers': train_args.num_layers,
            'hidden_size': train_args.hidden_size}

        agent = DRQN(
            cell=train_args.cell,
            action_size=environment.action_size,
            observation_size=environment.observation_size,
            **network_kwargs)
    else:
        raise NotImplementedError(f'Unknown algorithm {args.algorithm}')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    print('Device:', device)

    print(config.episodes)
    
    for episode in range(0, config.episodes + 1, args.mine_period):

        # 1. load agent checkpoint
        agent.load(args.train_id, episode=episode)
        print('agent loaded')

        # 2. sample data
        hiddens, beliefs = generate_hiddens_and_beliefs(
            agent,
            environment,
            num_samples=args.mine_num_samples,
            epsilon=args.epsilon,
            approximate=args.approximate,
        )
        print(f'generated hiddens of shape {hiddens.shape} and beliefs {beliefs}')

        # 3. move to device
        hiddens = hiddens.to(device)     # [N, H]
        beliefs = tuple(b.to(device) for b in beliefs)
        # 4. train / eval split
        N = hiddens.size(0)
        perm = torch.randperm(N, device=device)
        hiddens = hiddens[perm]
        beliefs = tuple(b[perm] for b in beliefs)
        split = int(N * 0.8)
        X_train = hiddens[:split]
        X_test  = hiddens[split:]

        for part_idx, belief_part in enumerate(beliefs):

            Y_train = belief_part[:split]
            Y_test  = belief_part[split:]

            # 5. fit on train
            probe = fit_linear_probe(
                X_train,
                Y_train,
                standardize=True,
                add_bias=True,
            )

            # 6. eval on test
            rsq, _ = eval_linear_probe(X_test, Y_test, probe)

            key = f"regression/rsq-{part_idx}"
            wandb.log({'train/episode': episode, key: rsq})
            print(f"[episode {episode}] {key} = {rsq:.4f}")

    wandb.finish()



if __name__ == '__main__':

    parser = ArgumentParser(
        description='Estimate MI for a certain training session',
    )
    parser.add_argument('name', type=str, nargs='?', default=None)

    # Run id
    parser.add_argument('train_id', type=str)

    # MINE estimator
    parser.add_argument('--mine_num_samples', type=int, default=10000)
    parser.add_argument('--mine_num_layers', type=int, default=2)
    parser.add_argument('--mine_hidden_size', type=int, default=256)
    parser.add_argument('--mine_alpha', type=float, default=0.01)
    parser.add_argument('--mine_num_epochs', type=int, default=100)
    parser.add_argument('--mine_batch_size', type=int, default=1024)
    parser.add_argument('--mine_learning_rate', type=float, default=1e-3)
    parser.add_argument('--mine_lambda', type=float, default=0.0)
    parser.add_argument('--mine_period', type=int, default=100)
    parser.add_argument('--representation_size', type=int, default=16)
    parser.add_argument('--belief_part', type=int, default=None)

    parser.add_argument('--valid_size', type=float, default=0.2)
    parser.add_argument('--train_set', action='store_true')
    parser.add_argument('--approximate', action='store_true')

    parser.add_argument('--epsilon', type=float, default=0.0)

    # Parse command line arguments
    args = parser.parse_args()
    print('\n'.join(f'\033[90m{k}=\033[0m{v}' for k, v in vars(args).items()))

    main(args)
