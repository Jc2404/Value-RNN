import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

from environments.tmaze import TMaze
from environments.hike import MountainHike
from environments.irrelevant import Irrelevant

from agents.drqn import DRQN

from utils import get_run_statistic

from argparse import ArgumentParser

def generate_hiddens_and_beliefs(agent, environment, num_samples, epsilon=0.2,
                                 approximate=True, include_trajectory=True):
    """
    Samples joint hidden states and beliefs using an epsilon-greedy policy
    based on the agent to sample the trajectories in the environment.

    Arguments
    - agent: Agent
        The agent whose epsilon-greedy policy is used.
    - environment: Environment
        The environment in which the trajectories are generated.
    - num_samples: int
        The number of hidden states and beliefs to generate.
    - epsilon: float
        The exploration rate of the epsilon-greedy policy.
    - approximate: bool
        Whether to use a faster approximation of the distributions where all
        beliefs and hidden states generated in any trajectory are returned.

    Returns
    - hiddens: tensor
        a batch of hidden states from the successive trajectories.
    - beliefs: tuple of tensors
        a batch of beliefs from the successive beliefs.
    """
    hiddens, beliefs, trajectory = [], [], []
    while len(hiddens) < num_samples:
        tra, hh, bb = agent.play(
            environment,
            epsilon=epsilon,
            return_hiddens=True,
            return_beliefs=True,
        )
    
        # TODO: this should be modified to sample a time step first
        # TODO: this should be modified to allow sampling past terminal states
        tra_list = tra.observed
        for i in tra_list: trajectory.append(i)
        if approximate:
            for h, b in zip(hh, bb):
                hiddens.append(h)
                beliefs.append([bi for bi in b])
                
        else:
            t = torch.randint(len(hh), ()).item()
            hiddens.append(hh[t])
            beliefs.append([bi for bi in bb[t]])
    print(f'sampled {num_samples}', flush = True)

    hiddens = hiddens[:num_samples]
    beliefs = beliefs[:num_samples]

    tuple_of_beliefs = [list() for _ in range(len(beliefs[0]))]
    for belief in beliefs:
        for i, b in enumerate(belief):
            tuple_of_beliefs[i].append(b)
    print('beliefs generated', flush = True)

    if include_trajectory:
        return torch.stack(hiddens), tuple(map(torch.stack, tuple_of_beliefs)), trajectory

    return torch.stack(hiddens), tuple(map(torch.stack, tuple_of_beliefs))

def main(args):
    log_file = args.log_path
    train_args = get_run_statistic(args.train_id)
    config = vars(train_args) | vars(args)
    environment = TMaze(
        bayes = True,
        length = train_args.length,
        stochasticity=train_args.stochasticity,
    )

    if train_args.irrelevant != 0:
        environment = Irrelevant(
            environment,
            state_size=train_args.irrelevant,
            bayes=True,
        )

    network_kwargs = {
        'num_layers': train_args.num_layers,
        'hidden_size': train_args.hidden_size,
    }
    agent = DRQN(
        cell=train_args.cell,
        action_size=environment.action_size,
        observation_size=environment.observation_size,
        **network_kwargs,
    )

    for episode in range(0, args.episodes + 1, args.mine_period):

        # load agent checkpoint
        agent.load(args.train_id, episode=episode)
        print('agent loaded')
        with open(log_file, "a") as f:
            f.write("\n==== Network weights ====\n")
            for name, p in agent.Q.named_parameters():
                w = p.detach().cpu()
                f.write(f"\n{name}  shape={tuple(w.shape)}\n")
                f.write(str(w.numpy()))
                f.write("\n")

        # sample hidden states + belief tuple
        hiddens, beliefs, trajectories = generate_hiddens_and_beliefs(
            agent,
            environment,
            num_samples=args.mine_num_samples,
            epsilon=args.epsilon,
            approximate=args.approximate,
        )
        print(f'generated hiddens: {hiddens.shape}, beliefs: {[b.shape for b in beliefs]}, traj: {len(trajectories)}')
        L = train_args.length
        

        for i in range(len(beliefs[0])):
            Maze = [["x"]* (L+1) for _ in range(7)]
            Maze[1][:] = beliefs[0][i][:L+1].tolist()
            Maze[0][-1] = beliefs[0][i][L+1].item()
            Maze[2][-1] = beliefs[0][i][L+2].item()
            Maze[5][:] = beliefs[0][i][L+3:-2].tolist()
            Maze[4][-1] = beliefs[0][i][-2].item()
            Maze[6][-1] = beliefs[0][i][-1].item()

            trajectory = trajectories[i]
            action = trajectory[:4]
            obs = trajectory[4:-1]
            reward = trajectory[-1]

            with open(log_file, "a") as f:
                f.write(f"\n==== Episode {episode} Step {i} ====\n")
                f.write("\nBeliefs \n")
                for line in Maze:
                    f.write(f"{line}\n")
                f.write("r u l d -- actions \n")
                f.write(f"{action}\n")
                f.write("u, d, c, j -- observations \n")
                f.write(f"{obs}\n")
                f.write(f"reward {reward}\n")


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Probe RNN beliefs with softmax + KL',
    )
    parser.add_argument('train_id', type=str)
    parser.add_argument('--mine_num_samples', type=int, default=10000)
    parser.add_argument('--mine_period', type=int, default=100)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--log_path', type=str, default="res.txt")
    parser.add_argument('--approximate', action='store_true')
    parser.add_argument('--epsilon', type=float, default=0.0)


    args = parser.parse_args()
    main(args)