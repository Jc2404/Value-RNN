import wandb
import torch
from argparse import ArgumentParser

from mine.mine import MutualInformationNeuralEstimator
from utils import generate_hiddens_and_beliefs, get_run_statistic
from probe.build_env import select_device, build_environment, build_agent
from probe.mine_io import MineConfig, save_mine_config


def main(args):
    train_args = get_run_statistic(args.train_id)

    device = select_device()
    print("Device:", device)

    env = build_environment(train_args)
    agent = build_agent(train_args, env)

    if args.use_wandb:
        config = vars(train_args) | vars(args)
        wandb.init(project=args.wandb_project, name=args.name, config=config, save_code=True)

    # Train estimator at each selected agent checkpoint
    for agent_episode in range(args.agent_episode_start, args.agent_episode_end + 1, args.agent_episode_step):
        agent.load(args.train_id, episode=agent_episode)
        print(f"Loaded agent {args.train_id} @ episode {agent_episode}")

        hiddens, beliefs = generate_hiddens_and_beliefs(
            agent, env,
            num_samples=args.mine_num_samples,
            epsilon=args.epsilon,
            approximate=args.approximate,
        )

        if len(beliefs) == 1 and args.belief_part is not None:
            raise ValueError("belief_part was set, but beliefs has length 1 (no parts).")

        # Determine sizes (consistent with your original code)
        belief_sizes = []
        representation_sizes = []
        for belief_part_tensor in beliefs:
            belief_sizes.append(belief_part_tensor.size(-1))
            if belief_part_tensor.ndim == 2:
                representation_sizes.append(None)
            elif belief_part_tensor.ndim == 3:
                representation_sizes.append(args.representation_size)
            else:
                raise ValueError("Expected belief tensors to have 2 or 3 dims.")

        mine = MutualInformationNeuralEstimator(
            hs_sizes=hiddens.size(-1),
            belief_sizes=belief_sizes,
            hidden_size=args.mine_hidden_size,
            num_layers=args.mine_num_layers,
            alpha=args.mine_alpha,
            representation_sizes=representation_sizes,
            belief_part=args.belief_part,
            device=device,
        )

        logger = wandb.log if args.use_wandb else (lambda _: None)

        mine.optimize(
            hiddens=hiddens,
            beliefs=beliefs,
            num_epochs=args.mine_num_epochs,
            logger=logger,
            learning_rate=args.mine_learning_rate,
            batch_size=args.mine_batch_size,
            lambd=args.mine_lambda,
            valid_size=args.valid_size,
        )

        # Save estimator weights
        mine_id = args.mine_id
        mine.save(mine_id, episode=agent_episode)

        # Save a manifest/config so evaluation can recreate the estimator correctly
        cfg = MineConfig(
            hs_sizes=hiddens.size(-1),
            belief_sizes=belief_sizes,
            representation_sizes=representation_sizes,
            hidden_size=args.mine_hidden_size,
            num_layers=args.mine_num_layers,
            alpha=args.mine_alpha,
            belief_part=args.belief_part,
        )
        save_mine_config(mine_id, agent_episode, cfg)
        print(f"Saved MINE: weights/{mine_id}-{agent_episode}-T.pth and cfg json")

        # Optional: evaluate immediately on a fresh sample (still “training script” but clearly separated)
        if args.eval_after_train:
            h_eval, b_eval = generate_hiddens_and_beliefs(
                agent, env,
                num_samples=args.mine_num_samples,
                epsilon=args.epsilon,
                approximate=args.approximate,
            )
            mi = mine.estimate(h_eval, b_eval)
            print(f"Episode {agent_episode}: MI={mi}")
            if args.use_wandb:
                key = "mine_train/mi" if args.belief_part is None else f"mine_train/mi-{args.belief_part}"
                wandb.log({"agent_episode": agent_episode, key: mi})

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    p = ArgumentParser("Train (fit) MINE estimator(s) and save to disk.")
    p.add_argument("train_id", type=str)
    p.add_argument("--name", type=str, default=None)

    # Where to save estimator weights (matches your save/load convention)
    p.add_argument("--mine_id", type=str, required=True, help="Identifier used in weights/{mine_id}-{episode}-T.pth")

    # Which agent checkpoints to train on
    p.add_argument("--agent_episode_start", type=int, default=0)
    p.add_argument("--agent_episode_end", type=int, default=0)
    p.add_argument("--agent_episode_step", type=int, default=100)

    # MINE training hyperparams
    p.add_argument("--mine_num_samples", type=int, default=10000)
    p.add_argument("--mine_num_layers", type=int, default=2)
    p.add_argument("--mine_hidden_size", type=int, default=256)
    p.add_argument("--mine_alpha", type=float, default=0.01)
    p.add_argument("--mine_num_epochs", type=int, default=200)
    p.add_argument("--mine_batch_size", type=int, default=1024)
    p.add_argument("--mine_learning_rate", type=float, default=1e-3)
    p.add_argument("--mine_lambda", type=float, default=0.0)
    p.add_argument("--valid_size", type=float, default=0.2)

    p.add_argument("--representation_size", type=int, default=16)
    p.add_argument("--belief_part", type=int, default=None)

    p.add_argument("--approximate", action="store_true")
    p.add_argument("--epsilon", type=float, default=0.0)

    # Logging and options
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="belief-mi-ref")
    p.add_argument("--eval_after_train", action="store_true")

    args = p.parse_args()
    main(args)
