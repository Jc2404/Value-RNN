# regression_eval.py
import os
from argparse import ArgumentParser

import torch
import wandb

from environments.tmaze import TMaze
from environments.hike import MountainHike
from environments.irrelevant import Irrelevant
from environments.starkweather import StarkweatherEnv
# from environments.tiger import Tiger

from agents.drqn import DRQN
from utils import generate_hiddens_and_beliefs, get_run_statistic


def eval_linear_probe(X, Y, probe):
    device = X.device
    Xn = X
    if probe["mean"] is not None:
        mean = probe["mean"].to(device)
        std = probe["std"].to(device)
        Xn = (Xn - mean) / std

    if probe["add_bias"]:
        ones = torch.ones(Xn.size(0), 1, device=device)
        Xn = torch.cat([Xn, ones], dim=1)

    W = probe["W"].to(device)
    Yhat = Xn @ W

    num = ((Y - Yhat) ** 2).sum()
    den = ((Y - Y.mean(0, keepdim=True)) ** 2).sum()
    rsq = 1 - (num / den)

    mse = torch.mean((Y - Yhat) ** 2)
    return rsq.item(), mse.item()


def build_environment(train_args, override_env=None, override_kwargs=None, bayes=True):
    """
    Default: build env matching the training run.
    Optionally: override_env + override_kwargs to evaluate on different settings.
    """
    override_kwargs = override_kwargs or {}

    env_name = override_env if override_env is not None else train_args.environment

    if env_name == "tmaze":
        # If overriding, use override_kwargs; otherwise use train_args
        length = override_kwargs.get("length", getattr(train_args, "length", 10))
        stochasticity = override_kwargs.get("stochasticity", getattr(train_args, "stochasticity", 0.0))
        env = TMaze(bayes=bayes, length=length, stochasticity=stochasticity)

    elif env_name == "hike":
        variations = override_kwargs.get("variations", getattr(train_args, "variations", 1))
        env = MountainHike(bayes=bayes, variations=variations)

    elif env_name == "starkweather":
        env = StarkweatherEnv(
            p_omission=override_kwargs.get("p_omission", getattr(train_args, "p_omission")),
            bin_size=override_kwargs.get("bin_size", getattr(train_args, "bin_size")),
            iti_hazard=override_kwargs.get("iti_hazard", getattr(train_args, "iti_hazard")),
            iti_min=override_kwargs.get("iti_min", getattr(train_args, "iti_min")),
            nITI_microstates=override_kwargs.get("nITI_microstates", getattr(train_args, "nITI_microstates")),
        )

    else:
        raise NotImplementedError(f"Unknown environment {env_name}")

    irrelevant = override_kwargs.get("irrelevant", getattr(train_args, "irrelevant", 0))
    if irrelevant != 0:
        env = Irrelevant(env, state_size=irrelevant, bayes=bayes)

    return env


def build_agent(train_args, environment):
    if train_args.algorithm != "drqn":
        raise NotImplementedError(f"Unknown algorithm {train_args.algorithm}")

    agent = DRQN(
        cell=train_args.cell,
        action_size=environment.action_size,
        observation_size=environment.observation_size,
        num_layers=train_args.num_layers,
        hidden_size=train_args.hidden_size,
    )
    return agent


def resolve_probe_path(args, train_id):
    if args.probe_path is not None:
        return args.probe_path

    if args.probe_dir is None:
        raise ValueError("Provide --probe_path or --probe_dir")

    # expected pattern used by regression_train.py
    path = os.path.join(args.probe_dir, f"probe_{train_id}_ep{args.probe_episode:06d}_part{args.part_idx}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Probe file not found: {path}")
    return path


def main(args):
    train_args = get_run_statistic(args.train_id)

    # Parse override kwargs (we keep it simple; add fields as needed)
    override_kwargs = {}
    if args.override_length is not None:
        override_kwargs["length"] = args.override_length
    if args.override_stochasticity is not None:
        override_kwargs["stochasticity"] = args.override_stochasticity
    if args.override_irrelevant is not None:
        override_kwargs["irrelevant"] = args.override_irrelevant

    # Build env/agent for evaluation
    environment = build_environment(
        train_args,
        override_env=args.eval_environment,
        override_kwargs=override_kwargs,
        bayes=True,
    )
    agent = build_agent(train_args, environment)

    # Load agent checkpoint
    agent.load(args.train_id, episode=args.agent_episode)

    # Load probe
    probe_path = resolve_probe_path(args, args.train_id)
    probe = torch.load(probe_path, map_location="cpu")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # wandb
    config = vars(train_args) | vars(args) | {
        "probe_path": probe_path,
        "eval_environment_effective": args.eval_environment or train_args.environment,
        "eval_environment_overrides": override_kwargs,
    }
    wandb.init(
        project="belief-regression",
        name=args.name,
        config=config,
        save_code=True,
    )

    # Sample data
    hiddens, beliefs = generate_hiddens_and_beliefs(
        agent,
        environment,
        num_samples=args.num_samples,
        epsilon=args.epsilon,
        approximate=args.approximate,
    )

    hiddens = hiddens.to(device)
    beliefs = tuple(b.to(device) for b in beliefs)

    if args.part_idx < 0 or args.part_idx >= len(beliefs):
        raise ValueError(f"part_idx={args.part_idx} out of range; beliefs parts={len(beliefs)}")

    Y = beliefs[args.part_idx]

    rsq, mse = eval_linear_probe(hiddens, Y, probe)

    wandb.log({
        "eval/agent_episode": args.agent_episode,
        "eval/rsq": rsq,
        "eval/mse": mse,
    })

    print(f"[eval] probe={os.path.basename(probe_path)} part={args.part_idx} rsq={rsq:.4f} mse={mse:.6f}")
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate a saved linear probe on an agent/environment")

    parser.add_argument("name", type=str, nargs="?", default=None)
    parser.add_argument("train_id", type=str)

    # Which agent checkpoint to evaluate
    parser.add_argument("--agent_episode", type=int, default=0)

    # Probe selection
    parser.add_argument("--part_idx", type=int, default=0)
    parser.add_argument("--probe_path", type=str, default=None)
    parser.add_argument("--probe_dir", type=str, default=None)
    parser.add_argument("--probe_episode", type=int, default=0)

    # Data sampling
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--approximate", action="store_true")

    # Optional: evaluate on different environment settings than the original run
    parser.add_argument("--eval_environment", type=str, default=None,
                        help="Override env name (e.g., tmaze, hike, starkweather)")

    # A couple common overrides (extend as you like)
    parser.add_argument("--override_length", type=int, default=None)
    parser.add_argument("--override_stochasticity", type=float, default=None)
    parser.add_argument("--override_irrelevant", type=int, default=None)

    args = parser.parse_args()
    print("\n".join(f"\033[90m{k}=\033[0m{v}" for k, v in vars(args).items()))
    main(args)
