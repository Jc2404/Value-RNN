# regression_train.py
import os
from argparse import ArgumentParser

import torch
import wandb

from environments.tmaze import TMaze
from environments.hike import MountainHike
from environments.irrelevant import Irrelevant
from environments.starkweather import StarkweatherEnv
# If you add Tiger later:
# from environments.tiger import Tiger

from agents.drqn import DRQN
from utils import generate_hiddens_and_beliefs, get_run_statistic


def fit_linear_probe(X, Y, add_bias=True, standardize=True):
    """
    X: [N, H] hidden
    Y: [N, K] belief
    Returns a dict suitable for torch.save(...)
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

    # W = argmin ||XW - Y||_2 via least squares
    res = torch.linalg.lstsq(Xn, Y)
    W = res.solution
    return {
        "W": W.detach().cpu(),
        "mean": None if mean is None else mean.detach().cpu(),
        "std": None if std is None else std.detach().cpu(),
        "add_bias": bool(add_bias),
        "standardize": bool(standardize),
        "x_dim": int(X.size(1)),
        "y_dim": int(Y.size(1)),
    }


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

    # R^2 (global across dims)
    num = ((Y - Yhat) ** 2).sum()
    den = ((Y - Y.mean(0, keepdim=True)) ** 2).sum()
    rsq = 1 - (num / den)

    mse = torch.mean((Y - Yhat) ** 2)
    return rsq.item(), mse.item(), Yhat


def build_environment_from_train_args(train_args, bayes=True):
    if train_args.environment == "tmaze":
        env = TMaze(
            bayes=bayes,
            length=train_args.length,
            stochasticity=train_args.stochasticity,
        )
    elif train_args.environment == "hike":
        env = MountainHike(
            bayes=bayes,
            variations=train_args.variations,
        )
    elif train_args.environment == "starkweather":
        env = StarkweatherEnv(
            p_omission=train_args.p_omission,
            bin_size=train_args.bin_size,
            iti_hazard=train_args.iti_hazard,
            iti_min=train_args.iti_min,
            nITI_microstates=train_args.nITI_microstates,
        )
        # Starkweather belief handling depends on your env implementation; if it supports bayes, add it there.
    else:
        raise NotImplementedError(f"Unknown environment {train_args.environment}")

    if getattr(train_args, "irrelevant", 0) != 0:
        env = Irrelevant(env, state_size=train_args.irrelevant, bayes=bayes)

    return env


def build_agent_from_train_args(train_args, environment):
    if train_args.algorithm != "drqn":
        raise NotImplementedError(f"Unknown algorithm {train_args.algorithm}")

    network_kwargs = {
        "num_layers": train_args.num_layers,
        "hidden_size": train_args.hidden_size,
    }
    agent = DRQN(
        cell=train_args.cell,
        action_size=environment.action_size,
        observation_size=environment.observation_size,
        **network_kwargs,
    )
    return agent


def save_probe(probe, outdir, train_id, episode, part_idx):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"probe_{train_id}_ep{episode:06d}_part{part_idx}.pt")
    torch.save(probe, path)
    # also keep a "latest" convenience copy
    latest = os.path.join(outdir, f"probe_{train_id}_latest_part{part_idx}.pt")
    torch.save(probe, latest)
    return path, latest


def main(args):
    # Retrieve training arguments from the run
    train_args = get_run_statistic(args.train_id)

    # Merge configurations
    config = vars(train_args) | vars(args)

    # Init wandb
    wandb.init(
        project="belief-regression",
        name=args.name,
        config=config,
        save_code=True,
    )
    config = wandb.config

    # Save code
    wandb.save("*.py")
    wandb.save("agents/*.py")
    wandb.save("environments/*.py")

    # Build env + agent
    environment = build_environment_from_train_args(train_args, bayes=True)
    agent = build_agent_from_train_args(train_args, environment)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    # Output directory: regression_weights/<train_id>/<wandb_runid>/
    run_id = wandb.run.id if wandb.run is not None else "no_wandb"
    outdir = os.path.join(args.outdir, args.train_id, run_id)
    os.makedirs(outdir, exist_ok=True)
    wandb.config.update({"regression_outdir": outdir}, allow_val_change=True)

    for episode in range(0, config.episodes + 1, args.period):
        # 1) load agent checkpoint
        agent.load(args.train_id, episode=episode)
        print(f"[episode {episode}] agent loaded")

        # 2) sample data
        hiddens, beliefs = generate_hiddens_and_beliefs(
            agent,
            environment,
            num_samples=args.num_samples,
            epsilon=args.epsilon,
            approximate=args.approximate,
        )
        print(f"[episode {episode}] hiddens {hiddens.shape}, beliefs parts={len(beliefs)}")

        # 3) move to device
        hiddens = hiddens.to(device)  # [N, H]
        beliefs = tuple(b.to(device) for b in beliefs)

        # 4) shuffle + split
        N = hiddens.size(0)
        perm = torch.randperm(N, device=device)
        hiddens = hiddens[perm]
        beliefs = tuple(b[perm] for b in beliefs)

        split = int(N * 0.8)
        X_train = hiddens[:split]
        X_test = hiddens[split:]

        # 5) fit/save per belief part
        for part_idx, belief_part in enumerate(beliefs):
            Y_train = belief_part[:split]
            Y_test = belief_part[split:]

            probe = fit_linear_probe(
                X_train, Y_train,
                standardize=args.standardize,
                add_bias=True,
            )

            rsq_train, mse_train, _ = eval_linear_probe(X_train, Y_train, probe)
            rsq_test, mse_test, _ = eval_linear_probe(X_test, Y_test, probe)

            saved_path, latest_path = save_probe(
                probe=probe,
                outdir=outdir,
                train_id=args.train_id,
                episode=episode,
                part_idx=part_idx,
            )

            # log
            wandb.log({
                "train/episode": episode,
                f"regression/rsq-train-{part_idx}": rsq_train,
                f"regression/mse-train-{part_idx}": mse_train,
                f"regression/rsq-test-{part_idx}": rsq_test,
                f"regression/mse-test-{part_idx}": mse_test,
            })

            print(
                f"[episode {episode}] part {part_idx} "
                f"rsq_test={rsq_test:.4f} rsq_train={rsq_train:.4f} "
                f"mse_train={mse_train:.6f} saved={os.path.basename(saved_path)}"
            )

            # optionally: track the file in wandb (keeps your “wandb and everything” usage)
            if args.wandb_save_weights:
                wandb.save(saved_path)
                wandb.save(latest_path)

    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser(description="Train linear regression probe(s) on hidden states -> belief")

    parser.add_argument("name", type=str, nargs="?", default=None)
    parser.add_argument("train_id", type=str)

    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--period", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--approximate", action="store_true")

    parser.add_argument("--no-standardize", action="store_false", dest="standardize")
    parser.set_defaults(standardize=True)

    parser.add_argument("--outdir", type=str, default="regression_weights")
    parser.add_argument("--wandb-save-weights", action="store_true", dest="wandb_save_weights")
    parser.set_defaults(wandb_save_weights=False)

    args = parser.parse_args()
    print("\n".join(f"\033[90m{k}=\033[0m{v}" for k, v in vars(args).items()))
    main(args)