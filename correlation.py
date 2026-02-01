"""
Faithful linear regression probe for decoding beliefs from RNN hidden states.

Matches the original methodology:
  - StandardScaler-like normalization (population std, ddof=0)
  - Optional bias term appended to X
  - Least-squares fit (lstsq)
  - Global R^2 computed as 1 - ||Y - Yhat||_F^2 / ||Y - mean(Y)||_F^2
  - IMPORTANT: uses approximate=True by default to mimic "all timesteps" stacking

Notes:
  - Uses float64 for regression stability (closer to SciPy)
  - Logs test/train R^2 + train MSE to wandb
"""

import wandb
import torch

from argparse import ArgumentParser

from environments.tmaze import TMaze
from environments.hike import MountainHike
from environments.irrelevant import Irrelevant
from environments.starkweather import StarkweatherEnv

from agents.drqn import DRQN

from utils import generate_hiddens_and_beliefs, get_run_statistic


# -----------------------------
# Linear regression probe
# -----------------------------
def fit_linear_probe(X, Y, add_bias=True, standardize=True, use_float64=True):
    """
    X: [N, H] hidden states (torch)
    Y: [N, K] belief vectors (torch)

    Returns probe dict with W and standardization stats.
    """
    device = X.device

    # standardize like sklearn StandardScaler (ddof=0)
    if standardize:
        mean = X.mean(0, keepdim=True)
        std = X.std(0, keepdim=True, unbiased=False).clamp_min(1e-6)
        Xn = (X - mean) / std
    else:
        mean, std = None, None
        Xn = X

    if add_bias:
        ones = torch.ones(Xn.size(0), 1, device=device, dtype=Xn.dtype)
        Xn = torch.cat([Xn, ones], dim=1)

    # closer to SciPy: do regression in float64
    if use_float64:
        Xr = Xn.double()
        Yr = Y.double()
    else:
        Xr = Xn
        Yr = Y

    # W = argmin ||Xr W - Yr||_2
    res = torch.linalg.lstsq(Xr, Yr)
    W = res.solution  # [H(+1), K]

    # Store W in float64 if fitted in float64
    return {
        "W": W,
        "mean": mean,
        "std": std,
        "add_bias": add_bias,
        "standardize": standardize,
        "use_float64": use_float64,
    }


@torch.no_grad()
def eval_linear_probe(X, Y, probe):
    """
    Returns:
      rsq (float), Yhat (tensor)
    """
    device = X.device

    if probe["standardize"]:
        Xn = (X - probe["mean"]) / probe["std"]
    else:
        Xn = X

    if probe["add_bias"]:
        ones = torch.ones(Xn.size(0), 1, device=device, dtype=Xn.dtype)
        Xn = torch.cat([Xn, ones], dim=1)

    if probe["use_float64"]:
        Xr = Xn.double()
        Yr = Y.double()
    else:
        Xr = Xn
        Yr = Y

    Yhat = Xr @ probe["W"]

    # Global R^2 (matches the original numpy/scipy implementation)
    num = ((Yr - Yhat) ** 2).sum()
    den = ((Yr - Yr.mean(0, keepdim=True)) ** 2).sum().clamp_min(1e-12)
    rsq = 1.0 - (num / den)

    return rsq.item(), Yhat


# -----------------------------
# Main script
# -----------------------------
def build_environment(train_args):
    if train_args.environment == "tmaze":
        env = TMaze(
            bayes=True,
            length=train_args.length,
            stochasticity=train_args.stochasticity,
        )
    elif train_args.environment == "hike":
        env = MountainHike(
            bayes=True,
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
    else:
        raise NotImplementedError(f"Unknown environment {train_args.environment}")

    if getattr(train_args, "irrelevant", 0) != 0:
        env = Irrelevant(
            env,
            state_size=train_args.irrelevant,
            bayes=True,
        )
    return env


def build_agent(train_args, environment):
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


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # mps support disabled in your original; keep consistent
    return torch.device("cpu")


def main(args):
    train_args = get_run_statistic(args.train_id)

    config = vars(train_args) | vars(args)

    wandb.init(
        project="belief-regression",
        name=args.name,
        config=config,
        save_code=True,
    )
    config = wandb.config

    wandb.save("*.py")
    wandb.save("agents/*.py")
    wandb.save("environments/*.py")

    environment = build_environment(train_args)
    agent = build_agent(train_args, environment)

    device = select_device()
    print("Device:", device)

    for episode in range(0, config.episodes + 1, args.period):
        # load agent checkpoint
        agent.load(args.train_id, episode=episode)
        print(f"[episode {episode}] agent loaded", flush=True)

        # sample hidden states + belief tuple
        hiddens, beliefs = generate_hiddens_and_beliefs(
            agent,
            environment,
            num_samples=args.num_samples,
            epsilon=args.epsilon,
            approximate=args.approximate,  # IMPORTANT for fidelity
        )

        # move to device
        hiddens = hiddens.to(device)                      # [N, H]
        beliefs = tuple(b.to(device) for b in beliefs)    # each [N, K_i]

        print(f"hiddens: {tuple(hiddens.shape)}", flush=True)
        print(f"belief parts: {[tuple(b.shape) for b in beliefs]}", flush=True)

        # shuffle + split
        N = hiddens.size(0)
        perm = torch.randperm(N, device=device)
        hiddens = hiddens[perm]
        beliefs = tuple(b[perm] for b in beliefs)

        split = int(N * (1.0 - args.valid_size))
        X_train, X_test = hiddens[:split], hiddens[split:]

        for part_idx, belief_part in enumerate(beliefs):
            Y_train, Y_test = belief_part[:split], belief_part[split:]

            # (optional) sanity: belief rows sum to ~1
            if args.check_belief_sums:
                s = Y_train.sum(dim=1)
                print(f"part {part_idx} belief sum: mean={s.mean().item():.4f}, "
                      f"min={s.min().item():.4f}, max={s.max().item():.4f}", flush=True)

            # fit on train
            probe = fit_linear_probe(
                X_train,
                Y_train,
                add_bias=True,
                standardize=args.standardize,
                use_float64=not args.no_float64,
            )

            # eval train
            rsq_train, Yhat_train = eval_linear_probe(X_train, Y_train, probe)
            train_mse = torch.mean((Y_train.double() - Yhat_train.double()) ** 2).item()

            # eval test
            rsq_test, _ = eval_linear_probe(X_test, Y_test, probe)

            wandb.log({
                "train/episode": episode,
                f"regression/rsq-{part_idx}": rsq_test,
                f"regression/rsq_train-{part_idx}": rsq_train,
                f"regression/train_mse-{part_idx}": train_mse,
            })

            print(
                f"[episode {episode}] part {part_idx}: "
                f"R2_test={rsq_test:.4f}, R2_train={rsq_train:.4f}, train_mse={train_mse:.6f}",
                flush=True,
            )

    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Linear regression probe (faithful to linreg_fit/linreg_eval)",
    )
    parser.add_argument("name", type=str, nargs="?", default=None)
    parser.add_argument("train_id", type=str)

    # sampling
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--period", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.0)

    # fidelity to original stacking: use all timesteps by default
    parser.add_argument("--approximate", action="store_true",
                        help="If set, return all timesteps in each trajectory (recommended for fidelity).")
    # Make it default True unless user explicitly disables:
    parser.set_defaults(approximate=True)

    # train/eval split
    parser.add_argument("--valid_size", type=float, default=0.2)

    # probe options
    parser.add_argument("--no-standardize", action="store_false", dest="standardize")
    parser.set_defaults(standardize=True)

    parser.add_argument("--no-float64", action="store_true",
                        help="Disable float64 regression (float32 only).")
    parser.add_argument("--check-belief-sums", action="store_true",
                        help="Print sanity stats for belief row-sums.")

    args = parser.parse_args()
    print("\n".join(f"{k}={v}" for k, v in vars(args).items()), flush=True)
    main(args)

