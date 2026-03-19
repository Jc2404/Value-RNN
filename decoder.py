import warnings
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from agents.drqn import DRQN
from environments.crybaby import CryingBaby
from environments.gridworld import GridWorld
from environments.hike import MountainHike
from environments.irrelevant import Irrelevant
from environments.starkweather import StarkweatherEnv
from environments.tiger import Tiger
from environments.tmaze import TMaze
from utils import generate_hiddens_and_beliefs, get_run_statistic


class SoftmaxProbe(nn.Module):
    """Linear softmax probe that outputs log-probabilities."""

    def __init__(self, in_dim, out_dim, add_bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=add_bias)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


class MLPProbe(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, add_bias=True, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=add_bias),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim, bias=add_bias),
        )

    def forward(self, x):
        return F.log_softmax(self.net(x), dim=-1)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_total_episodes(train_args):
    if hasattr(train_args, "episodes"):
        return int(train_args.episodes)
    if hasattr(train_args, "num_episodes"):
        return int(train_args.num_episodes)
    raise AttributeError("Could not find total number of episodes in train args.")


def build_environment(train_args):
    env_name = train_args.environment

    if env_name == "tmaze":
        environment = TMaze(
            bayes=True,
            length=train_args.length,
            stochasticity=train_args.stochasticity,
        )
    elif env_name == "hike":
        environment = MountainHike(
            bayes=True,
            variations=train_args.variations,
        )
    elif env_name == "starkweather":
        environment = StarkweatherEnv(
            p_omission=train_args.p_omission,
            bin_size=train_args.bin_size,
            iti_hazard=train_args.iti_hazard,
            iti_min=train_args.iti_min,
            nITI_microstates=train_args.nITI_microstates,
        )
    elif env_name == "tiger":
        environment = Tiger(
            bayes=True,
            listen_accuracy=train_args.listen_accuracy,
            reward_listen=train_args.reward_listen,
            reward_correct=train_args.reward_correct,
            reward_wrong=train_args.reward_wrong,
            horizon=train_args.horizon,
        )
    elif env_name == "gridworld":
        environment = GridWorld(
            bayes=True,
            size=train_args.size,
            tprob=train_args.tprob,
            reward_scheme=train_args.reward_scheme,
            reward_margin=train_args.reward_margin,
            step_cost=train_args.step_cost,
        )
    elif env_name == "crybaby":
        environment = CryingBaby(
            bayes=True,
            p_hungry_if_full_wait=train_args.p_hungry_if_full_wait,
            p_stay_hungry_wait=train_args.p_stay_hungry_wait,
            p_full_if_feed=train_args.p_full_if_feed,
            p_cry_if_hungry=train_args.p_cry_if_hungry,
            p_cry_if_full=train_args.p_cry_if_full,
            reward_cry=train_args.reward_cry,
            cost_feed=train_args.cost_feed,
            reward_quiet=getattr(train_args, "reward_quiet", 0.0),
            p0_hungry=getattr(train_args, "p0_hungry", 0.5),
            horizon=getattr(train_args, "horizon", 50),
        )
    else:
        raise NotImplementedError(f"Unknown environment {env_name}")

    if getattr(train_args, "irrelevant", 0) != 0:
        environment = Irrelevant(
            environment,
            state_size=train_args.irrelevant,
            bayes=True,
        )

    return environment


def build_agent(train_args, environment, device):
    if train_args.algorithm != "drqn":
        raise NotImplementedError(f"Unknown algorithm {train_args.algorithm}")

    agent = DRQN(
        cell=train_args.cell,
        action_size=environment.action_size,
        observation_size=environment.observation_size,
        num_layers=train_args.num_layers,
        hidden_size=train_args.hidden_size,
    )
    agent.Q.to(device)
    agent.Q_tar.to(device)
    return agent


def project_name_from_args(args):
    if args.probe_type == "linear":
        return "belief-linear-probe"
    if args.use_MLP:
        return "belief-mlp-probe"
    return "belief-softmax-probe"


def prepare_features(X, standardize=True):
    if standardize:
        mean = X.mean(0, keepdim=True)
        std = X.std(0, keepdim=True, unbiased=False).clamp_min(1e-6)
        Xn = (X - mean) / std
    else:
        mean, std = None, None
        Xn = X
    return Xn, mean, std


def fit_linear_probe_torch(X_train, Y_train, standardize=True, add_bias=True, use_float64=True):
    device = X_train.device
    Xn, mean, std = prepare_features(X_train, standardize=standardize)

    if add_bias:
        ones = torch.ones(Xn.size(0), 1, device=device, dtype=Xn.dtype)
        Xn = torch.cat([Xn, ones], dim=1)

    Xr = Xn.double() if use_float64 else Xn
    Yr = Y_train.double() if use_float64 else Y_train
    solution = torch.linalg.lstsq(Xr, Yr).solution

    return {
        "W": solution,
        "mean": mean,
        "std": std,
        "standardize": standardize,
        "add_bias": add_bias,
        "use_float64": use_float64,
    }


@torch.no_grad()
def eval_linear_probe_torch(X, Y, probe_state):
    device = X.device
    if probe_state["standardize"]:
        Xn = (X - probe_state["mean"]) / probe_state["std"]
    else:
        Xn = X

    if probe_state["add_bias"]:
        ones = torch.ones(Xn.size(0), 1, device=device, dtype=Xn.dtype)
        Xn = torch.cat([Xn, ones], dim=1)

    Xr = Xn.double() if probe_state["use_float64"] else Xn
    Yr = Y.double() if probe_state["use_float64"] else Y
    Yhat = Xr @ probe_state["W"]

    mse = ((Yr - Yhat) ** 2).mean().item()
    num = ((Yr - Yhat) ** 2).sum()
    den = ((Yr - Yr.mean(0, keepdim=True)) ** 2).sum().clamp_min(1e-12)
    rsq = (1.0 - (num / den)).item()
    return {
        "mse": mse,
        "r2": rsq,
        "predictions": Yhat,
    }


@torch.no_grad()
def eval_softmax_probe(X, Y, state):
    probe = state["probe"]
    if state["standardize"]:
        Xn = (X - state["mean"]) / state["std"]
    else:
        Xn = X

    log_probs = probe(Xn)
    probs = log_probs.exp()
    kl = F.kl_div(log_probs, Y, reduction="batchmean").item()
    ce = -(Y * log_probs).sum(dim=-1).mean().item()

    return {
        "kl": kl,
        "ce": ce,
        "probs": probs,
        "log_probs": log_probs,
    }


def fit_softmax_probe(
    X_train,
    Y_train,
    X_eval,
    Y_eval,
    add_bias=True,
    standardize=True,
    epochs=200,
    lr=1e-2,
    batch_size=1024,
    use_MLP=True,
    mlp_hidden_dim=128,
    mlp_dropout=0.0,
    logger=None,
):
    device = X_train.device
    N, H = X_train.shape
    K = Y_train.shape[1]

    Xn_train, mean, std = prepare_features(X_train, standardize=standardize)
    Xn_eval = (X_eval - mean) / std if standardize else X_eval

    if use_MLP:
        probe = MLPProbe(
            H,
            K,
            hidden_dim=mlp_hidden_dim,
            add_bias=add_bias,
            dropout=mlp_dropout,
        ).to(device)
    else:
        probe = SoftmaxProbe(H, K, add_bias=add_bias).to(device)

    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction="batchmean")

    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(N, device=device)
        total_train_loss = 0.0
        total_train_ce = 0.0
        num_batches = 0

        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xn_train[idx]
            yb = Y_train[idx]

            log_probs = probe(xb)
            loss = criterion(log_probs, yb)
            ce = -(yb * log_probs).sum(dim=-1).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_train_loss += loss.item()
            total_train_ce += ce.item()
            num_batches += 1

        probe.eval()
        with torch.no_grad():
            eval_log_probs = probe(Xn_eval)
            eval_kl = criterion(eval_log_probs, Y_eval).item()
            eval_ce = (-(Y_eval * eval_log_probs).sum(dim=-1).mean()).item()

        avg_train_kl = total_train_loss / max(num_batches, 1)
        avg_train_ce = total_train_ce / max(num_batches, 1)

        if logger is not None:
            logger(
                {
                    "probe_optim/epoch": ep + 1,
                    "probe_optim/train_kl": avg_train_kl,
                    "probe_optim/train_ce": avg_train_ce,
                    "probe_optim/eval_kl": eval_kl,
                    "probe_optim/eval_ce": eval_ce,
                }
            )

        if (ep + 1) % 50 == 0 or ep == 0:
            print(
                f"[Probe epoch {ep + 1}/{epochs}] "
                f"train_KL={avg_train_kl:.4f} eval_KL={eval_kl:.4f}"
            )

    return {
        "probe": probe,
        "mean": mean,
        "std": std,
        "standardize": standardize,
    }


def make_probe_logger(episode, part_idx, probe_tag, num_epochs):
    def _log(metrics):
        payload = dict(metrics)
        payload["train/episode"] = int(episode)
        payload["probe/part_idx"] = int(part_idx)
        payload["probe/type"] = probe_tag
        if "probe_optim/epoch" in payload:
            epoch = int(payload["probe_optim/epoch"])
            payload["probe_optim/global_step"] = (
                int(episode) * int(num_epochs) + epoch + int(part_idx) * max(1, int(num_epochs)) * 100000
            )
        wandb.log(payload)
    return _log


def main(args):
    train_args = get_run_statistic(args.train_id)
    config = vars(train_args) | vars(args)
    project_name = project_name_from_args(args)

    wandb.init(
        project=project_name,
        name=args.name,
        config=config,
        save_code=True,
    )
    wandb.define_metric("probe_optim/global_step")
    wandb.define_metric("probe_optim/*", step_metric="probe_optim/global_step")

    wandb.save("*.py")
    wandb.save("agents/*.py")
    wandb.save("environments/*.py")

    device = select_device()
    print("Device:", device)

    environment = build_environment(train_args)
    agent = build_agent(train_args, environment, device)

    total_episodes = get_total_episodes(train_args)
    print("Total training episodes:", total_episodes)

    for episode in range(0, total_episodes + 1, args.mine_period):
        agent.load(args.train_id, episode=episode)
        print(f"Loaded checkpoint at episode {episode}")

        hiddens, beliefs = generate_hiddens_and_beliefs(
            agent,
            environment,
            num_samples=args.mine_num_samples,
            epsilon=args.epsilon,
            approximate=args.approximate,
        )

        hiddens = hiddens.to(device)
        beliefs = tuple(b.to(device) for b in beliefs)
        print(f"generated hiddens: {hiddens.shape}, beliefs: {[b.shape for b in beliefs]}")

        N = hiddens.size(0)
        perm = torch.randperm(N, device=device)
        hiddens = hiddens[perm]
        beliefs = tuple(b[perm] for b in beliefs)
        split = int(N * (1.0 - args.valid_fraction))
        X_train = hiddens[:split]
        X_eval = hiddens[split:]

        for part_idx, belief_part in enumerate(beliefs):
            if belief_part.ndim != 2:
                msg = (
                    f"Skipping belief part {part_idx} with ndim={belief_part.ndim}. "
                    "This probe script only supports vector belief parts."
                )
                warnings.warn(msg)
                wandb.log(
                    {
                        "train/episode": episode,
                        "probe/part_idx": part_idx,
                        "probe/skipped": 1,
                    }
                )
                continue

            Y_train = belief_part[:split]
            Y_eval = belief_part[split:]

            if args.probe_type == "linear":
                probe_state = fit_linear_probe_torch(
                    X_train,
                    Y_train,
                    standardize=args.probe_standardize,
                    add_bias=not args.no_bias,
                    use_float64=not args.no_float64,
                )
                train_metrics = eval_linear_probe_torch(X_train, Y_train, probe_state)
                eval_metrics = eval_linear_probe_torch(X_eval, Y_eval, probe_state)

                wandb.log(
                    {
                        "train/episode": episode,
                        "probe/part_idx": part_idx,
                        "probe/type": "linear",
                        f"probe_linear/train_mse-{part_idx}": train_metrics["mse"],
                        f"probe_linear/eval_mse-{part_idx}": eval_metrics["mse"],
                        f"probe_linear/train_r2-{part_idx}": train_metrics["r2"],
                        f"probe_linear/eval_r2-{part_idx}": eval_metrics["r2"],
                    }
                )
                print(
                    f"[episode {episode}] belief {part_idx} linear: "
                    f"train_R2={train_metrics['r2']:.4f}, eval_R2={eval_metrics['r2']:.4f}, "
                    f"train_MSE={train_metrics['mse']:.6f}, eval_MSE={eval_metrics['mse']:.6f}"
                )
            else:
                probe_logger = make_probe_logger(
                    episode=episode,
                    part_idx=part_idx,
                    probe_tag="mlp" if args.use_MLP else "softmax",
                    num_epochs=args.probe_epochs,
                )
                probe_state = fit_softmax_probe(
                    X_train,
                    Y_train,
                    X_eval,
                    Y_eval,
                    add_bias=not args.no_bias,
                    standardize=args.probe_standardize,
                    epochs=args.probe_epochs,
                    lr=args.probe_lr,
                    batch_size=args.probe_batch_size,
                    use_MLP=args.use_MLP,
                    mlp_hidden_dim=args.mlp_hidden_dim,
                    mlp_dropout=args.mlp_dropout,
                    logger=probe_logger,
                )

                train_metrics = eval_softmax_probe(X_train, Y_train, probe_state)
                eval_metrics = eval_softmax_probe(X_eval, Y_eval, probe_state)

                wandb.log(
                    {
                        "train/episode": episode,
                        "probe/part_idx": part_idx,
                        "probe/type": "mlp" if args.use_MLP else "softmax",
                        f"probe_softmax/train_kl-{part_idx}": train_metrics["kl"],
                        f"probe_softmax/eval_kl-{part_idx}": eval_metrics["kl"],
                        f"probe_softmax/train_ce-{part_idx}": train_metrics["ce"],
                        f"probe_softmax/eval_ce-{part_idx}": eval_metrics["ce"],
                    }
                )
                print(
                    f"[episode {episode}] belief {part_idx} {('mlp' if args.use_MLP else 'softmax')}: "
                    f"train_KL={train_metrics['kl']:.4f}, eval_KL={eval_metrics['kl']:.4f}, "
                    f"train_CE={train_metrics['ce']:.4f}, eval_CE={eval_metrics['ce']:.4f}"
                )

    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser(description="Probe RNN beliefs with softmax/MLP KL probe or linear probe")
    parser.add_argument("name", type=str, nargs="?", default=None)
    parser.add_argument("train_id", type=str)

    parser.add_argument("--mine_num_samples", type=int, default=10000)
    parser.add_argument("--mine_period", type=int, default=100)
    parser.add_argument("--approximate", action="store_true")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--valid_fraction", type=float, default=0.2)

    parser.add_argument("--probe-type", choices=["softmax", "linear"], default="softmax")
    parser.add_argument("--use_MLP", action="store_true")
    parser.add_argument("--probe_epochs", type=int, default=300)
    parser.add_argument("--probe_lr", type=float, default=1e-2)
    parser.add_argument("--probe_batch_size", type=int, default=1024)
    parser.add_argument("--probe_standardize", action="store_true")
    parser.add_argument("--mlp_hidden_dim", type=int, default=128)
    parser.add_argument("--mlp_dropout", type=float, default=0.0)
    parser.add_argument("--no_bias", action="store_true")
    parser.add_argument("--no_float64", action="store_true")

    args = parser.parse_args()
    main(args)
