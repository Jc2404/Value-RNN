# protocolB_decoder_all.py
# Protocol B: for EACH (episode, env-variant), REFIT each enabled decoder on that variant,
# then evaluate on that SAME variant (resample unless --train_set).
#
# Supports 4 decoder types (enabled via flags):
#   1) MINE MI (mutual information)
#   2) Linear regression (belief -> R^2)
#   3) Softmax belief probe (belief -> KL, CE)
#   4) State decoder (hidden -> state labels; LL, pcor)  [uses generate_hiddens_and_states]
#
# Includes:
#   - W&B step fix (define_metric + MINE logger wrapper with monotonic global_step)
#   - Excel export with one sheet per episode (x = task_value, y = metrics in columns)
#
# Assumptions:
#   - get_run_statistic(train_id) returns train_args with .environment, .episodes, etc.
#   - generate_hiddens_and_beliefs(agent, env, ...) returns (hiddens [N,H], beliefs tuple)
#   - generate_hiddens_and_states(agent, env, ...) returns (hiddens [N,H], states [N] or [N,K])
#   - DRQN.load(train_id, episode=episode) works.
#
# NOTE:
#   - For MI/regression/softmax probes: the "label" is belief vectors (probabilities).
#   - For state decoder: the "label" is discrete states (ints 0..K-1). If yours are not,
#     you MUST map them before training (you can do it in generate_hiddens_and_states).

import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from argparse import ArgumentParser

from environments.tmaze import TMaze
from environments.hike import MountainHike
from environments.irrelevant import Irrelevant
from environments.starkweather import StarkweatherEnv
from environments.tiger import Tiger
from environments.gridworld import GridWorld
from environments.crybaby import CryingBaby

from agents.drqn import DRQN
from mine.mine import MutualInformationNeuralEstimator
from utils import (
    generate_hiddens_and_beliefs,
    generate_hiddens_and_states,
    get_run_statistic,
)


# -----------------------------
# Device + env helpers
# -----------------------------
def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_variant(vname: str):
    """
    Examples:
      "tmaze_length=45" -> ("tmaze_length", 45.0)
      "starkweather_p_omission=0.1" -> ("starkweather_p_omission", 0.1)
      "base" -> ("base", None)
    """
    if vname == "base":
        return "base", None
    k, v = vname.split("=")
    return k, float(v)


def build_environment(train_args, overrides=None):
    overrides = overrides or {}
    env_name = train_args.environment

    if env_name == "tmaze":
        env = TMaze(
            bayes=True,
            length=overrides.get("length", train_args.length),
            stochasticity=overrides.get("stochasticity", train_args.stochasticity),
        )
    elif env_name == "hike":
        env = MountainHike(
            bayes=True,
            variations=overrides.get("variations", train_args.variations),
        )
    elif env_name == "starkweather":
        env = StarkweatherEnv(
            p_omission=overrides.get("p_omission", train_args.p_omission),
            bin_size=overrides.get("bin_size", train_args.bin_size),
            iti_hazard=overrides.get("iti_hazard", train_args.iti_hazard),
            iti_min=overrides.get("iti_min", train_args.iti_min),
            nITI_microstates=overrides.get("nITI_microstates", train_args.nITI_microstates),
        )
    elif env_name == "tiger":
        env = Tiger(
            bayes=True,
            listen_accuracy=overrides.get("listen_accuracy", train_args.listen_accuracy),
            reward_listen=overrides.get("reward_listen", train_args.reward_listen),
            reward_correct=overrides.get("reward_correct", train_args.reward_correct),
            reward_wrong=overrides.get("reward_wrong", train_args.reward_wrong),
            horizon=overrides.get("horizon", train_args.horizon),
        )
    elif env_name == "gridworld":
        env = GridWorld(
            bayes=True,
            size=overrides.get("size", train_args.size),
            tprob=overrides.get("tprob", train_args.tprob),
            reward_scheme=overrides.get("reward_scheme", train_args.reward_scheme),
            reward_margin=overrides.get("reward_margin", train_args.reward_margin),
            step_cost=overrides.get("step_cost", train_args.step_cost),
        )
    elif env_name == "crybaby":
        env = CryingBaby(
            bayes=True,
            p_cry_if_hungry=overrides.get("p_cry_if_hungry", train_args.p_cry_if_hungry),
            p_cry_if_full=overrides.get("p_cry_if_full", train_args.p_cry_if_full),
            p_hungry_if_full_wait=overrides.get("p_hungry_if_full_wait", train_args.p_hungry_if_full_wait),
            p_stay_hungry_wait=overrides.get("p_stay_hungry_wait", train_args.p_stay_hungry_wait),
            p_full_if_feed=overrides.get("p_full_if_feed", train_args.p_full_if_feed),
            reward_cry=overrides.get("reward_cry", train_args.reward_cry),
            cost_feed=overrides.get("cost_feed", train_args.cost_feed),
        )
    else:
        raise NotImplementedError(f"Unknown environment {env_name}")

    if getattr(train_args, "irrelevant", 0) != 0:
        env = Irrelevant(env, state_size=train_args.irrelevant, bayes=True)

    return env


def pick_variants(train_args, args):
    """
    Returns: list of (variant_name, overrides_dict)

    Only one test flag is honored at a time; others ignored (first match wins).
    If none selected, returns [("base", {})].
    """
    env_name = train_args.environment
    variants = []

    if env_name == "tmaze":
        if args.test_length:
            for L in [35, 40, 45, 50, 55, 60]:
                variants.append((f"tmaze_length={L}", {"length": L}))
            return variants
        if args.test_stochasticity:
            for s in [0, 0.05, 0.1, 0.15]:
                variants.append((f"tmaze_stochasticity={s}", {"stochasticity": s}))
            return variants

    if env_name == "hike":
        if args.test_variations:
            for v in [1, 2, 4, 8]:
                variants.append((f"hike_variations={v}", {"variations": v}))
            return variants

    if env_name == "starkweather":
        if args.test_p_omission:
            for p in [0.0, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2]:
                variants.append((f"starkweather_p_omission={p}", {"p_omission": p}))
            return variants
        if args.test_bin_size:
            grid = [train_args.bin_size, max(1, train_args.bin_size // 2), train_args.bin_size * 2]
            seen, grid2 = set(), []
            for x in grid:
                if x not in seen:
                    grid2.append(x)
                    seen.add(x)
            for b in grid2:
                variants.append((f"starkweather_bin_size={b}", {"bin_size": b}))
            return variants
        if args.test_iti_hazard:
            for h in [0.01, 0.05, 0.1, 0.2]:
                variants.append((f"starkweather_iti_hazard={h}", {"iti_hazard": h}))
            return variants
        if args.test_iti_min:
            for m in [0, 5, 10, 20]:
                variants.append((f"starkweather_iti_min={m}", {"iti_min": m}))
            return variants
        if args.test_nITI_microstates:
            for n in [1, 2, 4, 8]:
                variants.append((f"starkweather_nITI_microstates={n}", {"nITI_microstates": n}))
            return variants
        
    if env_name == "tiger":
        if args.test_listen_accuracy:
            for p in [0.55, 0.65, 0.75, 0.85, 0.95, 1.0]:
                variants.append((f"listen_accuracy={p}", {"listen_accuracy": p}))
            return variants
        if args.test_reward_listen:
            for r in [-5, -2, -1.5, -1, -0.5, -0.1]:
                variants.append((f"reward_listen={r}", {"reward_listen": r}))
            return variants
        
    if env_name == "gridworld":
        if args.test_grid_size:
            for s in [6, 8, 10, 12, 14]:
                variants.append((f"grid_size={s}", {"size": s}))
            return variants
        if args.test_tprob:
            for p in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                variants.append((f"tprob={p}", {"tprob": p}))
            return variants
        
    if env_name == "crybaby":
        if args.test_p_cry_if_hungry:
            grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            return [(f"crybaby_p_cry_if_hungry={p}", {"p_cry_if_hungry": p}) for p in grid]
        if args.test_p_cry_if_full:
            grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            return [(f"crybaby_p_cry_if_full={p}", {"p_cry_if_full": p}) for p in grid]    

    return variants

# -----------------------------
# Common train/test split
# -----------------------------
def shuffle_split_tensors(X, Ys, valid_size, device):
    """
    X: [N, ...] tensor
    Ys: tensor or tuple(tensor) with same leading N
    valid_size: fraction for test/valid split
    Returns: X_tr, X_te, Ys_tr, Ys_te
    """
    N = X.size(0)
    perm = torch.randperm(N, device=device)
    X = X[perm]

    if isinstance(Ys, tuple):
        Ys = tuple(y[perm] for y in Ys)
    else:
        Ys = Ys[perm]

    split = int(N * (1.0 - valid_size))
    X_tr, X_te = X[:split], X[split:]

    if isinstance(Ys, tuple):
        Ys_tr = tuple(y[:split] for y in Ys)
        Ys_te = tuple(y[split:] for y in Ys)
    else:
        Ys_tr, Ys_te = Ys[:split], Ys[split:]

    return X_tr, X_te, Ys_tr, Ys_te


# -----------------------------
# (1) MINE MI
# -----------------------------
def build_mine(hiddens, beliefs, args, device):
    belief_sizes, representation_sizes = [], []
    for belief_part in beliefs:
        belief_sizes.append(belief_part.size(-1))
        if belief_part.ndim == 2:
            representation_sizes.append(None)
        elif belief_part.ndim == 3:
            representation_sizes.append(args.representation_size)
        else:
            raise ValueError("Expected belief parts with 2 or 3 dims")

    return MutualInformationNeuralEstimator(
        hs_sizes=hiddens.size(-1),
        belief_sizes=belief_sizes,
        hidden_size=args.mine_hidden_size,
        num_layers=args.mine_num_layers,
        alpha=args.mine_alpha,
        representation_sizes=representation_sizes,
        belief_part=args.belief_part,
        device=device,
    )


def make_mine_logger(episode, vname, num_epochs):
    """
    Ensures W&B step is monotonic by logging a global_step.
    """
    # Create a stable integer ID per variant string for step spacing (optional).
    # We keep it simple: global_step = episode * num_epochs + epoch.
    def _log(d):
        d = dict(d)
        if "mine_optim/epoch" in d:
            ep = int(d["mine_optim/epoch"])
            d["mine_optim/global_step"] = int(episode) * int(num_epochs) + ep
        d["train/episode"] = episode
        d["task/variant"] = vname
        wandb.log(d)
    return _log


# -----------------------------
# (2) Linear regression (belief R^2)
# faithful to your global R^2 metric
# -----------------------------
def fit_linear_probe_torch(X, Y, add_bias=True, standardize=True, use_float64=True):
    device = X.device

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

    if use_float64:
        Xr = Xn.double()
        Yr = Y.double()
    else:
        Xr = Xn
        Yr = Y

    res = torch.linalg.lstsq(Xr, Yr)
    W = res.solution

    return {
        "W": W,
        "mean": mean,
        "std": std,
        "add_bias": add_bias,
        "standardize": standardize,
        "use_float64": use_float64,
    }


@torch.no_grad()
def eval_linear_probe_torch(X, Y, probe):
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

    num = ((Yr - Yhat) ** 2).sum()
    den = ((Yr - Yr.mean(0, keepdim=True)) ** 2).sum().clamp_min(1e-12)
    rsq = 1.0 - (num / den)

    return rsq.item(), Yhat


# -----------------------------
# (3) Softmax belief probe (KL, CE)
# (your torch KLDivLoss-based training)
# -----------------------------
class SoftmaxProbe(nn.Module):
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


def fit_softmax_probe_torch(X, Y, args):
    """
    Fits per belief-part probe on (X,Y) with KLDivLoss.
    """
    device = X.device
    N, H = X.shape
    K = Y.shape[1]

    if args.standardize:
        mean = X.mean(0, keepdim=True)
        std = X.std(0, keepdim=True).clamp_min(1e-6)
        Xn = (X - mean) / std
    else:
        mean, std = None, None
        Xn = X

    if args.use_mlp_probe:
        probe = MLPProbe(H, K, hidden_dim=args.mlp_hidden_dim, dropout=args.mlp_dropout).to(device)
    else:
        probe = SoftmaxProbe(H, K).to(device)

    opt = torch.optim.Adam(probe.parameters(), lr=args.probe_lr)
    criterion = nn.KLDivLoss(reduction="batchmean")

    probe.train()
    for _ in range(args.probe_epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, args.probe_batch_size):
            idx = perm[i:i + args.probe_batch_size]
            xb = Xn[idx]
            yb = Y[idx]

            log_probs = probe(xb)
            loss = criterion(log_probs, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

    return {"probe": probe, "mean": mean, "std": std}


@torch.no_grad()
def eval_softmax_probe_torch(X, Y, state, standardize=True):
    probe = state["probe"]

    if standardize and state["mean"] is not None:
        Xn = (X - state["mean"]) / state["std"]
    else:
        Xn = X

    log_probs = probe(Xn)
    probs = log_probs.exp()

    kl = F.kl_div(log_probs, Y, reduction="batchmean").item()
    ce = -(Y * log_probs).sum(dim=-1).mean().item()

    return {"kl": kl, "ce": ce, "probs": probs}


# -----------------------------
# (4) State decoder (LL, pcor)
# Fixed-K variant (K from environment if available)
# -----------------------------
def safelog_torch(x, eps=1e-12):
    return torch.log(torch.clamp(x, min=eps))


class LinearMultinomialProbe(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.linear(x)  # logits


def fit_state_decoder_torch_fixedK(X_train, y_train, K, args):
    device = X_train.device
    N, H = X_train.shape

    if y_train.ndim > 1:
        y_train = y_train.squeeze(-1)
    y_train = y_train.long()

    y_min = int(y_train.min().item())
    y_max = int(y_train.max().item())
    if y_min < 0 or y_max >= K:
        raise ValueError(
            f"[state-decoder] y_train out of range for K={K}: min={y_min}, max={y_max}. "
            f"Your state IDs may need remapping to [0, K-1]."
        )

    if args.standardize:
        mean = X_train.mean(0, keepdim=True)
        std = X_train.std(0, keepdim=True).clamp_min(1e-6)
    else:
        mean, std = None, None

    probe = LinearMultinomialProbe(H, K, bias=True).to(device)

    # reg analogue
    weight_decay = 1.0 / max(args.C, 1e-12)
    opt = torch.optim.Adam(probe.parameters(), lr=args.probe_lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    probe.train()
    for _ in range(args.probe_epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, args.probe_batch_size):
            idx = perm[i:i + args.probe_batch_size]
            xb = X_train[idx]
            yb = y_train[idx]

            if args.standardize and mean is not None:
                xb = (xb - mean) / std

            logits = probe(xb)
            loss = criterion(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

    return {"probe": probe, "mean": mean, "std": std, "standardize": args.standardize, "K": K}


@torch.no_grad()
def eval_state_decoder_torch_fixedK(X, y, state):
    probe = state["probe"]
    K = state["K"]

    if y.ndim > 1:
        y = y.squeeze(-1)
    y = y.long()

    y_min = int(y.min().item())
    y_max = int(y.max().item())
    if y_min < 0 or y_max >= K:
        raise ValueError(
            f"[state-decoder] y out of range for K={K}: min={y_min}, max={y_max}. "
            f"Your state IDs may need remapping to [0, K-1]."
        )

    if state["standardize"] and state["mean"] is not None:
        X = (X - state["mean"]) / state["std"]

    logits = probe(X)
    pte_hat = F.softmax(logits, dim=-1)
    yhat = torch.argmax(pte_hat, dim=-1)

    p_true = pte_hat.gather(1, y.view(-1, 1)).squeeze(1)
    LL = safelog_torch(p_true).mean().item()
    pcor = (yhat == y).float().mean().item() * 100.0

    classes = torch.unique(y)
    phat_mean = torch.stack([
        pte_hat[y == c].mean(dim=0) if (y == c).any() else torch.zeros(K, device=X.device)
        for c in classes
    ], dim=0)

    return {"LL": LL, "pcor": pcor, "phat_mean": phat_mean.detach().cpu(), "classes": classes.detach().cpu()}


def get_num_states_from_env(env):
    """
    Preferred: env.num_states
    Fallback: try env.environment.num_states (Irrelevant wrapper style)
    If unavailable, returns None.
    """
    if hasattr(env, "K"):
        try:
            return int(env.K)
        except Exception:
            pass
    if hasattr(env, "environment") and hasattr(env.environment, "K"): # unused
        try:
            return int(env.K)
        except Exception:
            pass
    return None


# -----------------------------
# Main (Protocol B)
# -----------------------------
def main(args):
    train_args = get_run_statistic(args.train_id)
    device = select_device()
    print("Device:", device)

    variants = pick_variants(train_args, args)

    # Build env0 to get action/obs sizes consistent
    env0 = build_environment(train_args, overrides=variants[0][1])

    # W&B init
    cfg = vars(train_args) | vars(args)
    wandb.init(project=args.wandb_project, name=args.name, config=cfg, save_code=True)

    # ---- W&B step fix ----
    wandb.define_metric("train/episode")
    wandb.define_metric("mine_optim/global_step")
    wandb.define_metric("*", step_metric="train/episode")
    wandb.define_metric("mine_optim/*", step_metric="mine_optim/global_step")
    # ----------------------

    wandb.save("*.py")
    wandb.save("agents/*.py")
    wandb.save("environments/*.py")

    # Excel aggregation
    os.makedirs(args.report_dir, exist_ok=True)
    excel_path = os.path.join(args.report_dir, f"protocolB_{args.name}_{args.train_id}.xlsx")
    episode_rows = {}

    # agent
    agent = DRQN(
        cell=train_args.cell,
        action_size=env0.action_size,
        observation_size=env0.observation_size,
        num_layers=train_args.num_layers,
        hidden_size=train_args.hidden_size,
    )

    # If nothing enabled, fail fast (common user mistake)
    if not (args.run_mi or args.run_regression or args.run_softmax_belief or args.run_state_decoder):
        raise ValueError("No decoders enabled. Use at least one of: --run_mi --run_linreg --run_softmax --run_state")

    if args.end_episode < 0 or args.end_episode > train_args.episodes:
        args.end_episode = train_args.episodes
    for episode in range(0, args.end_episode + 1, args.period):
        agent.load(args.train_id, episode=episode)
        print(f"[episode {episode}] agent loaded", flush=True)

        episode_rows[episode] = []

        for vname, overrides in variants:
            venv = build_environment(train_args, overrides=overrides)

            task_name, task_value = parse_variant(vname)

            # Row template
            def add_row(metric_name, value):
                episode_rows[episode].append({
                    "variant": vname,
                    "task_name": task_name,
                    "task_value": task_value,
                    "metric": metric_name,
                    "value": float(value) if value is not None else float("nan"),
                })

            if args.run_mi or args.run_regression or args.run_softmax_belief:
                # Train sample
                h_tr, b_tr = generate_hiddens_and_beliefs(
                    agent, venv,
                    num_samples=args.num_samples,
                    epsilon=args.epsilon,
                    approximate=args.approximate,
                )
                h_tr = h_tr.to(device)
                b_tr = tuple(bb.to(device) for bb in b_tr)

                if not args.train_set:
                    h_ev, b_ev = generate_hiddens_and_beliefs(
                        agent, venv,
                        num_samples=args.num_samples,
                        epsilon=args.epsilon,
                        approximate=args.approximate,
                    )
                    h_ev = h_ev.to(device)
                    b_ev = tuple(bb.to(device) for bb in b_ev)
                else:
                    h_ev, b_ev = h_tr, b_tr

                if args.run_regression or args.run_softmax_belief:
                    Xtr, Xte, Btr, Bte = shuffle_split_tensors(h_tr, b_tr, args.valid_size, device)

                # ---------------- MI ----------------
                if args.run_mi:
                    mine = build_mine(h_tr, b_tr, args, device)

                    mine.optimize(
                        h_tr, b_tr,
                        num_epochs=args.mine_num_epochs,
                        logger=make_mine_logger(episode, vname, args.mine_num_epochs),
                        learning_rate=args.mine_learning_rate,
                        batch_size=args.mine_batch_size,
                        lambd=args.mine_lambda,
                        valid_size=args.valid_size,
                    )

                    mi = mine.estimate(h_ev, b_ev)

                    key = f"mi/refit_on/{vname}"
                    if args.belief_part is not None:
                        key += f"-part{args.belief_part}"
                    if args.epsilon != 0.0:
                        key += f"-eps{args.epsilon}"

                    wandb.log({"train/episode": episode, "task/variant": vname, key: mi})
                    add_row("MI", mi)
                    print(f"[episode {episode}] {vname} MI(refit) = {mi}", flush=True)

                # ---------------- Linear regression ----------------
                if args.run_regression:
                    # Evaluate per belief-part
                    for part_idx, (Ytr, Yte) in enumerate(zip(Btr, Bte)):
                        probe = fit_linear_probe_torch(
                            Xtr, Ytr,
                            add_bias=True,
                            standardize=args.standardize,
                            use_float64=not args.no_float64,
                        )
                        rsq_te, _ = eval_linear_probe_torch(Xte, Yte, probe)
                        rsq_tr, _ = eval_linear_probe_torch(Xtr, Ytr, probe)

                        wandb.log({
                            "train/episode": episode,
                            "task/variant": vname,
                            f"linreg/rsq-{part_idx}": rsq_te,
                            f"linreg/rsq_train-{part_idx}": rsq_tr,
                        })
                        add_row(f"linreg_rsq-{part_idx}", rsq_te)

                # ---------------- Softmax belief probe ----------------
                if args.run_softmax_belief:
                    for part_idx, (Ytr, Yte) in enumerate(zip(Btr, Bte)):
                        sm_state = fit_softmax_probe_torch(Xtr, Ytr, args)
                        res_te = eval_softmax_probe_torch(Xte, Yte, sm_state, standardize=args.standardize)
                        res_tr = eval_softmax_probe_torch(Xtr, Ytr, sm_state, standardize=args.standardize)

                        wandb.log({
                            "train/episode": episode,
                            "task/variant": vname,
                            f"softmax/kl-{part_idx}": res_te["kl"],
                            f"softmax/ce-{part_idx}": res_te["ce"],
                            f"softmax/kl_train-{part_idx}": res_tr["kl"],
                            f"softmax/ce_train-{part_idx}": res_tr["ce"],
                        })
                        add_row(f"softmax_kl-{part_idx}", res_te["kl"])
                        add_row(f"softmax_ce-{part_idx}", res_te["ce"])

            # ---------------- State decoder
            if args.run_state_decoder:
                h_s, states = generate_hiddens_and_states(
                    agent, venv,
                    num_samples=args.num_samples,
                    epsilon=args.epsilon,
                    approximate=args.approximate,
                )
                h_s = h_s.to(device)
                states = states.to(device)

                # If one-hot, convert to labels
                if states.ndim == 2:
                    states = states.argmax(dim=1)

                # Get K from env if possible
                K = get_num_states_from_env(venv)
                if K is None:
                    # fallback: infer from data (less preferred)
                    K = int(states.max().item() + 1)

                Xtr, Xte, ytr, yte = shuffle_split_tensors(h_s, states, args.valid_size, device)

                dec = fit_state_decoder_torch_fixedK(Xtr, ytr, K=K, args=args)

                res_te = eval_state_decoder_torch_fixedK(Xte, yte, dec)
                res_tr = eval_state_decoder_torch_fixedK(Xtr, ytr, dec)

                wandb.log({
                    "train/episode": episode,
                    "task/variant": vname,
                    "state/LL": res_te["LL"],
                    "state/pcor": res_te["pcor"],
                    "state/LL_train": res_tr["LL"],
                    "state/pcor_train": res_tr["pcor"],
                })
                add_row("state_LL", res_te["LL"])
                add_row("state_pcor", res_te["pcor"])

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for ep, rows in episode_rows.items():
            df = pd.DataFrame(rows)
            if df.empty:
                continue

            df_sorted = df.sort_values(by=["task_value"], na_position="first")

            wide = df_sorted.pivot_table(
                index=["variant", "task_name", "task_value"],
                columns=["metric"],
                values="value",
                aggfunc="mean",
            ).reset_index()

            sheet_name = f"ep_{ep}"
            wide.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    print(f"Saved Excel: {excel_path}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser(description="Protocol B: refit decoders on each env variant (MI/linreg/softmax/state).")
    parser.add_argument("name", type=str, nargs="?", default=None)
    parser.add_argument("train_id", type=str)

    # W&B / report
    parser.add_argument("--wandb_project", type=str, default="decoder-protocolB")
    parser.add_argument("--report_dir", type=str, default="report")

    # common sampling / schedule
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--period", type=int, default=100)
    parser.add_argument("--end_episode", type=int, default=-1, help="Agent checkpoint end.")
    parser.add_argument("--approximate", action="store_true")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--train_set", action="store_true")
    parser.add_argument("--valid_size", type=float, default=0.2)

    # ---- Test flags (only one honored) ----
    # TMaze
    parser.add_argument("--test_length", action="store_true")
    parser.add_argument("--test_stochasticity", action="store_true")
    # Hike
    parser.add_argument("--test_variations", action="store_true")
    # Starkweather
    parser.add_argument("--test_p_omission", action="store_true")
    parser.add_argument("--test_bin_size", action="store_true")
    parser.add_argument("--test_iti_hazard", action="store_true")
    parser.add_argument("--test_iti_min", action="store_true")
    parser.add_argument("--test_nITI_microstates", action="store_true")
    # Tiger
    parser.add_argument("--test_listen_accuracy", action="store_true")
    parser.add_argument("--test_reward_listen", action="store_true")
    # GridWorld
    parser.add_argument("--test_grid_size", action="store_true")
    parser.add_argument("--test_tprob", action="store_true")
    # CryBaby
    parser.add_argument("--test_p_cry_if_hungry", action="store_true")
    parser.add_argument("--test_p_cry_if_full", action="store_true")

    # enable decoders (MUST set at least one)
    parser.add_argument("--run_mi", action="store_true")
    parser.add_argument("--run_regression", action="store_true")
    parser.add_argument("--run_softmax_belief", action="store_true")
    parser.add_argument("--run_state_decoder", action="store_true")

    # shared probe hyperparams (used by softmax + state; linreg uses closed form)
    parser.add_argument("--probe_epochs", type=int, default=100)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_batch_size", type=int, default=1024)
    parser.add_argument("--C", type=float, default=1.0)  # for state decoder weight decay analogue

    # standardization common flag (linreg + softmax + state)
    parser.add_argument("--no-standardize", action="store_false", dest="standardize")
    parser.set_defaults(standardize=True)

    # linreg precision
    parser.add_argument("--no-float64", action="store_true",
                        help="Disable float64 in linreg (use float32 only).")

    # softmax probe architecture
    parser.add_argument("--use_mlp_probe", action="store_true")
    parser.add_argument("--mlp_hidden_dim", type=int, default=128)
    parser.add_argument("--mlp_dropout", type=float, default=0.0)

    # MINE params
    parser.add_argument("--mine_num_layers", type=int, default=2)
    parser.add_argument("--mine_hidden_size", type=int, default=256)
    parser.add_argument("--mine_alpha", type=float, default=0.01)
    parser.add_argument("--mine_num_epochs", type=int, default=300)
    parser.add_argument("--mine_batch_size", type=int, default=1024)
    parser.add_argument("--mine_learning_rate", type=float, default=1e-3)
    parser.add_argument("--mine_lambda", type=float, default=0.0)
    parser.add_argument("--representation_size", type=int, default=16)
    parser.add_argument("--belief_part", type=int, default=None)

    args = parser.parse_args()
    print("\n".join(f"{k}={v}" for k, v in vars(args).items()), flush=True)
    main(args)
