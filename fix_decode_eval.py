import os
import wandb
import torch
import pandas as pd
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

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

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from scipy.special import softmax

BELIEF_METRIC_EPS = 1e-12


def select_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def parse_variant(vname: str):
    k, v = vname.split("=")
    try:
        return k, float(v)
    except ValueError:
        return k, v


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
    env_name = train_args.environment

    if env_name == "tmaze":
        if args.test_stochasticity:
            grid = [0.0, 0.1, 0.2, 0.3, 0.4]
            return [(f"tmaze_stochasticity={s}", {"stochasticity": s}) for s in grid]

    elif env_name == "hike":
        if args.test_variations:
            grid = [1, 2, 4, 8]
            return [(f"hike_variations={v}", {"variations": v}) for v in grid]

    elif env_name == "starkweather":
        if args.test_p_omission:
            grid = [0.0, 0.1, 0.2, 0.3, 0.4]
            return [(f"starkweather_p_omission={p}", {"p_omission": p}) for p in grid]
        if args.test_bin_size:
            grid = [train_args.bin_size, max(1, train_args.bin_size // 2), train_args.bin_size * 2]
            out, seen = [], set()
            for b in grid:
                if b not in seen:
                    out.append((f"starkweather_bin_size={b}", {"bin_size": b}))
                    seen.add(b)
            return out
        if args.test_iti_hazard:
            grid = [0.01, 0.05, 0.1, 0.2]
            return [(f"starkweather_iti_hazard={h}", {"iti_hazard": h}) for h in grid]
        if args.test_iti_min:
            grid = [0, 5, 10, 20]
            return [(f"starkweather_iti_min={m}", {"iti_min": m}) for m in grid]
        if args.test_nITI_microstates:
            grid = [1, 2, 4, 8]
            return [(f"starkweather_nITI_microstates={n}", {"nITI_microstates": n}) for n in grid]

    elif env_name == "tiger":
        if args.test_listen_accuracy:
            grid = [0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
            return [(f"tiger_listen_accuracy={a}", {"listen_accuracy": a}) for a in grid]
        if args.test_reward_listen:
            grid = [-5, -3, -1, 0, 1]
            return [(f"tiger_reward_listen={r}", {"reward_listen": r}) for r in grid]
        
    elif env_name == "gridworld":
        if args.test_tprob:
            grid = [0.6, 0.7, 0.8, 0.9, 1.0]
            return [(f"gridworld_tprob={p}", {"tprob": p}) for p in grid]
        if args.test_reward_scheme:
            grid = ["symmetric", "center", "scaled"]
            return [(f"gridworld_reward_scheme={s}", {"reward_scheme": s}) for s in grid]
        if args.test_reward_margin:
            grid = [0, 1, 2, 3, 4]
            return [(f"gridworld_reward_margin={m}", {"reward_margin": m}) for m in grid]
    
    elif env_name == "crybaby":
        if args.test_p_cry_if_hungry:
            grid = [0.6, 0.7, 0.8, 0.9, 1.0]
            return [(f"crybaby_p_cry_if_hungry={p}", {"p_cry_if_hungry": p}) for p in grid]
        if args.test_p_cry_if_full:
            grid = [0.0, 0.1, 0.2, 0.3, 0.4]
            return [(f"crybaby_p_cry_if_full={p}", {"p_cry_if_full": p}) for p in grid]
        
    return []


def shuffle_split_torch(X, Ys, valid_size, device):
    """
    X: torch [N,...]
    Ys: tuple/list of torch [N,...]
    Returns Xtr, Xte, Ys_tr(tuple), Ys_te(tuple)
    """
    N = X.size(0)
    perm = torch.randperm(N, device=device)
    X = X[perm]
    Ys = tuple(y[perm] for y in Ys)
    split = int(N * (1.0 - valid_size))
    return X[:split], X[split:], tuple(y[:split] for y in Ys), tuple(y[split:] for y in Ys)


def build_mine(hiddens, beliefs, args, device):
    belief_sizes, representation_sizes = [], []
    for b in beliefs:
        belief_sizes.append(b.size(-1))
        if b.ndim == 2:
            representation_sizes.append(None)
        elif b.ndim == 3:
            representation_sizes.append(args.representation_size)
        else:
            raise ValueError("Expected belief parts with 2 or 3 dims")

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
    return mine

def fit_linreg_torch(X_train, Y_train, standardize=True, add_bias=True, use_float64=True):
    """
    Faithful intent:
      - StandardScaler on train only (ddof=0)
      - optional bias
      - least squares
    """
    device = X_train.device
    if standardize:
        mean = X_train.mean(0, keepdim=True)
        std = X_train.std(0, keepdim=True, unbiased=False).clamp_min(1e-6)
        Xn = (X_train - mean) / std
    else:
        mean, std = None, None
        Xn = X_train

    if add_bias:
        ones = torch.ones(Xn.size(0), 1, device=device, dtype=Xn.dtype)
        Xn = torch.cat([Xn, ones], dim=1)

    if use_float64:
        Xr = Xn.double()
        Yr = Y_train.double()
    else:
        Xr = Xn
        Yr = Y_train

    res = torch.linalg.lstsq(Xr, Yr)
    W = res.solution  # [H(+1), K]
    return {
        "W": W,
        "mean": mean,
        "std": std,
        "standardize": standardize,
        "add_bias": add_bias,
        "use_float64": use_float64,
    }


@torch.no_grad()
def eval_linreg_torch(X, Y, probe):
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
    return rsq.item()

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


def belief_probe_loss(log_probs, targets, loss_type="kl"):
    if loss_type == "kl":
        return F.kl_div(log_probs, targets, reduction="batchmean")
    if loss_type == "mse":
        return F.mse_loss(log_probs.exp(), targets)
    raise ValueError(f"Unknown belief loss: {loss_type}")


def compute_belief_metrics(target_probs, pred_log_probs):
    pred_probs = pred_log_probs.exp()
    mix_probs = 0.5 * (target_probs + pred_probs)
    mix_log_probs = mix_probs.clamp_min(BELIEF_METRIC_EPS).log()

    kl = F.kl_div(pred_log_probs, target_probs, reduction="batchmean").item()
    ce = -(target_probs * pred_log_probs).sum(dim=-1).mean().item()
    true_entropy = -(target_probs * target_probs.clamp_min(BELIEF_METRIC_EPS).log()).sum(dim=-1).mean().item()
    pred_entropy = -(pred_probs * pred_log_probs).sum(dim=-1).mean().item()
    js = 0.5 * (
        F.kl_div(mix_log_probs, target_probs, reduction="batchmean")
        + F.kl_div(mix_log_probs, pred_probs, reduction="batchmean")
    ).item()

    return {
        "kl": kl,
        "ce": ce,
        "true_entropy": true_entropy,
        "pred_entropy": pred_entropy,
        "js": js,
        "probs": pred_probs,
        "log_probs": pred_log_probs,
    }


def resolve_softmax_probe_specs(args):
    specs = []
    explicit = False

    if getattr(args, "run_softmax_linear_probe", False):
        specs.append({"name": "linear", "use_mlp": False, "legacy_names": False})
        explicit = True
    if getattr(args, "run_softmax_mlp_probe", False):
        specs.append({"name": "mlp", "use_mlp": True, "legacy_names": False})
        explicit = True

    if not explicit and args.run_softmax_belief:
        specs.append({
            "name": "mlp" if args.use_mlp_probe else "linear",
            "use_mlp": args.use_mlp_probe,
            "legacy_names": True,
        })

    return specs


def fit_belief_kl_probe(X_train, Y_train, args, *, use_mlp_probe=False):
    """
    Train once on base; freeze; eval elsewhere.
    """
    device = X_train.device
    N, H = X_train.shape
    K = Y_train.shape[1]

    if args.probe_standardize:
        mean = X_train.mean(0, keepdim=True)
        std = X_train.std(0, keepdim=True, unbiased=False).clamp_min(1e-6)
        Xn = (X_train - mean) / std
    else:
        mean, std = None, None
        Xn = X_train

    probe = (MLPProbe(H, K, add_bias=True).to(device)
             if use_mlp_probe else SoftmaxProbe(H, K, add_bias=True).to(device))

    opt = torch.optim.Adam(probe.parameters(), lr=args.probe_lr)

    probe.train()
    for _ in range(args.probe_epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, args.probe_batch_size):
            idx = perm[i:i + args.probe_batch_size]
            xb = Xn[idx]
            yb = Y_train[idx]
            logp = probe(xb)
            loss = belief_probe_loss(logp, yb, loss_type=args.belief_loss)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return {"probe": probe, "mean": mean, "std": std, "standardize": args.probe_standardize}


@torch.no_grad()
def eval_belief_kl_probe(X, Y, state):
    if state["standardize"]:
        X = (X - state["mean"]) / state["std"]
    logp = state["probe"](X)
    return compute_belief_metrics(Y, logp)


# State decoder
def safelog_np(x):
    y = x.copy()
    y[y == 0] = np.finfo(np.float32).eps
    return np.log(y)


def fit_state_decoder_sklearn(X_train_t, y_train_t, K, standardize=True, C=1.0, class_weight=None):
    """
    Faithful to decode_X_from_y_fit:
      - StandardScaler (fit on train)
      - LogisticRegression multinomial
    """
    X_train = X_train_t.detach().cpu().numpy()
    y_train = y_train_t.detach().cpu().numpy().astype(int)

    if y_train.min() < 0 or y_train.max() >= K:
        raise ValueError(f"y_train out of range for K={K}: "
                         f"min={y_train.min()}, max={y_train.max()}")

    # standardize on train
    if standardize:
        scaler = preprocessing.StandardScaler().fit(X_train)
        Xs = scaler.transform(X_train)
    else:
        scaler = None
        Xs = X_train

    uniq = np.unique(y_train)
    if len(uniq) == 1:
        c0 = int(uniq[0])
        return {
            "scaler": scaler,
            "clf": None,            # signal constant model
            "K": int(K),
            "constant_class": c0,   # always predict this
        }

    clf = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=5000
    )
    clf.fit(Xs, y_train)

    return {"scaler": scaler, "clf": clf, "K": K}


def eval_state_decoder_sklearn(X, y, mdl):
    K = int(mdl["K"])
    clf = mdl["clf"]
    scaler = mdl["scaler"]

    # to numpy
    if hasattr(X, "detach"):
        X = X.detach().cpu().numpy()
    else:
        X = np.asarray(X)

    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()
    else:
        y = np.asarray(y)

    y = y.astype(int).reshape(-1)

    # sanity
    if y.min() < 0 or y.max() >= K:
        raise ValueError(f"y out of range for K={K}: min={y.min()}, max={y.max()}")

    if scaler is not None:
        X = scaler.transform(X)
    N = len(y)
    
    if mdl.get("clf", None) is None:
        c0 = int(mdl["constant_class"])
        p_full = np.zeros((N, K), dtype=np.float32)
        p_full[:, c0] = 1.0
    else:
        clf = mdl["clf"]
        p_trained = clf.predict_proba(X)                 # [N, K_trained]
        trained_classes = clf.classes_.astype(int)       # labels
        p_full = np.zeros((N, K), dtype=p_trained.dtype)
        p_full[:, trained_classes] = p_trained

    yhat = np.argmax(p_full, axis=1)
    pcor = 100.0 * np.mean(yhat == y)

    p_true = p_full[np.arange(N), y]
    LL = safelog_np(p_true).mean()

    classes = np.unique(y)
    phat_mean = np.vstack([
        p_full[y == c].mean(axis=0) if np.any(y == c) else np.zeros(K, dtype=p_full.dtype)
        for c in classes
    ])

    return {"LL": float(LL), "pcor": float(pcor), "phat_mean": phat_mean, "classes": classes}


# Main
def main(args):
    train_args = get_run_statistic(args.train_id)
    device = select_device()
    print("Device:", device)

    variants = pick_variants(train_args, args)
    if not variants:
        raise RuntimeError("No variants selected. Provide exactly ONE --test_* flag for your environment.")

    base_env = build_environment(train_args, overrides=None)

    cfg = vars(train_args) | vars(args)
    wandb.init(project=args.wandb_project, name=args.name, config=cfg, save_code=True)
    wandb.save("*.py")
    wandb.save("agents/*.py")
    wandb.save("environments/*.py")

    os.makedirs(args.report_dir, exist_ok=True)
    excel_path = os.path.join(args.report_dir, f"protocolA_{args.name}_{args.train_id}.xlsx")

    # episode -> list of row dicts
    episode_rows = {}
    if args.end_episode < 0 or args.end_episode > train_args.episodes:
        args.end_episode = train_args.episodes
    for episode in range(0, args.end_episode + 1, args.period):
        # Build agent fresh for safety
        agent = DRQN(
            cell=train_args.cell,
            action_size=base_env.action_size,
            observation_size=base_env.observation_size,
            num_layers=train_args.num_layers,
            hidden_size=train_args.hidden_size,
        )
        agent.load(args.train_id, episode=episode, weights_dir=args.weights_dir)
        print(f"[episode {episode}] agent loaded", flush=True)

        episode_rows[episode] = []

        base_h_belief = base_beliefs = None
        softmax_specs = resolve_softmax_probe_specs(args)

        if args.run_mi or args.run_regression or softmax_specs:
            h, b = generate_hiddens_and_beliefs(
                agent, base_env,
                num_samples=args.probe_num_samples if (args.run_regression or softmax_specs) else args.mine_num_samples,
                epsilon=args.epsilon,
                approximate=args.approximate,
            )
            base_h_belief = h.to(device)
            base_beliefs = tuple(bb.to(device) for bb in b)

        base_h_state = base_states = None
        if args.run_state_decoder:
            hs, st = generate_hiddens_and_states(
                agent, base_env,
                num_samples=args.probe_num_samples,
                epsilon=args.epsilon,
                approximate=args.approximate,
            )
            base_h_state = hs.to(device)
            base_states = st.to(device)
            if base_states.ndim == 2:
                base_states = base_states.argmax(dim=1)

        # MINE
        mine = None
        mi_base = None
        if args.run_mi:
            h_base, b_base = generate_hiddens_and_beliefs(
                agent, base_env,
                num_samples=args.mine_num_samples,
                epsilon=args.epsilon,
                approximate=args.approximate,
            )
            h_base = h_base.to(device)
            b_base = tuple(bb.to(device) for bb in b_base)

            mine = build_mine(h_base, b_base, args, device)
            mine.optimize(
                h_base, b_base,
                num_epochs=args.mine_num_epochs,
                logger=wandb.log,
                learning_rate=args.mine_learning_rate,
                batch_size=args.mine_batch_size,
                lambd=args.mine_lambda,
                valid_size=args.mine_valid_size,
            )

            # base eval (resample unless --train_set)
            if args.train_set:
                h_eval, b_eval = h_base, b_base
            else:
                h_eval, b_eval = generate_hiddens_and_beliefs(
                    agent, base_env,
                    num_samples=args.mine_num_samples,
                    epsilon=args.epsilon,
                    approximate=args.approximate,
                )
                h_eval = h_eval.to(device)
                b_eval = tuple(bb.to(device) for bb in b_eval)

            mi_base = mine.estimate(h_eval, b_eval)
            wandb.log({"train/episode": episode, "mi/base": mi_base}, step=episode)

        # Regression on beliefs
        linreg_probes = None
        reg_base = {}
        if args.run_regression:
            Xtr, Xte, Btr, Bte = shuffle_split_torch(
                base_h_belief, base_beliefs, args.probe_valid_size, device
            )
            linreg_probes = []
            for part_idx, (Ytr, Yte) in enumerate(zip(Btr, Bte)):
                pr = fit_linreg_torch(
                    Xtr, Ytr,
                    standardize=args.probe_standardize,
                    add_bias=True,
                    use_float64=not args.reg_no_float64,
                )
                r2_base = eval_linreg_torch(Xte, Yte, pr)
                reg_base[f"R2_b{part_idx}"] = r2_base
                linreg_probes.append(pr)
            wandb.log({"train/episode": episode, **{f"regression/base_R2_b{i}": v for i, v in enumerate([reg_base[k] for k in sorted(reg_base.keys())])}}, step=episode)

        # Belief KL probe
        kl_probes = None
        kl_base = {}
        ce_base = {}
        true_entropy_base = {}
        pred_entropy_base = {}
        js_base = {}
        if softmax_specs:
            Xtr, Xte, Btr, Bte = shuffle_split_torch(
                base_h_belief, base_beliefs, args.probe_valid_size, device
            )
            kl_probes = {}
            softmax_log = {"train/episode": episode}
            for spec in softmax_specs:
                probes_for_spec = []
                spec_name = spec["name"]
                for part_idx, (Ytr, Yte) in enumerate(zip(Btr, Bte)):
                    st = fit_belief_kl_probe(Xtr, Ytr, args, use_mlp_probe=spec["use_mlp"])
                    metrics = eval_belief_kl_probe(Xte, Yte, st)

                    if spec["legacy_names"]:
                        kl_key = f"KL_b{part_idx}"
                        ce_key = f"CE_b{part_idx}"
                        true_entropy_key = f"H_true_b{part_idx}"
                        pred_entropy_key = f"H_pred_b{part_idx}"
                        js_key = f"JS_b{part_idx}"
                        log_kl_key = f"belief_softmax/base_KL_b{part_idx}"
                        log_ce_key = f"belief_softmax/base_CE_b{part_idx}"
                        log_true_entropy_key = f"belief_softmax/base_H_true_b{part_idx}"
                        log_pred_entropy_key = f"belief_softmax/base_H_pred_b{part_idx}"
                        log_js_key = f"belief_softmax/base_JS_b{part_idx}"
                    else:
                        kl_key = f"KL_{spec_name}_b{part_idx}"
                        ce_key = f"CE_{spec_name}_b{part_idx}"
                        true_entropy_key = f"H_true_{spec_name}_b{part_idx}"
                        pred_entropy_key = f"H_pred_{spec_name}_b{part_idx}"
                        js_key = f"JS_{spec_name}_b{part_idx}"
                        log_kl_key = f"belief_softmax/{spec_name}/base_KL_b{part_idx}"
                        log_ce_key = f"belief_softmax/{spec_name}/base_CE_b{part_idx}"
                        log_true_entropy_key = f"belief_softmax/{spec_name}/base_H_true_b{part_idx}"
                        log_pred_entropy_key = f"belief_softmax/{spec_name}/base_H_pred_b{part_idx}"
                        log_js_key = f"belief_softmax/{spec_name}/base_JS_b{part_idx}"

                    kl_base[kl_key] = metrics["kl"]
                    ce_base[ce_key] = metrics["ce"]
                    true_entropy_base[true_entropy_key] = metrics["true_entropy"]
                    pred_entropy_base[pred_entropy_key] = metrics["pred_entropy"]
                    js_base[js_key] = metrics["js"]
                    softmax_log[log_kl_key] = metrics["kl"]
                    softmax_log[log_ce_key] = metrics["ce"]
                    softmax_log[log_true_entropy_key] = metrics["true_entropy"]
                    softmax_log[log_pred_entropy_key] = metrics["pred_entropy"]
                    softmax_log[log_js_key] = metrics["js"]
                    probes_for_spec.append(st)
                kl_probes[spec_name] = probes_for_spec
            wandb.log(softmax_log, step=episode)

        # State decoder
        state_mdl = None
        state_base = None
        if args.run_state_decoder:
            # split base
            N = base_h_state.size(0)
            perm = torch.randperm(N, device=device)
            hS = base_h_state[perm]
            yS = base_states[perm]
            split = int(N * (1.0 - args.probe_valid_size))
            Xtr, Xte = hS[:split], hS[split:]
            ytr, yte = yS[:split], yS[split:]
            K = base_env.K

            state_mdl = fit_state_decoder_sklearn(
                Xtr, ytr, K, 
                standardize=args.probe_standardize,
                C=args.state_C,
                class_weight=None,
            )
            res_base = eval_state_decoder_sklearn(Xte, yte, state_mdl)
            state_base = {"LL_state": res_base["LL"], "Acc_state": res_base["pcor"]}
            wandb.log({"train/episode": episode,
                       "state_decoder/base_LL": state_base["LL_state"],
                       "state_decoder/base_acc": state_base["Acc_state"]}, step=episode)


        base_row = {"task_name": "base", "task_value": None}
        if args.run_mi:
            base_row["MI"] = mi_base
        if args.run_regression:
            base_row.update(reg_base)
        if softmax_specs:
            base_row.update(kl_base)
            base_row.update(ce_base)
            base_row.update(true_entropy_base)
            base_row.update(pred_entropy_base)
            base_row.update(js_base)
        if args.run_state_decoder:
            base_row.update(state_base)
        episode_rows[episode].append(base_row)

        for vname, overrides in variants:
            venv = build_environment(train_args, overrides=overrides)
            task_name, task_value = parse_variant(vname)

            row = {"task_name": task_name, "task_value": task_value}

            # MI (frozen MINE)
            if args.run_mi:
                hv, bv = generate_hiddens_and_beliefs(
                    agent, venv,
                    num_samples=args.mine_num_samples,
                    epsilon=args.epsilon,
                    approximate=args.approximate,
                )
                hv = hv.to(device)
                bv = tuple(bb.to(device) for bb in bv)

                print("h_base dim:", h_base.size(-1), "h_v dim:", hv.size(-1))
                print("b_base dims:", [bb.size(-1) for bb in b_base])
                print("b_v dims   :", [bb.size(-1) for bb in bv])
                mi_v = mine.estimate(hv, bv)
                row["MI"] = mi_v
                wandb.log({"train/episode": episode,
                           f"mi/frozen_on_base__eval_on/{vname}": mi_v}, step=episode)

            # Regression (frozen)
            if args.run_regression:
                hv, bv = generate_hiddens_and_beliefs(
                    agent, venv,
                    num_samples=args.probe_num_samples,
                    epsilon=args.epsilon,
                    approximate=args.approximate,
                )
                hv = hv.to(device)
                bv = tuple(bb.to(device) for bb in bv)

                # evaluate on FULL variant dataset (no split needed since no fitting)
                for part_idx, (bp, pr) in enumerate(zip(bv, linreg_probes)):
                    r2 = eval_linreg_torch(hv, bp, pr)
                    row[f"R2_b{part_idx}"] = r2
                    wandb.log({"train/episode": episode,
                               f"regression/frozen_base_R2_b{part_idx}__{vname}": r2}, step=episode)

            # Belief KL
            if softmax_specs:
                hv, bv = generate_hiddens_and_beliefs(
                    agent, venv,
                    num_samples=args.probe_num_samples,
                    epsilon=args.epsilon,
                    approximate=args.approximate,
                )
                hv = hv.to(device)
                bv = tuple(bb.to(device) for bb in bv)

                softmax_log = {"train/episode": episode}
                for spec in softmax_specs:
                    spec_name = spec["name"]
                    for part_idx, (bp, st) in enumerate(zip(bv, kl_probes[spec_name])):
                        metrics = eval_belief_kl_probe(hv, bp, st)
                        if spec["legacy_names"]:
                            row[f"KL_b{part_idx}"] = metrics["kl"]
                            row[f"CE_b{part_idx}"] = metrics["ce"]
                            row[f"H_true_b{part_idx}"] = metrics["true_entropy"]
                            row[f"H_pred_b{part_idx}"] = metrics["pred_entropy"]
                            row[f"JS_b{part_idx}"] = metrics["js"]
                            log_kl_key = f"belief_softmax/frozen_base_KL_b{part_idx}__{vname}"
                            log_ce_key = f"belief_softmax/frozen_base_CE_b{part_idx}__{vname}"
                            log_true_entropy_key = f"belief_softmax/frozen_base_H_true_b{part_idx}__{vname}"
                            log_pred_entropy_key = f"belief_softmax/frozen_base_H_pred_b{part_idx}__{vname}"
                            log_js_key = f"belief_softmax/frozen_base_JS_b{part_idx}__{vname}"
                        else:
                            row[f"KL_{spec_name}_b{part_idx}"] = metrics["kl"]
                            row[f"CE_{spec_name}_b{part_idx}"] = metrics["ce"]
                            row[f"H_true_{spec_name}_b{part_idx}"] = metrics["true_entropy"]
                            row[f"H_pred_{spec_name}_b{part_idx}"] = metrics["pred_entropy"]
                            row[f"JS_{spec_name}_b{part_idx}"] = metrics["js"]
                            log_kl_key = f"belief_softmax/{spec_name}/frozen_base_KL_b{part_idx}__{vname}"
                            log_ce_key = f"belief_softmax/{spec_name}/frozen_base_CE_b{part_idx}__{vname}"
                            log_true_entropy_key = f"belief_softmax/{spec_name}/frozen_base_H_true_b{part_idx}__{vname}"
                            log_pred_entropy_key = f"belief_softmax/{spec_name}/frozen_base_H_pred_b{part_idx}__{vname}"
                            log_js_key = f"belief_softmax/{spec_name}/frozen_base_JS_b{part_idx}__{vname}"

                        softmax_log[log_kl_key] = metrics["kl"]
                        softmax_log[log_ce_key] = metrics["ce"]
                        softmax_log[log_true_entropy_key] = metrics["true_entropy"]
                        softmax_log[log_pred_entropy_key] = metrics["pred_entropy"]
                        softmax_log[log_js_key] = metrics["js"]
                wandb.log(softmax_log, step=episode)

            # State decoder
            if args.run_state_decoder:
                hs, st = generate_hiddens_and_states(
                    agent, venv,
                    num_samples=args.probe_num_samples,
                    epsilon=args.epsilon,
                    approximate=args.approximate,
                )
                hs = hs.to(device)
                st = st.to(device)
                if st.ndim == 2:
                    st = st.argmax(dim=1)

                res = eval_state_decoder_sklearn(hs, st, state_mdl)
                row["LL_state"] = res["LL"]
                row["Acc_state"] = res["pcor"]
                wandb.log({"train/episode": episode,
                           f"state_decoder/frozen_base_LL__{vname}": row["LL_state"],
                           f"state_decoder/frozen_base_acc__{vname}": row["Acc_state"]}, step=episode)

            episode_rows[episode].append(row)

        print(f"[episode {episode}] done; rows={len(episode_rows[episode])}", flush=True)


    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for ep, rows in episode_rows.items():
            df = pd.DataFrame(rows)
            df = df.sort_values(by=["task_value"], na_position="first")
            df.to_excel(writer, sheet_name=f"ep_{ep}"[:31], index=False)

    print(f"Saved Excel: {excel_path}")
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser("Protocol A frozen-probe evaluation + Excel per episode (MI + probes).")

    parser.add_argument("name", type=str, nargs="?", default=None)
    parser.add_argument("train_id", type=str)
    parser.add_argument("--wandb_project", type=str, default="protocolA-frozen-all")
    parser.add_argument("--report_dir", type=str, default="report")
    parser.add_argument("--weights_dir", type=str, default="weights")

    # ---- schedule / sampling
    parser.add_argument("--period", type=int, default=100, help="Agent checkpoint interval.")
    parser.add_argument("--end_episode", type=int, default=-1, help="Agent checkpoint end.")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--approximate", action="store_true")

    # ---- variant test flag
    parser.add_argument("--test_length", action="store_true")
    parser.add_argument("--test_stochasticity", action="store_true")
    parser.add_argument("--test_variations", action="store_true")
    parser.add_argument("--test_p_omission", action="store_true")
    parser.add_argument("--test_bin_size", action="store_true")
    parser.add_argument("--test_iti_hazard", action="store_true")
    parser.add_argument("--test_iti_min", action="store_true")
    parser.add_argument("--test_nITI_microstates", action="store_true")
    parser.add_argument("--test_listen_accuracy", action="store_true")
    parser.add_argument("--test_reward_listen", action="store_true")
    parser.add_argument("--test_tprob", action="store_true")
    parser.add_argument("--test_reward_scheme", action="store_true")
    parser.add_argument("--test_reward_margin", action="store_true")
    parser.add_argument("--test_p_cry_if_hungry", action="store_true")
    parser.add_argument("--test_p_cry_if_full", action="store_true")

    # ---- Experiments
    parser.add_argument("--run_mi", action="store_true")
    parser.add_argument("--run_regression", action="store_true")
    parser.add_argument("--run_softmax_belief", action="store_true")
    parser.add_argument("--run_state_decoder", action="store_true")

    # ---- shared probe flag
    parser.add_argument("--probe_num_samples", type=int, default=10000)
    parser.add_argument("--probe_valid_size", type=float, default=0.2)
    parser.add_argument("--probe_epochs", type=int, default=200)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_batch_size", type=int, default=1024)
    parser.add_argument("--probe_standardize", action="store_true")
    parser.set_defaults(probe_standardize=True)

    # ---- regression
    parser.add_argument("--reg_no_float64", action="store_true",
                        help="Disable float64 in least squares (float32 only).")

    # ---- belief KL probe
    parser.add_argument("--run_softmax_linear_probe", action="store_true",
                        help="Run the 1-layer linear softmax belief probe.")
    parser.add_argument("--run_softmax_mlp_probe", action="store_true",
                        help="Run the MLP softmax belief probe.")
    parser.add_argument("--use_mlp_probe", action="store_true",
                        help="If set, use MLP probe instead of linear for belief KL.")
    parser.add_argument("--belief_loss", choices=["kl", "mse"], default="kl",
                        help="Training loss for the softmax belief probe.")

    # ---- state decoder
    parser.add_argument("--state_C", type=float, default=1.0,
                        help="Inverse regularization strength for sklearn LogisticRegression.")

    # ---- MINE specific ----
    parser.add_argument("--mine_num_samples", type=int, default=10000)
    parser.add_argument("--mine_num_layers", type=int, default=2)
    parser.add_argument("--mine_hidden_size", type=int, default=256)
    parser.add_argument("--mine_alpha", type=float, default=0.01)
    parser.add_argument("--mine_num_epochs", type=int, default=300)
    parser.add_argument("--mine_batch_size", type=int, default=1024)
    parser.add_argument("--mine_learning_rate", type=float, default=1e-3)
    parser.add_argument("--mine_lambda", type=float, default=0.0)
    parser.add_argument("--mine_valid_size", type=float, default=0.2)
    parser.add_argument("--train_set", action="store_true")
    parser.add_argument("--representation_size", type=int, default=16)
    parser.add_argument("--belief_part", type=int, default=None)

    args = parser.parse_args()

    softmax_specs = resolve_softmax_probe_specs(args)

    if not (args.run_mi or args.run_regression or softmax_specs or args.run_state_decoder):
        raise RuntimeError("No analyses enabled. Add at least one flag: "
                           "--run_mi / --run_regression / --run_softmax_belief "
                           "/ --run_softmax_linear_probe / --run_softmax_mlp_probe / --run_state_decoder")

    print("\n".join(f"{k}={v}" for k, v in vars(args).items()), flush=True)
    main(args)
