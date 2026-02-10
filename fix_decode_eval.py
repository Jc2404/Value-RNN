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


def select_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def parse_variant(vname: str):
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
    else:
        raise NotImplementedError(f"Unknown environment {env_name}")

    if getattr(train_args, "irrelevant", 0) != 0:
        env = Irrelevant(env, state_size=train_args.irrelevant, bayes=True)

    return env


def pick_variants(train_args, args):
    env_name = train_args.environment

    if env_name == "tmaze":
        if args.test_length:
            grid = [40, 45, 49, 50, 51, 55, 60]
            return [(f"tmaze_length={L}", {"length": L}) for L in grid]
        if args.test_stochasticity:
            grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
            return [(f"tmaze_stochasticity={s}", {"stochasticity": s}) for s in grid]

    if env_name == "hike":
        if args.test_variations:
            grid = [1, 2, 4, 8]
            return [(f"hike_variations={v}", {"variations": v}) for v in grid]

    if env_name == "starkweather":
        if args.test_p_omission:
            grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
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


def fit_belief_kl_probe(X_train, Y_train, args):
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
             if args.sm_use_mlp else SoftmaxProbe(H, K, add_bias=True).to(device))

    opt = torch.optim.Adam(probe.parameters(), lr=args.probe_lr)
    criterion = nn.KLDivLoss(reduction="batchmean")

    probe.train()
    for _ in range(args.probe_epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, args.probe_batch_size):
            idx = perm[i:i + args.probe_batch_size]
            xb = Xn[idx]
            yb = Y_train[idx]
            logp = probe(xb)
            loss = criterion(logp, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return {"probe": probe, "mean": mean, "std": std, "standardize": args.probe_standardize}


@torch.no_grad()
def eval_belief_kl_probe(X, Y, state):
    if state["standardize"]:
        X = (X - state["mean"]) / state["std"]
    logp = state["probe"](X)
    kl = F.kl_div(logp, Y, reduction="batchmean").item()
    ce = -(Y * logp).sum(dim=-1).mean().item()
    return kl, ce


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
        agent.load(args.train_id, episode=episode)
        print(f"[episode {episode}] agent loaded", flush=True)

        episode_rows[episode] = []

        base_h_belief = base_beliefs = None
        if args.run_mi or args.run_regression or args.run_softmax_belief:
            h, b = generate_hiddens_and_beliefs(
                agent, base_env,
                num_samples=args.probe_num_samples if (args.run_regression or args.run_softmax_belief) else args.mine_num_samples,
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
        if args.run_softmax_belief:
            Xtr, Xte, Btr, Bte = shuffle_split_torch(
                base_h_belief, base_beliefs, args.probe_valid_size, device
            )
            kl_probes = []
            for part_idx, (Ytr, Yte) in enumerate(zip(Btr, Bte)):
                st = fit_belief_kl_probe(Xtr, Ytr, args)
                kl, ce = eval_belief_kl_probe(Xte, Yte, st)
                kl_base[f"KL_b{part_idx}"] = kl
                ce_base[f"CE_b{part_idx}"] = ce
                kl_probes.append(st)
            wandb.log({"train/episode": episode, **{f"belief_softmax/base_KL_b{i}": kl_base[f'KL_b{i}'] for i in range(len(kl_probes))},
                       **{f"belief_softmax/base_CE_b{i}": ce_base[f'CE_b{i}'] for i in range(len(kl_probes))}}, step=episode)

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
        if args.run_softmax_belief:
            base_row.update(kl_base)
            base_row.update(ce_base)
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
            if args.run_softmax_belief:
                hv, bv = generate_hiddens_and_beliefs(
                    agent, venv,
                    num_samples=args.probe_num_samples,
                    epsilon=args.epsilon,
                    approximate=args.approximate,
                )
                hv = hv.to(device)
                bv = tuple(bb.to(device) for bb in bv)

                for part_idx, (bp, st) in enumerate(zip(bv, kl_probes)):
                    kl, ce = eval_belief_kl_probe(hv, bp, st)
                    row[f"KL_b{part_idx}"] = kl
                    row[f"CE_b{part_idx}"] = ce
                    wandb.log({"train/episode": episode,
                               f"belief_softmax/frozen_base_KL_b{part_idx}__{vname}": kl,
                               f"belief_softmax/frozen_base_CE_b{part_idx}__{vname}": ce}, step=episode)

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

    # ---- Experiments
    parser.add_argument("--run_mi", action="store_true")
    parser.add_argument("--run_regression", action="store_true")
    parser.add_argument("--run_softmax_belief", action="store_true")
    parser.add_argument("--run_state_decoder", action="store_true")

    # ---- shared probe flag
    parser.add_argument("--probe_num_samples", type=int, default=10000)
    parser.add_argument("--probe_valid_size", type=float, default=0.2)
    parser.add_argument("--probe_epochs", type=int, default=300)
    parser.add_argument("--probe_lr", type=float, default=1e-2)
    parser.add_argument("--probe_batch_size", type=int, default=1024)
    parser.add_argument("--probe_standardize", action="store_true")
    parser.set_defaults(probe_standardize=True)

    # ---- regression
    parser.add_argument("--reg_no_float64", action="store_true",
                        help="Disable float64 in least squares (float32 only).")

    # ---- belief KL probe
    parser.add_argument("--sm_use_mlp", action="store_true",
                        help="If set, use MLP probe instead of linear for belief KL.")

    # ---- state decoder
    parser.add_argument("--state_C", type=float, default=1.0,
                        help="Inverse regularization strength for sklearn LogisticRegression.")

    # ---- MINE specific ----
    parser.add_argument("--mine_num_samples", type=int, default=10000)
    parser.add_argument("--mine_num_layers", type=int, default=2)
    parser.add_argument("--mine_hidden_size", type=int, default=256)
    parser.add_argument("--mine_alpha", type=float, default=0.01)
    parser.add_argument("--mine_num_epochs", type=int, default=200)
    parser.add_argument("--mine_batch_size", type=int, default=1024)
    parser.add_argument("--mine_learning_rate", type=float, default=1e-3)
    parser.add_argument("--mine_lambda", type=float, default=0.0)
    parser.add_argument("--mine_valid_size", type=float, default=0.2)
    parser.add_argument("--train_set", action="store_true")
    parser.add_argument("--representation_size", type=int, default=16)
    parser.add_argument("--belief_part", type=int, default=None)

    args = parser.parse_args()

    if not (args.run_mi or args.run_regression or args.run_softmax_belief or args.run_state_decoder):
        raise RuntimeError("No analyses enabled. Add at least one flag: "
                           "--run_mi / --run_regression / --run_softmax_belief / --run_state_decoder")

    print("\n".join(f"{k}={v}" for k, v in vars(args).items()), flush=True)
    main(args)
