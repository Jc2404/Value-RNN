# train_frozen_decoders.py
#
# Trains (and saves) frozen decoders on the BASE env for each agent checkpoint:
#   - Linear least-squares belief decoder (R^2)
#   - Softmax/KL belief decoder (KL + CE), optionally MLP via --sm_use_mlp
#
# Parser mirrors the relevant flags from fix_decode_eval.py for compatibility.
#
# Example:
#   python train_frozen_decoders.py my_decoders_run <train_id> \
#       --run_regression --run_softmax_belief \
#       --period 500 --end_episode 5000 \
#       --probe_num_samples 10000 --probe_epochs 200 --probe_batch_size 1024 \
#       --probe_lr 1e-3 --probe_valid_size 0.2 \
#       --epsilon 0.0
#
# Outputs:
#   weights/decoders/<train_id>/ep_<E>/linreg_b<i>.pth
#   weights/decoders/<train_id>/ep_<E>/softmax_b<i>.pth

import os
import csv
from argparse import ArgumentParser
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from agents.drqn import DRQN
from utils import get_run_statistic, generate_hiddens_and_beliefs

# Import "exact" implementations from fix_decode_eval.py
# (keeps behavior aligned with your protocol-A evaluator)
from fix_decode_eval import (  # noqa: E402
    select_device,
    build_environment,
    fit_linreg_torch,
    eval_linreg_torch,
    belief_probe_loss,
    SoftmaxProbe,
    MLPProbe,
    eval_belief_kl_probe,
)

# -----------------------------
# Helpers
# -----------------------------
def shuffle_split_tensors(
    X: torch.Tensor,
    Ys: Tuple[torch.Tensor, ...],
    valid_size: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
    """
    X: [N, ...]
    Ys: tuple of [N, ...]
    """
    N = X.size(0)
    perm = torch.randperm(N, device=device)
    X = X[perm]
    Ys = tuple(y[perm] for y in Ys)

    split = int(N * (1.0 - valid_size))
    Xtr, Xva = X[:split], X[split:]
    Ytr = tuple(y[:split] for y in Ys)
    Yva = tuple(y[split:] for y in Ys)
    return Xtr, Xva, Ytr, Yva


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_rows(path: str, rows: List[Dict]):
    ensure_dir(os.path.dirname(path))
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_linreg_probe(path: str, probe: Dict):
    """
    probe is dict returned by fit_linreg_torch (contains tensors + flags).
    """
    payload = {
        "type": "linreg",
        "probe": probe,
    }
    torch.save(payload, path)


def save_softmax_probe(path: str, state: Dict, in_dim: int, out_dim: int, use_mlp: bool):
    """
    state is {probe, mean, std, standardize} where probe is an nn.Module.
    We save state_dict + standardization stats (tensors).
    """
    payload = {
        "type": "softmax_kl",
        "use_mlp": bool(use_mlp),
        "in_dim": int(in_dim),
        "out_dim": int(out_dim),
        "standardize": bool(state["standardize"]),
        "train_loss": state.get("train_loss", "kl"),
        "mean": state["mean"].detach().cpu() if state["mean"] is not None else None,
        "std": state["std"].detach().cpu() if state["std"] is not None else None,
        "probe_state_dict": {k: v.detach().cpu() for k, v in state["probe"].state_dict().items()},
    }
    torch.save(payload, path)


def train_softmax_kl_with_logging(
    Xtr: torch.Tensor,
    Ytr: torch.Tensor,
    Xva: torch.Tensor,
    Yva: torch.Tensor,
    *,
    args,
    part_idx: int,
    episode: int,
    device: torch.device,
    logger,
) -> Dict:
    """
    Same optimizer/criterion/shuffle structure as fit_belief_kl_probe in fix_decode_eval.py,
    but logs per-epoch train/valid KL+CE to W&B.
    :contentReference[oaicite:1]{index=1}
    """
    N, H = Xtr.shape
    K = Ytr.shape[1]

    # Standardize (train stats only) to match intent (and is what fix_decode_eval does).
    if args.probe_standardize:
        mean = Xtr.mean(0, keepdim=True)
        std = Xtr.std(0, keepdim=True, unbiased=False).clamp_min(1e-6)
        Xtr_n = (Xtr - mean) / std
        Xva_n = (Xva - mean) / std
    else:
        mean, std = None, None
        Xtr_n, Xva_n = Xtr, Xva

    probe: nn.Module = (MLPProbe(H, K, add_bias=True).to(device)
                        if args.sm_use_mlp else SoftmaxProbe(H, K, add_bias=True).to(device))

    opt = torch.optim.Adam(probe.parameters(), lr=args.probe_lr)

    # W&B: separate decoder-step axis, but also attach episode for easy filtering
    # We'll log per epoch with a monotonically increasing integer:
    # global_step = episode * probe_epochs + epoch
    for ep_i in range(args.probe_epochs):
        probe.train()
        perm = torch.randperm(N, device=device)
        running = 0.0
        seen = 0

        for i in range(0, N, args.probe_batch_size):
            idx = perm[i:i + args.probe_batch_size]
            xb = Xtr_n[idx]
            yb = Ytr[idx]
            logp = probe(xb)
            loss = belief_probe_loss(logp, yb, loss_type=args.belief_loss)

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = int(xb.size(0))
            running += float(loss.item()) * bs
            seen += bs

        # eval (train+valid) using the same eval function used in fix_decode_eval
        state = {"probe": probe, "mean": mean, "std": std, "standardize": args.probe_standardize}
        probe.eval()
        with torch.no_grad():
            kl_tr, ce_tr = eval_belief_kl_probe(Xtr, Ytr, state)
            kl_va, ce_va = eval_belief_kl_probe(Xva, Yva, state)

        logger(
            {
                "epoch": ep_i,

                # keep episode as metadata if you want
                "checkpoint_episode": episode,

                # Train vs eval on the same panels
                f"softmax/KL_train_b{part_idx}": kl_tr,
                f"softmax/KL_eval_b{part_idx}": kl_va,
                f"softmax/CE_train_b{part_idx}": ce_tr,
                f"softmax/CE_eval_b{part_idx}": ce_va,

                # Optional: training objective
                f"softmax/loss_train_{args.belief_loss}_b{part_idx}": running / max(seen, 1),
            },
            step=ep_i,
        )
    return {
        "probe": probe,
        "mean": mean,
        "std": std,
        "standardize": args.probe_standardize,
        "train_loss": args.belief_loss,
    }


# -----------------------------
# Main
# -----------------------------
def main(args):
    train_args = get_run_statistic(args.train_id)
    device = select_device()
    print("Device:", device)

    # Build base env (same as protocol-A: base env is train_args env with no overrides)
    base_env = build_environment(train_args, overrides=None)

    if not (args.run_regression or args.run_softmax_belief):
        raise RuntimeError("Nothing to train. Enable at least one of: --run_regression --run_softmax_belief")

    # Episode schedule (same logic as fix_decode_eval.py)
    if args.end_episode < 0 or args.end_episode > train_args.episodes:
        args.end_episode = train_args.episodes

    # Root save dir
    root = os.path.join(args.weights_dir, "decoders", args.train_id)
    if args.decoder_subdir is not None:
        root = os.path.join(root, args.decoder_subdir)
    ensure_dir(root)

    for episode in range(0, args.end_episode + 1, args.period):
        # Load agent checkpoint (fresh instance per ep, same as fix_decode_eval main loop)

        agent = DRQN(
            cell=train_args.cell,
            action_size=base_env.action_size,
            observation_size=base_env.observation_size,
            num_layers=train_args.num_layers,
            hidden_size=train_args.hidden_size,
        )
        agent.load(args.train_id, episode=episode, weights_dir=args.weights_dir)
        print(f"[episode {episode}] agent loaded", flush=True)

        # Sample hidden/beliefs on base env
        h, beliefs = generate_hiddens_and_beliefs(
            agent,
            base_env,
            num_samples=args.probe_num_samples,
            epsilon=args.epsilon,
            approximate=args.approximate,
        )
        h = h.to(device)
        beliefs = tuple(b.to(device) for b in beliefs)

        # Split once; reuse for both decoder types
        Xtr, Xva, Ytr_tuple, Yva_tuple = shuffle_split_tensors(
            h, beliefs, valid_size=args.probe_valid_size, device=device
        )

        # Per-episode output dir
        ep_dir = os.path.join(root, f"ep_{episode}")
        ensure_dir(ep_dir)

        run_base_name = args.name if args.name is not None else "decoder_train"
        metrics_rows: List[Dict] = []

        def log_metrics(payload: Dict, step=None):
            wandb.log(payload, step=step)
            row = dict(payload)
            if step is not None:
                row["step"] = step
            metrics_rows.append(row)

        wandb.init(
            project=args.wandb_project,
            name=f"{run_base_name}_ep{episode}",
            group=args.name,
            job_type="decoder_probe",
            config={**(vars(train_args) | vars(args)), "checkpoint_episode": episode},
            reinit=True,
            save_code=True,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

        # -------------------------
        # (1) Linear regression probes
        # -------------------------
        if args.run_regression:
            for part_idx, (Ytr, Yva) in enumerate(zip(Ytr_tuple, Yva_tuple)):
                probe = fit_linreg_torch(
                    Xtr,
                    Ytr,
                    standardize=args.probe_standardize,
                    add_bias=True,
                    use_float64=(not args.reg_no_float64),
                )

                r2_tr = eval_linreg_torch(Xtr, Ytr, probe)
                r2_va = eval_linreg_torch(Xva, Yva, probe)

                # Log at episode-level (use decoder/global_step = episode*probe_epochs)
                for ep_i in range(args.probe_epochs):
                    log_metrics(
                        {
                            "epoch": ep_i,
                            "checkpoint_episode": episode,
                            f"linreg/R2_train_b{part_idx}": r2_tr,
                            f"linreg/R2_eval_b{part_idx}": r2_va,
                        },
                        step=ep_i,
                    )

                out_path = os.path.join(ep_dir, f"linreg_b{part_idx}.pth")
                save_linreg_probe(out_path, probe)

        # -------------------------
        # (2) Softmax/KL probes
        # -------------------------
        if args.run_softmax_belief:
            for part_idx, (Ytr, Yva) in enumerate(zip(Ytr_tuple, Yva_tuple)):
                state = train_softmax_kl_with_logging(
                    Xtr, Ytr, Xva, Yva,
                    args=args,
                    part_idx=part_idx,
                    episode=episode,
                    device=device,
                    logger=log_metrics,
                )

                # Save weights
                out_path = os.path.join(ep_dir, f"{run_base_name}_softmax_b{part_idx}.pth")
                save_softmax_probe(
                    out_path,
                    state,
                    in_dim=int(Xtr.size(1)),
                    out_dim=int(Ytr.size(1)),
                    use_mlp=bool(args.sm_use_mlp),
                )

        if metrics_rows:
            metrics_path = os.path.join(ep_dir, "metrics.csv")
            save_rows(metrics_path, metrics_rows)
        print(f"[episode {episode}] saved decoders -> {ep_dir}", flush=True)
        wandb.finish()

    print("Done.", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser("Train & save frozen belief decoders per agent checkpoint (compatible with fix_decode_eval flags).")

    # Match fix_decode_eval positional args
    parser.add_argument("name", type=str, nargs="?", default=None)
    parser.add_argument("train_id", type=str)

    # W&B / outputs
    parser.add_argument("--wandb_project", type=str, default="decoder-train")
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--decoder_subdir", type=str, default=None,
                        help="Optional extra subdirectory under weights/decoders/<train_id>/ for this decoder run.")

    # ---- schedule / sampling (same names as fix_decode_eval.py)
    parser.add_argument("--period", type=int, default=100, help="Agent checkpoint interval.")
    parser.add_argument("--end_episode", type=int, default=-1, help="Agent checkpoint end.")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--approximate", action="store_true")

    # ---- enable which decoders to train (reuse fix_decode_eval flags)
    parser.add_argument("--run_regression", action="store_true")
    parser.add_argument("--run_softmax_belief", action="store_true")

    # ---- shared probe hyperparams (same names as fix_decode_eval.py)
    parser.add_argument("--probe_num_samples", type=int, default=10000)
    parser.add_argument("--probe_valid_size", type=float, default=0.2)
    parser.add_argument("--probe_epochs", type=int, default=300)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_batch_size", type=int, default=1024)
    parser.add_argument("--probe_standardize", action="store_true")
    parser.set_defaults(probe_standardize=True)

    # ---- regression
    parser.add_argument("--reg_no_float64", action="store_true",
                        help="Disable float64 in least squares (float32 only).")

    # ---- belief KL probe
    parser.add_argument("--sm_use_mlp", action="store_true",
                        help="If set, use MLP probe instead of linear for belief KL.")
    parser.add_argument("--belief_loss", choices=["kl", "mse"], default="kl",
                        help="Training loss for the softmax belief probe.")

    args = parser.parse_args()
    get_run_statistic(args.train_id)
    main(args)
