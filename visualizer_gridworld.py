# visualizer_gridworld.py
#
# Example:
#   python visualizer_gridworld.py cry w0s6rvqt gridworld --steps 30 --end_episode 120 --period 20 --epsilon 0.0
#
# Output:
#   report/behavior_<name>_<train_id>/gridworld/ep_<E>/t_<t>.png
import glob
from typing import Optional
from fix_decode_eval import SoftmaxProbe, MLPProbe
import os
import random
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from agents.drqn import DRQN
from environments.gridworld import GridWorld
from utils import get_run_statistic


def select_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def onehot(size: int, idx: int, device: torch.device) -> torch.Tensor:
    v = torch.zeros(size, dtype=torch.float32, device=device)
    v[idx] = 1.0
    return v


def get_true_state_xy(env: GridWorld) -> Tuple[int, int]:
    # env.state is GridPos(x,y), both 1-indexed
    s = env.state
    if s is None:
        return (1, 1)
    return (int(s.x), int(s.y))


@torch.no_grad()
def rollout_and_save_snapshots(
    agent: DRQN,
    env: GridWorld,
    device: torch.device,
    epsilon: float,
    steps: int,
    out_dir: str,
    title_prefix: str,
    smx_state: Optional[dict] = None,
):
    agent.Q.eval()

    obs = env.reset()  # obs is length-4 float tensor: [wall_up, wall_down, wall_left, wall_right]
    done = False

    # DRQN input must match training: [a_{t-1}, o_t] only
    a0 = torch.zeros(env.action_size, dtype=torch.float32, device=device)
    o0 = obs.to(device).float()  # shape [O]
    last = torch.cat([a0, o0], dim=0)  # [A+O]
    hidden = None
    reward_cells = [((p.x, p.y), float(r)) for p, r in env.rewards.items()]
    # Save step-0 snapshot (before first action)
    for t in range(steps):
        # --- snapshot BEFORE choosing action at time t ---
        belief = env.get_belief()[0].detach().cpu()  # [K] where K=size^2
        x_true, y_true = get_true_state_xy(env)
        obs_vec = obs.detach().cpu().numpy().astype(np.float32)

        tau_t = last.view(1, 1, -1)  # [1,1,A+O]
        qvals, hidden_next = agent.Q(tau_t, hidden)

        # Extract representation EXACTLY like DRQN.play() / MI pipeline:
        # hidden_next may be tuple/list for LSTM; take h = hidden_next[0]
        h = hidden_next[0] if isinstance(hidden_next, (tuple, list)) else hidden_next
        repr_t = h.detach().flatten().to(device)  # [H_flat]

        # Decode predicted belief if decoder provided
        pred_belief = None
        if smx_state is not None:
            if repr_t.numel() != smx_state["in_dim"]:
                print(f"[warn] skip decode at t={t}: repr_dim={repr_t.numel()} != decoder_in_dim={smx_state['in_dim']}", flush=True)
            else:
                pred_belief = predict_softmax_belief(repr_t.view(1, -1), smx_state)[0].detach().cpu()

        # epsilon-greedy action using qvals
        if random.random() < float(epsilon):
            a = int(env.exploration())
        else:
            a = int(torch.argmax(qvals[0, 0]).item())

        hidden = hidden_next

        # plot + save
        save_path = os.path.join(out_dir, f"t_{t:03d}.png")
        plot_belief_grid_snapshot(
            belief=belief,
            pred_belief=pred_belief,
            grid_size=env.W,  # W==H==size
            true_xy=(x_true, y_true),
            obs_vec=obs_vec,
            action=a,
            reward_cells=reward_cells,
            title=f"{title_prefix} | t={t}",
            save_path=save_path,
        )

        # step env (advance to next obs for t+1)
        obs2, rew, done = env.step(a)
        obs = obs2
        o0 = obs2.to(device).float()

        # next recurrent input: [a_t, o_{t+1}]
        a1 = onehot(env.action_size, a, device=device)
        last = torch.cat([a1, o0], dim=0)  # [A+O]

        if done:
            # still stop early if env terminates
            break

def _find_softmax_path(ep_dir: str, decoder_name: str, part_idx: int) -> Optional[str]:
    p1 = os.path.join(ep_dir, f"{decoder_name}_softmax_b{part_idx}.pth")
    if os.path.isfile(p1):
        return p1
    cands = glob.glob(os.path.join(ep_dir, f"*softmax_b{part_idx}.pth"))
    return cands[0] if len(cands) > 0 else None


def load_softmax_probe(weights_dir: str, train_id: str, episode: int, decoder_name: str, part_idx: int = 0,
                       device: torch.device = torch.device("cpu")) -> Optional[dict]:
    ep_dir = os.path.join(weights_dir, "decoders", train_id, f"ep_{episode}")
    path = _find_softmax_path(ep_dir, decoder_name, part_idx)
    if path is None:
        return None

    payload = torch.load(path, map_location="cpu")
    if payload.get("type") != "softmax_kl":
        raise ValueError(f"Unexpected softmax payload type in {path}: {payload.get('type')}")

    in_dim = int(payload["in_dim"])
    out_dim = int(payload["out_dim"])
    use_mlp = bool(payload["use_mlp"])

    probe = (MLPProbe(in_dim, out_dim, add_bias=True) if use_mlp else SoftmaxProbe(in_dim, out_dim, add_bias=True))
    probe.load_state_dict(payload["probe_state_dict"])
    probe.to(device)
    probe.eval()

    mean = payload["mean"]
    std = payload["std"]
    if mean is not None:
        mean = mean.to(device)
    if std is not None:
        std = std.to(device)

    return {
        "probe": probe,
        "mean": mean,
        "std": std,
        "standardize": bool(payload["standardize"]),
        "path": path,
        "in_dim": in_dim,
        "out_dim": out_dim,
    }


@torch.no_grad()
def predict_softmax_belief(X: torch.Tensor, state: dict) -> torch.Tensor:
    """
    X: [N,H]
    returns: [N,K] probs
    """
    Xn = X
    if state.get("standardize", False):
        Xn = (Xn - state["mean"]) / state["std"]
    logp = state["probe"](Xn)  # returns log-probs
    return torch.exp(logp)

def plot_belief_grid_snapshot(
    belief: torch.Tensor,
    pred_belief: Optional[torch.Tensor],
    grid_size: int,
    true_xy: Tuple[int, int],
    obs_vec: np.ndarray,
    action: int,
    reward_cells: List[Tuple[Tuple[int, int], float]],
    title: str,
    save_path: str,
):
    W = H = int(grid_size)
    K = W * H
    assert belief.numel() == K, f"belief size mismatch: got {belief.numel()}, expected {K}"

    # env state ordering: for x in 1..W: for y in 1..H
    true_map = belief.view(W, H).T.numpy()  # [H,W]

    pred_map = None
    if pred_belief is not None:
        assert pred_belief.numel() == K, f"pred belief mismatch: got {pred_belief.numel()}, expected {K}"
        pred_map = pred_belief.view(W, H).T.numpy()

    x_true, y_true = true_xy
    cx, cy = x_true - 1, y_true - 1

    # Layout: [true heatmap | pred heatmap | text]
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.75], wspace=0.25)

    # Shared vmin/vmax so colors are comparable
    vmin = 0.0
    vmax = float(np.max(true_map)) if pred_map is None else float(max(np.max(true_map), np.max(pred_map)))
    vmax = max(vmax, 1e-8)

    def draw_rewards_and_true(ax):
        # reward cells
        if reward_cells:
            rx = [xy[0] - 1 for (xy, r) in reward_cells]
            ry = [xy[1] - 1 for (xy, r) in reward_cells]
            ax.scatter(rx, ry, marker="s", s=140, facecolors="none", linewidths=2.5)
            for (xy, r) in reward_cells:
                x0, y0 = xy[0] - 1, xy[1] - 1
                ax.text(x0 + 0.15, y0 + 0.15, f"{r:g}", fontsize=10)

        # true state
        ax.scatter([cx], [cy], marker="x", s=140, linewidths=3)

        if W <= 15:
            ax.set_xticks(range(W))
            ax.set_yticks(range(H))
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_xlabel("x (col)")
        ax.set_ylabel("y (row)")

    # True belief heatmap
    ax_true = fig.add_subplot(gs[0, 0])
    im1 = ax_true.imshow(true_map, origin="lower", vmin=vmin, vmax=vmax)
    ax_true.set_title("True belief b(s)")
    draw_rewards_and_true(ax_true)
    cbar1 = fig.colorbar(im1, ax=ax_true, fraction=0.046, pad=0.04)
    cbar1.set_label("belief mass")

    # Predicted belief heatmap
    ax_pred = fig.add_subplot(gs[0, 1])
    if pred_map is None:
        ax_pred.text(0.5, 0.5, "No decoder / dim mismatch", ha="center", va="center", fontsize=12)
        ax_pred.set_axis_off()
    else:
        im2 = ax_pred.imshow(pred_map, origin="lower", vmin=vmin, vmax=vmax)
        ax_pred.set_title("Predicted belief ẑb(s)")
        draw_rewards_and_true(ax_pred)
        cbar2 = fig.colorbar(im2, ax=ax_pred, fraction=0.046, pad=0.04)
        cbar2.set_label("belief mass")

    # Text panel
    ax_text = fig.add_subplot(gs[0, 2])
    ax_text.axis("off")
    obs_str = "[" + ", ".join(f"{float(v):.0f}" for v in obs_vec.tolist()) + "]"
    txt = (
        f"{title}\n\n"
        f"true state (x,y): ({x_true},{y_true})\n"
        f"action: {action}\n"
        f"observation (U,D,L,R): {obs_str}\n"
        f"reward cells shown: {len(reward_cells)}\n"
    )
    ax_text.text(0.0, 1.0, txt, va="top", ha="left", fontsize=12)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(args):
    device = select_device()
    set_seed(args.seed)

    train_args = get_run_statistic(args.train_id)

    # Build GridWorld (bayes=True for exact belief)
    env = GridWorld(
        size=args.size,
        tprob=args.tprob,
        discount=args.discount,
        max_steps=args.horizon,
        bayes=True,
        seed=args.seed,
        reward_scheme=args.reward_scheme,
        reward_margin=args.reward_margin,
        step_cost=args.step_cost,
    )

    agent = DRQN(
        cell=train_args.cell,
        action_size=env.action_size,
        observation_size=env.observation_size,
        num_layers=train_args.num_layers,
        hidden_size=train_args.hidden_size,
    )
    agent.Q.to(device)

    # Determine checkpoints
    if args.end_episode < 0:
        if hasattr(train_args, "episodes"):
            args.end_episode = int(train_args.episodes)
        else:
            raise ValueError("--end_episode must be provided if train_args.episodes is unavailable")

    checkpoints = list(range(0, args.end_episode + 1, args.period))
    if checkpoints[-1] != args.end_episode:
        checkpoints.append(args.end_episode)

    name = args.name if args.name is not None else "noname"
    base_out_dir = os.path.join(args.report_dir, f"behavior_{name}_{args.train_id}", "gridworld")
    os.makedirs(base_out_dir, exist_ok=True)

    for ep in checkpoints:
        agent.load(args.train_id, episode=ep)
        print(f"[loaded] train_id={args.train_id} episode={ep}", flush=True)

        smx_state = None
        if args.overlay_decoders:
            if args.decoder_name is None:
                raise ValueError("--decoder_name is required when --overlay_decoders is set.")
            smx_state = load_softmax_probe(
                args.decoder_weights_dir, args.train_id, ep,
                decoder_name=args.decoder_name, part_idx=0, device=device
            )
            if smx_state is None:
                print(f"[warn] no softmax decoder found for ep={ep}", flush=True)

        ep_dir = os.path.join(base_out_dir, f"ep_{ep}")
        os.makedirs(ep_dir, exist_ok=True)

        title_prefix = (
            f"gridworld | train_id={args.train_id} | episode={ep} | eps={args.epsilon} "
            f"| size={args.size} tprob={args.tprob}"
        )

        rollout_and_save_snapshots(
            agent=agent,
            env=env,
            device=device,
            epsilon=args.epsilon,
            steps=args.steps,
            out_dir=ep_dir,
            title_prefix=title_prefix,
            smx_state=smx_state,
        )

        print(f"[saved] {ep_dir} (up to {args.steps} steps)", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser("GridWorld belief-grid visualizer (per-step snapshots).")

    parser.add_argument("name", type=str, nargs="?", default=None)
    parser.add_argument("train_id", type=str)

    parser.add_argument("--report_dir", type=str, default="report")
    parser.add_argument("--period", type=int, default=500)
    parser.add_argument("--end_episode", type=int, default=-1)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)


    # simulate only this many steps per checkpoint
    parser.add_argument("--steps", type=int, default=30)

    # env params (match GridWorld __init__)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--tprob", type=float, default=0.7)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--reward_scheme", type=str, default="julia",
                      choices=["julia", "symmetric", "center", "scaled"])
    parser.add_argument("--reward_margin", type=int, default=2)
    parser.add_argument("--step_cost", type=float, default=0.0)

    parser.add_argument("--overlay_decoders", action="store_true")
    parser.add_argument("--decoder_name", type=str, default=None)
    parser.add_argument("--decoder_weights_dir", type=str, default="weights")

    args = parser.parse_args()
    print("\n".join(f"{k}={v}" for k, v in vars(args).items()), flush=True)
    main(args)