# visualize_behavior.py
#
# Usage examples:
#   python visualize_behavior.py myrunid tiger --end_episode 2000 --period 500 --epsilon 0.0 --seed 0
#   python visualize_behavior.py myrunid tiger --listen_accuracy 0.95 --end_episode 5000 --period 1000
#   python visualize_behavior.py myrunid crybaby --p_cry_if_hungry 0.8 --p_cry_if_full 0.2 --end_episode 3000 --period 500
#
# Output:
#   report/behavior_<name>_<train_id>/ep_<E>_<env>.png
import glob
import wandb
import torch.nn.functional as F
from fix_decode_eval import SoftmaxProbe, MLPProbe
import os
import random
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from agents.classic_belief import BeliefPolicy
from agents.drqn import DRQN
from environments.tiger import Tiger
from environments.crybaby import CryingBaby
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


def build_env_from_args(args) -> Tuple[object, List[str], List[str]]:
    """
    Returns:
      env, obs_labels, act_labels
    """
    if args.environment == "tiger":
        env = Tiger(
            bayes=True,
            listen_accuracy=args.listen_accuracy,
            reward_listen=args.reward_listen,
            reward_correct=args.reward_correct,
            reward_wrong=args.reward_wrong,
            horizon=args.horizon,
        )
        obs_labels = ["hear_left", "hear_right", "null"]
        act_labels = ["listen", "open_left", "open_right"]
        return env, obs_labels, act_labels

    if args.environment == "crybaby":
        env = CryingBaby(
            bayes=True,
            p_hungry_if_full_wait=args.p_hungry_if_full_wait,
            p_stay_hungry_wait=args.p_stay_hungry_wait,
            p_full_if_feed=args.p_full_if_feed,
            p_cry_if_hungry=args.p_cry_if_hungry,
            p_cry_if_full=args.p_cry_if_full,
            reward_cry=args.reward_cry,
            cost_feed=args.cost_feed,
            reward_quiet=args.reward_quiet,
            p0_hungry=args.p0_hungry,
            horizon=args.horizon,
        )
        obs_labels = ["cry", "quiet"]
        act_labels = ["wait", "feed"]
        return env, obs_labels, act_labels

    raise ValueError(f"Unsupported environment: {args.environment}")


def get_true_state01(env, env_name: str) -> int:
    # Tiger: env.tiger_left is bool (True=left)
    if env_name == "tiger":
        return 0 if bool(env.tiger_left) else 1
    # CryingBaby: env.state is 0(HUNGRY) / 1(FULL)
    if env_name == "crybaby":
        return int(env.state)
    raise ValueError(env_name)

def _hidden_to_vec(hidden) -> torch.Tensor:
    """
    Match DRQN.play() behavior for MI/decoder training:
      - take hidden_states[0] (h for GRU, h for LSTM)
      - flatten all layers (and batch=1)
    """
    if isinstance(hidden, (tuple, list)):
        # In this repo:
        #   GRU returns (h,)
        #   LSTM returns (h, c)
        h = hidden[0]
    else:
        h = hidden

    return h.detach().flatten()

@torch.no_grad()
def rollout_full(
    agent: DRQN,
    env,
    env_name: str,
    device: torch.device,
    epsilon: float = 0.0,
    total_steps: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    agent.Q.eval()

    if total_steps is None:
        total_steps = int(env.horizon())

    belief_p, state01, obs_idx, act_idx, terminal_after_step = [], [], [], [], []
    hs = []

    obs = env.reset()
    o0 = obs.to(device).float()
    hidden = None
    a_prev = torch.zeros(env.action_size, dtype=torch.float32, device=device)
    last = torch.cat([a_prev, o0], dim=0)  # [A+O]

    for _t in range(total_steps):
        b = env.get_belief()[0]
        belief_p.append(float(b[0].item()))
        state01.append(get_true_state01(env, env_name))
        obs_idx.append(int(o0.argmax().item()))

        tau_t = last.view(1, 1, -1)
        qvals, hidden_next = agent.Q(tau_t, hidden)

        # record representation AFTER consuming current input (aligned with qvals)
        hs.append(_hidden_to_vec(hidden_next).detach().cpu().numpy().astype(np.float32))

        if random.random() < float(epsilon):
            a = int(env.exploration())
        else:
            a = int(torch.argmax(qvals[0, 0]).item())

        act_idx.append(a)
        hidden = hidden_next

        obs2, _rew, done = env.step(a)
        terminal_after_step.append(bool(done))

        a1 = onehot(env.action_size, a, device=device)
        if done:
            obs = env.reset()
            o0 = obs.to(device).float()
            hidden = None
            a_prev = torch.zeros(env.action_size, dtype=torch.float32, device=device)
        else:
            o0 = obs2.to(device).float()
            a_prev = a1

        last = torch.cat([a_prev, o0], dim=0)  # [A+O]

    return {
        "belief_p": np.asarray(belief_p, dtype=np.float32),
        "state01": np.asarray(state01, dtype=np.int64),
        "obs_idx": np.asarray(obs_idx, dtype=np.int64),
        "act_idx": np.asarray(act_idx, dtype=np.int64),
        "terminal_after_step": np.asarray(terminal_after_step, dtype=np.bool_),
        "h": np.asarray(hs, dtype=np.float32),   # <-- CRITICAL
    }


def rollout_belief_policy(
    policy: BeliefPolicy,
    env,
    env_name: str,
    epsilon: float = 0.0,
    total_steps: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    if total_steps is None:
        total_steps = int(env.horizon())

    obs = env.reset()

    belief_p, state01, obs_idx, act_idx, terminal_after_step = [], [], [], [], []
    episode_step = 0

    for t in range(total_steps):
        b = env.get_belief()[0]
        belief_p.append(float(b[0].item()))
        state01.append(get_true_state01(env, env_name))
        obs_idx.append(int(obs.argmax().item()))

        if random.random() < float(epsilon):
            a = int(env.exploration())
        else:
            a = int(policy.act(env, remaining_steps=policy._remaining_steps(env, episode_step)))

        act_idx.append(a)

        obs, _rew, done = env.step(a)
        terminal_after_step.append(bool(done))
        if done:
            obs = env.reset()
            episode_step = 0
        else:
            episode_step += 1

    return {
        "belief_p": np.asarray(belief_p, dtype=np.float32),
        "state01": np.asarray(state01, dtype=np.int64),
        "obs_idx": np.asarray(obs_idx, dtype=np.int64),
        "act_idx": np.asarray(act_idx, dtype=np.int64),
        "terminal_after_step": np.asarray(terminal_after_step, dtype=np.bool_),
    }

def _find_softmax_path(ep_dir: str, decoder_name: str, part_idx: int) -> Optional[str]:
    # Primary (matches train_decoder.py): "{name}_softmax_b{idx}.pth" :contentReference[oaicite:3]{index=3}
    p1 = os.path.join(ep_dir, f"{decoder_name}_softmax_b{part_idx}.pth")
    if os.path.isfile(p1):
        return p1
    # Fallbacks for older naming variants
    cands = glob.glob(os.path.join(ep_dir, f"*softmax_b{part_idx}.pth"))
    return cands[0] if len(cands) > 0 else None


def load_linreg_probe(weights_dir: str, train_id: str, episode: int, part_idx: int = 0) -> Optional[dict]:
    """
    Loads:
      weights/decoders/<train_id>/ep_<E>/linreg_b<i>.pth  :contentReference[oaicite:4]{index=4}
    Returns the internal 'probe' dict saved by fit_linreg_torch.
    """
    ep_dir = os.path.join(weights_dir, "decoders", train_id, f"ep_{episode}")
    path = os.path.join(ep_dir, f"linreg_b{part_idx}.pth")
    if not os.path.isfile(path):
        return None
    payload = torch.load(path, map_location="cpu")
    if payload.get("type") != "linreg":
        raise ValueError(f"Unexpected linreg payload type in {path}: {payload.get('type')}")
    return payload["probe"]


def load_softmax_probe(weights_dir: str, train_id: str, episode: int, decoder_name: str, part_idx: int = 0,
                       device: torch.device = torch.device("cpu")) -> Optional[dict]:
    """
    Loads:
      weights/decoders/<train_id>/ep_<E>/{decoder_name}_softmax_b<i>.pth  :contentReference[oaicite:5]{index=5}

    Returns state dict:
      {"probe": nn.Module, "mean": Tensor|None, "std": Tensor|None, "standardize": bool}
    """
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
    }


@torch.no_grad()
def predict_linreg_belief(X: torch.Tensor, probe: dict) -> torch.Tensor:
    """
    X: [N,H]
    probe: dict from fit_linreg_torch (W, mean, std, add_bias, standardize, use_float64) 
    returns: [N,K] in the simplex (clip+renorm for plotting)
    """
    device = X.device
    Xn = X
    if probe.get("standardize", False):
        Xn = (Xn - probe["mean"].to(device)) / probe["std"].to(device)

    if probe.get("add_bias", True):
        ones = torch.ones(Xn.size(0), 1, device=device, dtype=Xn.dtype)
        Xn = torch.cat([Xn, ones], dim=1)

    W = probe["W"].to(device)
    if probe.get("use_float64", False):
        Yhat = Xn.double() @ W.double()
        Yhat = Yhat.float()
    else:
        Yhat = Xn @ W

    # make it usable as "belief" for visualization (non-negative + renorm)
    Yhat = torch.clamp(Yhat, min=0.0)
    Yhat = Yhat / (Yhat.sum(dim=1, keepdim=True) + 1e-8)
    return Yhat


@torch.no_grad()
def predict_softmax_belief(X: torch.Tensor, state: dict) -> torch.Tensor:
    """
    X: [N,H]
    state: {"probe","mean","std","standardize"}
    returns: [N,K] probs
    """
    Xn = X
    if state.get("standardize", False):
        Xn = (Xn - state["mean"]) / state["std"]
    logp = state["probe"](Xn)  # log_softmax output 
    return torch.exp(logp)

def plot_rollout(
    data: Dict[str, np.ndarray],
    obs_labels: List[str],
    act_labels: List[str],
    title: str,
    save_path: str,
):
    T = len(data["belief_p"])
    x = np.arange(T)
    terminal_steps = np.flatnonzero(data.get("terminal_after_step", np.zeros(T, dtype=np.bool_)))
    belief_terminal_steps = np.flatnonzero(
        data.get("belief_policy_terminal_after_step", np.zeros(T, dtype=np.bool_))
    )

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[2.0, 1.2, 1.4],
        hspace=0.35
    )

    # -------------------------------------------------
    # 1) Belief
    # -------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        x,
        data["belief_p"],
        marker="o",
        linewidth=1.8,
        markersize=3,
        alpha=0.85,
        label="network belief p",
    )
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("belief p")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    decoder_overlays = data.get("decoder_overlays", [])
    decoder_styles = [
        {"color": "tab:green", "linestyle": "-", "alpha": 0.85},
        {"color": "tab:purple", "linestyle": "--", "alpha": 0.8},
    ]
    for idx, overlay in enumerate(decoder_overlays):
        style = decoder_styles[min(idx, len(decoder_styles) - 1)]
        ax1.plot(
            x,
            overlay["values"],
            linewidth=2.0,
            alpha=style["alpha"],
            color=style["color"],
            linestyle=style["linestyle"],
            label=overlay["label"],
        )
    if "belief_policy_p" in data:
        x_belief = np.arange(len(data["belief_policy_p"]))
        ax1.plot(
            x_belief,
            data["belief_policy_p"],
            linewidth=2.0,
            linestyle="--",
            alpha=0.6,
            color="tab:orange",
            label="belief-policy belief p",
        )

    for idx in terminal_steps:
        ax1.axvline(idx + 0.5, color="black", linestyle=":", linewidth=1.0, alpha=0.35)
    for idx in belief_terminal_steps:
        ax1.axvline(idx + 0.5, color="tab:orange", linestyle="--", linewidth=1.0, alpha=0.18)

    ax1.legend(loc="upper right")

    ax_state = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax_state.step(
        x,
        data["act_idx"],
        where="post",
        linewidth=2.5,
        alpha=0.65,
        color="tab:blue",
        label="network action",
    )
    if "belief_policy_act_idx" in data:
        x_belief = np.arange(len(data["belief_policy_act_idx"]))
        ax_state.step(
            x_belief,
            data["belief_policy_act_idx"],
            where="post",
            linewidth=2.5,
            linestyle="--",
            alpha=0.55,
            color="tab:orange",
            label="belief-policy action",
        )
        ax_state.legend(loc="upper right")

    for idx in terminal_steps:
        ax_state.axvline(idx + 0.5, color="black", linestyle=":", linewidth=1.0, alpha=0.35)
    for idx in belief_terminal_steps:
        ax_state.axvline(idx + 0.5, color="tab:orange", linestyle="--", linewidth=1.0, alpha=0.18)

    ax_state.set_ylim(-0.5, len(act_labels) - 0.5)
    ax_state.set_yticks(range(len(act_labels)))
    ax_state.set_yticklabels(act_labels)
    ax_state.set_ylabel("action")
    ax_state.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)

    ax2.step(
        x,
        data["obs_idx"],
        where="post",
        linewidth=2.0,
        alpha=0.7,
        color="red",
        label="observation",
    )

    # true state projected onto same axis
    ax2.step(
        x,
        data["state01"],
        where="post",
        linewidth=2.0,
        alpha=0.7,
        color="blue",
        linestyle="--",
        label="true state",
    )

    for idx in terminal_steps:
        ax2.axvline(idx + 0.5, color="black", linestyle=":", linewidth=1.0, alpha=0.35)
    for idx in belief_terminal_steps:
        ax2.axvline(idx + 0.5, color="tab:orange", linestyle="--", linewidth=1.0, alpha=0.18)

    ax2.set_yticks(range(len(obs_labels)))
    ax2.set_yticklabels(obs_labels)

    ax2.set_ylabel("observation / true state")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax2.set_xlabel("time step (t)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_return_comparison(
    episodes: List[int],
    network_returns: List[float],
    belief_returns: List[float],
    title: str,
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        episodes,
        network_returns,
        marker="o",
        linewidth=2.0,
        alpha=0.85,
        color="tab:blue",
        label="trained network",
    )
    ax.plot(
        episodes,
        belief_returns,
        marker="s",
        linewidth=2.0,
        linestyle="--",
        alpha=0.65,
        color="tab:orange",
        label="belief policy",
    )

    ax.set_xlabel("checkpoint episode")
    ax.set_ylabel("mean return")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(args):
    device = select_device()
    set_seed(args.seed)

    # Load training config so we can instantiate the agent architecture consistently
    train_args = get_run_statistic(args.train_id)

    # Build env from CLI (this script only supports tiger/crybaby)
    env, obs_labels, act_labels = build_env_from_args(args)

    # Agent
    agent = DRQN(
        cell=train_args.cell,
        action_size=env.action_size,
        observation_size=env.observation_size,
        num_layers=train_args.num_layers,
        hidden_size=train_args.hidden_size,
    )
    agent.Q.to(device)
    belief_policy = BeliefPolicy() if args.compare_belief_policy else None

    # Determine checkpoints
    if args.end_episode < 0 or args.end_episode > getattr(train_args, "episodes", args.end_episode):
        # fall back to train_args.episodes if present
        if hasattr(train_args, "episodes"):
            args.end_episode = int(train_args.episodes)

    checkpoints = list(range(0, args.end_episode + 1, args.period))
    if checkpoints[-1] != args.end_episode:
        checkpoints.append(args.end_episode)

    # Output dir
    name = args.name if args.name is not None else "noname"
    out_dir = os.path.join(args.report_dir, f"behavior_{name}_{args.train_id}")
    os.makedirs(out_dir, exist_ok=True)
    network_returns, belief_returns = [], []

    for ep in checkpoints:
        agent.load(args.train_id, episode=ep)
        print(f"[loaded] train_id={args.train_id} episode={ep}", flush=True)

        decoder_states = []

        if args.overlay_decoders:
            decoder_names = args.decoder_name or []
            if len(decoder_names) == 0:
                raise ValueError("At least one --decoder_name is required when --overlay_decoders is set.")
            if len(decoder_names) > 2:
                raise ValueError("Use at most two --decoder_name flags: first for 1-layer, second for MLP.")

            decoder_labels = ["1-layer decoded p", "MLP decoded p"]
            for idx, decoder_name in enumerate(decoder_names):
                state = load_softmax_probe(
                    args.decoder_weights_dir,
                    args.train_id,
                    ep,
                    decoder_name=decoder_name,
                    part_idx=0,
                    device=device,
                )
                if state is None:
                    print(
                        f"[warn] decoder not found for ep={ep}: {decoder_name} under "
                        f"{os.path.join(args.decoder_weights_dir, 'decoders', args.train_id, f'ep_{ep}')}",
                        flush=True,
                    )
                    continue
                decoder_states.append((decoder_labels[idx], state))

        rollout_seed = args.seed + ep
        set_seed(rollout_seed)
        env, obs_labels, act_labels = build_env_from_args(args)
        data = rollout_full(
            agent=agent,
            env=env,
            env_name=args.environment,
            device=device,
            epsilon=args.epsilon,
            total_steps=(args.plot_steps if args.plot_steps > 0 else None),
        )

        if args.compare_belief_policy:
            set_seed(rollout_seed)
            belief_env, _, _ = build_env_from_args(args)
            belief_data = rollout_belief_policy(
                policy=belief_policy,
                env=belief_env,
                env_name=args.environment,
                epsilon=args.epsilon,
                total_steps=(args.plot_steps if args.plot_steps > 0 else None),
            )
            data["belief_policy_p"] = belief_data["belief_p"]
            data["belief_policy_act_idx"] = belief_data["act_idx"]
            data["belief_policy_terminal_after_step"] = belief_data["terminal_after_step"]

        if args.overlay_decoders and "h" in data:
            Htraj = torch.from_numpy(data["h"]).to(device).float()  # [T,H]
            overlays = []
            for label, state in decoder_states:
                p = predict_softmax_belief(Htraj, state)[:, 0].detach().cpu().numpy()
                overlays.append({"label": label, "values": p.astype(np.float32)})
            if overlays:
                data["decoder_overlays"] = overlays

        title = f"{args.environment} | train_id={args.train_id} | episode={ep} | eps-greedy={args.epsilon}"
        save_path = os.path.join(out_dir, f"ep_{ep}_{args.environment}.png")
        plot_rollout(data, obs_labels, act_labels, title, save_path)
        print(f"[saved] {save_path}", flush=True)

        if args.compare_belief_policy:
            eval_seed = args.seed + 100000 + ep

            set_seed(eval_seed)
            eval_env, _, _ = build_env_from_args(args)
            net_mean_return, _ = agent.eval(eval_env, args.return_num_rollouts)
            network_returns.append(float(net_mean_return))

            set_seed(eval_seed)
            belief_eval_env, _, _ = build_env_from_args(args)
            belief_mean_return, _ = belief_policy.eval(belief_eval_env, args.return_num_rollouts)
            belief_returns.append(float(belief_mean_return))

    if args.compare_belief_policy:
        return_path = os.path.join(out_dir, f"returns_{args.environment}.png")
        return_title = (
            f"{args.environment} return comparison | train_id={args.train_id} "
            f"| rollouts={args.return_num_rollouts}"
        )
        plot_return_comparison(
            episodes=checkpoints,
            network_returns=network_returns,
            belief_returns=belief_returns,
            title=return_title,
            save_path=return_path,
        )
        print(f"[saved] {return_path}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser("Visualize belief/state + obs/action over time for Tiger/CryingBaby.")

    parser.add_argument("name", type=str, nargs="?", default=None)
    parser.add_argument("train_id", type=str)

    parser.add_argument("--report_dir", type=str, default="report")
    parser.add_argument("--period", type=int, default=500)
    parser.add_argument("--end_episode", type=int, default=-1)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot_steps", type=int, default=-1,
                        help="Total number of plotted decision steps. If <= 0, uses env.horizon().")

    parser.add_argument("--overlay_decoders", action="store_true",
                        help="Load frozen belief decoders and overlay decoded belief onto true belief.")
    parser.add_argument("--decoder_name", type=str, action="append", default=None,
                        help="Decoder run name. Pass once for the 1-layer decoder, and a second time for the MLP decoder.")
    parser.add_argument("--decoder_weights_dir", type=str, default="weights",
                        help="Root weights directory (expects <dir>/decoders/<train_id>/ep_<E>/...).")
    parser.add_argument("--compare_belief_policy", action="store_true",
                        help="Overlay a belief-policy rollout and plot return-vs-checkpoint against it.")
    parser.add_argument("--return_num_rollouts", type=int, default=100,
                        help="Number of rollouts per checkpoint when plotting network vs belief-policy return.")

    sub = parser.add_subparsers(dest="environment", required=True)

    # --- Tiger
    p_tiger = sub.add_parser("tiger")
    p_tiger.add_argument("--listen_accuracy", type=float, default=0.85)
    p_tiger.add_argument("--reward_listen", type=float, default=-1.0)
    p_tiger.add_argument("--reward_correct", type=float, default=10.0)
    p_tiger.add_argument("--reward_wrong", type=float, default=-100.0)
    p_tiger.add_argument("--horizon", type=int, default=20)

    # --- CryingBaby
    p_cb = sub.add_parser("crybaby")
    p_cb.add_argument("--p_hungry_if_full_wait", type=float, default=0.10)
    p_cb.add_argument("--p_stay_hungry_wait", type=float, default=0.90)
    p_cb.add_argument("--p_full_if_feed", type=float, default=0.95)
    p_cb.add_argument("--p_cry_if_hungry", type=float, default=0.90)
    p_cb.add_argument("--p_cry_if_full", type=float, default=0.10)
    p_cb.add_argument("--p0_hungry", type=float, default=0.50)
    p_cb.add_argument("--reward_cry", type=float, default=-1.0)
    p_cb.add_argument("--reward_quiet", type=float, default=0.0)
    p_cb.add_argument("--cost_feed", type=float, default=-0.2)
    p_cb.add_argument("--horizon", type=int, default=50)

    args = parser.parse_args()
    print("\n".join(f"{k}={v}" for k, v in vars(args).items()), flush=True)
    main(args)
