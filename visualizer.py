# visualize_behavior.py
#
# Usage examples:
#   python visualize_behavior.py myrunid tiger --end_episode 2000 --period 500 --epsilon 0.0 --seed 0
#   python visualize_behavior.py myrunid tiger --listen_accuracy 0.95 --end_episode 5000 --period 1000
#   python visualize_behavior.py myrunid crybaby --p_cry_if_hungry 0.8 --p_cry_if_full 0.2 --end_episode 3000 --period 500
#
# Output:
#   report/behavior_<name>_<train_id>/ep_<E>_<env>.png

import os
import random
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

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
        return 1 if bool(env.tiger_left) else 0
    # CryingBaby: env.state is 0(HUNGRY) / 1(FULL)
    if env_name == "crybaby":
        return int(env.state)
    raise ValueError(env_name)


@torch.no_grad()
def rollout_full(
    agent: DRQN,
    env,
    env_name: str,
    device: torch.device,
    epsilon: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Rollout one episode, recording per time-step (before acting):
      belief_p[t] = belief[0]
      state01[t]  = true hidden state in {0,1}
      obs_idx[t]  = current observation index (before action)
      act_idx[t]  = chosen action index
    """
    agent.Q.eval()

    # reset env
    obs = env.reset()  # one-hot torch tensor (on CPU)
    done = False

    # initial "last observed" vector: [a_onehot, o_onehot, r]
    # action is "none" => zero vector, reward=0
    a0 = torch.zeros(env.action_size, dtype=torch.float32, device=device)
    o0 = obs.to(device).float()
    r0 = torch.zeros(1, dtype=torch.float32, device=device)
    last = torch.cat([a0, o0], dim=0)  # [A+O+1]
    hidden = None

    belief_p: List[float] = []
    state01: List[int] = []
    obs_idx: List[int] = []
    act_idx: List[int] = []

    for _t in range(env.horizon()):
        # record BEFORE taking action
        if not hasattr(env, "get_belief"):
            raise RuntimeError("Env has no get_belief(); need bayes=True env.")
        b = env.get_belief()[0]  # shape [2]
        belief_p.append(float(b[0].item()))
        state01.append(get_true_state01(env, env_name))
        obs_idx.append(int(o0.argmax().item()))

        # epsilon-greedy action
        if random.random() < float(epsilon):
            a = env.exploration()
            a = int(a)
            # still need to advance hidden state consistently
            tau_t = last.view(1, 1, -1)
            _, hidden = agent.Q(tau_t, hidden)
        else:
            tau_t = last.view(1, 1, -1)  # [1,1,D]
            qvals, hidden = agent.Q(tau_t, hidden)  # qvals: [1,1,A]
            a = int(torch.argmax(qvals[0, 0]).item())

        act_idx.append(a)

        # step env
        obs2, rew, done = env.step(a)
        o0 = obs2.to(device).float()

        # update last observed vector for next step
        a1 = onehot(env.action_size, a, device=device)
        r1 = torch.tensor([float(rew)], dtype=torch.float32, device=device)
        last = torch.cat([a1, o0], dim=0)

        if done:
            break

    return {
        "belief_p": np.asarray(belief_p, dtype=np.float32),
        "state01": np.asarray(state01, dtype=np.int64),
        "obs_idx": np.asarray(obs_idx, dtype=np.int64),
        "act_idx": np.asarray(act_idx, dtype=np.int64),
    }


def plot_rollout(
    data: Dict[str, np.ndarray],
    obs_labels: List[str],
    act_labels: List[str],
    title: str,
    save_path: str,
):
    T = len(data["belief_p"])
    x = np.arange(T)

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
        label="belief p (first component)",
    )
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("belief p")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    ax_state = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax_state.step(
        x,
        data["act_idx"],
        where="post",
        linewidth=2.5,
    )
    ax_state.set_ylim(-0.1, 1.1)
    ax_state.set_yticks(range(len(act_labels)))
    ax_state.set_yticklabels(act_labels)
    ax_state.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)

    # Observation
    ax2.step(
        x,
        data["obs_idx"],
        where="post",
        linewidth=2.0,
        alpha=0.7,
        color = 'red',
        label="observation",
    )
    ax2.set_yticks(range(len(obs_labels)))
    ax2.set_yticklabels(obs_labels)
    ax2.set_ylabel("observation")
    ax2.grid(True, alpha=0.3)

    # Action (right axis)
    ax2b = ax2.twinx()
    ax2b.step(
        x,
        data["state01"],
        where="post",
        linewidth=2.0,
        alpha=0.7,
        label="true state",
    )
    ax2b.set_yticks([0, 1])
    ax2b.set_yticklabels(["0", "1"])
    ax2b.set_ylabel("true state")

    ax2.set_xlabel("time step (t)")

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

    for ep in checkpoints:
        agent.load(args.train_id, episode=ep)
        print(f"[loaded] train_id={args.train_id} episode={ep}", flush=True)

        data = rollout_full(
            agent=agent,
            env=env,
            env_name=args.environment,
            device=device,
            epsilon=args.epsilon,
        )

        title = f"{args.environment} | train_id={args.train_id} | episode={ep} | eps-greedy={args.epsilon}"
        save_path = os.path.join(out_dir, f"ep_{ep}_{args.environment}.png")
        plot_rollout(data, obs_labels, act_labels, title, save_path)
        print(f"[saved] {save_path}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser("Visualize belief/state + obs/action over time for Tiger/CryingBaby.")

    parser.add_argument("name", type=str, nargs="?", default=None)
    parser.add_argument("train_id", type=str)

    parser.add_argument("--report_dir", type=str, default="report")
    parser.add_argument("--period", type=int, default=500)
    parser.add_argument("--end_episode", type=int, default=-1)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)

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