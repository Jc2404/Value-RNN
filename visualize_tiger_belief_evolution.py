"""
Visualize Tiger belief evolution for a trained DRQN checkpoint.

This script plots, for each selected checkpoint episode:
1. Exact Tiger belief over left/right.
2. Observation overlaid with the true hidden tiger location.
3. Action taken by the agent overlaid with the action chosen by the
   exact belief planner on the same underlying trajectory.
4. Agent outcome, shown either as immediate reward or discounted
   return-to-go within each stitched episode.

If an episode terminates before the requested number of plotted steps,
the script resets the environment and keeps stitching together the next
trajectory so that every saved figure contains exactly `n` decision steps.
Episodes are also truncated at `env.horizon()` to match the standard
rollout logic used elsewhere in this repo.

Examples
--------
python visualize_tiger_belief_evolution.py runs/my_run
python visualize_tiger_belief_evolution.py runs/my_run/agent/train_run_info.json -n 80 --period 400 --end-episode 1600
python visualize_tiger_belief_evolution.py abc123 --weights-dir weights --end-episode 1600 --period 400 -n 80
"""

from __future__ import annotations

import json
import os
import random
import re
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from agents.classic_belief import BeliefPolicy
from agents.drqn import DRQN
from environments.tiger import Tiger


PROJECT_PATH = "jc2404-university-of-cambridge/belief-train_reproduction"
CHECKPOINT_RE = re.compile(r"^(?P<run_id>.+)-(?P<episode>\d+)-(?P<kind>Q|Q_tar)\.pth$")

AXIS_LABEL_FONT_SIZE = 19
TICK_LABEL_FONT_SIZE = 17
LEGEND_FONT_SIZE = 17


def select_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def namespace_from_dict(payload: Dict) -> SimpleNamespace:
    data = dict(payload)
    if "episodes" not in data and "num_episodes" in data:
        data["episodes"] = int(data["num_episodes"])
    return SimpleNamespace(**data)


def resolve_arg(cli_args, train_args, name: str, default=None):
    cli_value = getattr(cli_args, name, None)
    if cli_value is not None:
        return cli_value
    if train_args is not None and hasattr(train_args, name):
        return getattr(train_args, name)
    return default


def parse_checkpoint_name(name: str) -> Optional[Tuple[str, int, str]]:
    match = CHECKPOINT_RE.match(name)
    if match is None:
        return None
    return (
        match.group("run_id"),
        int(match.group("episode")),
        match.group("kind"),
    )


def read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_train_run_info(path: Path) -> Tuple[str, SimpleNamespace, Optional[str]]:
    payload = read_json(path)
    run_id = payload["run_id"]
    weights_dir = payload.get("weights_dir")
    train_args = namespace_from_dict(payload.get("args", {}))
    return run_id, train_args, weights_dir


def load_resolved_config(path: Path) -> Tuple[str, SimpleNamespace, Optional[str]]:
    payload = read_json(path)
    train_run_info = payload.get("train_run_info")
    if not train_run_info:
        raise ValueError(
            f"{path} does not contain a completed 'train_run_info' block."
        )
    run_id = train_run_info["run_id"]
    weights_dir = train_run_info.get("weights_dir") or payload.get("weights_dir")
    train_args = namespace_from_dict(train_run_info.get("args", {}))
    return run_id, train_args, weights_dir


def candidate_metadata_paths(agent_path: Path) -> List[Path]:
    candidates: List[Path] = []

    if agent_path.is_file():
        if agent_path.name == "train_run_info.json":
            candidates.append(agent_path)
        elif agent_path.name == "resolved_config.json":
            candidates.append(agent_path)
        elif agent_path.suffix == ".pth":
            candidates.extend(
                [
                    agent_path.parent.parent / "agent" / "train_run_info.json",
                    agent_path.parent.parent / "resolved_config.json",
                ]
            )
    elif agent_path.is_dir():
        candidates.extend(
            [
                agent_path / "train_run_info.json",
                agent_path / "agent" / "train_run_info.json",
                agent_path / "resolved_config.json",
            ]
        )
        if agent_path.name == "weights":
            candidates.extend(
                [
                    agent_path.parent / "agent" / "train_run_info.json",
                    agent_path.parent / "resolved_config.json",
                ]
            )

    seen = set()
    unique: List[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def infer_run_id_from_path(agent_path: Path) -> Tuple[Optional[str], Optional[str]]:
    if agent_path.is_file():
        parsed = parse_checkpoint_name(agent_path.name)
        if parsed is not None:
            run_id, _episode, _kind = parsed
            return run_id, str(agent_path.parent)
        return None, None

    if not agent_path.is_dir():
        return None, None

    run_ids = set()
    for checkpoint in agent_path.glob("*-Q.pth"):
        parsed = parse_checkpoint_name(checkpoint.name)
        if parsed is None:
            continue
        run_ids.add(parsed[0])

    if len(run_ids) == 1:
        return next(iter(run_ids)), str(agent_path)
    return None, None


def load_train_args_from_wandb(run_id: str) -> SimpleNamespace:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is required when agent_ref is a run id. "
            "Alternatively, pass a local pipeline run folder or train_run_info.json."
        ) from exc

    api = wandb.Api()
    run = api.run(f"{PROJECT_PATH}/{run_id}")
    config = {k: v for k, v in run.config.items() if not k.startswith("_")}
    if "train/episode" in run.summary:
        config["episodes"] = int(run.summary["train/episode"])
    return namespace_from_dict(config)


def resolve_agent_reference(
    agent_ref: str,
    cli_weights_dir: Optional[str],
) -> Tuple[str, SimpleNamespace, str]:
    candidate = Path(agent_ref).expanduser()

    if candidate.exists():
        path = candidate.resolve()

        for meta_path in candidate_metadata_paths(path):
            if not meta_path.is_file():
                continue
            if meta_path.name == "train_run_info.json":
                run_id, train_args, default_weights_dir = load_train_run_info(meta_path)
                weights_dir = cli_weights_dir or default_weights_dir or "weights"
                return run_id, train_args, weights_dir
            if meta_path.name == "resolved_config.json":
                run_id, train_args, default_weights_dir = load_resolved_config(meta_path)
                weights_dir = cli_weights_dir or default_weights_dir or "weights"
                return run_id, train_args, weights_dir

        run_id, inferred_weights_dir = infer_run_id_from_path(path)
        if run_id is not None:
            weights_dir = cli_weights_dir or inferred_weights_dir or "weights"
            return run_id, namespace_from_dict({}), weights_dir

        raise ValueError(
            "Could not resolve agent metadata from the provided path. "
            "Pass a pipeline run folder, train_run_info.json, resolved_config.json, "
            "a weights directory containing one run, or a WandB run id."
        )

    run_id = agent_ref
    train_args = load_train_args_from_wandb(run_id)
    weights_dir = cli_weights_dir or "weights"
    return run_id, train_args, weights_dir


def available_checkpoints(run_id: str, weights_dir: str) -> List[int]:
    weights_path = Path(weights_dir)
    episodes = set()
    for checkpoint in weights_path.glob(f"{run_id}-*-Q.pth"):
        parsed = parse_checkpoint_name(checkpoint.name)
        if parsed is None:
            continue
        episodes.add(parsed[1])
    return sorted(episodes)


def build_checkpoint_schedule(args, available: List[int]) -> List[int]:
    if args.period <= 0:
        raise ValueError("--period must be positive.")

    if not available:
        raise FileNotFoundError(
            f"No checkpoints matching '*-Q.pth' were found in {args.weights_dir!r}."
        )

    start_episode = int(args.start_episode)
    end_episode = max(available) if args.end_episode < 0 else int(args.end_episode)
    if end_episode < start_episode:
        raise ValueError("--end-episode must be >= --start-episode.")

    checkpoints = list(range(start_episode, end_episode + 1, args.period))
    if not checkpoints or checkpoints[-1] != end_episode:
        checkpoints.append(end_episode)

    missing = [episode for episode in checkpoints if episode not in set(available)]
    if missing:
        preview = ", ".join(str(ep) for ep in available[:12])
        if len(available) > 12:
            preview += ", ..."
        raise FileNotFoundError(
            "Some requested checkpoints do not exist in the weights directory. "
            f"Missing: {missing}. Available: [{preview}]"
        )

    return checkpoints


def build_tiger_env(args, train_args) -> Tiger:
    return Tiger(
        listen_accuracy=float(resolve_arg(args, train_args, "listen_accuracy", 0.85)),
        reward_listen=float(resolve_arg(args, train_args, "reward_listen", -1.0)),
        reward_correct=float(resolve_arg(args, train_args, "reward_correct", 10.0)),
        reward_wrong=float(resolve_arg(args, train_args, "reward_wrong", -100.0)),
        horizon=int(resolve_arg(args, train_args, "horizon", 20)),
        bayes=True,
    )


def build_agent(args, train_args, env: Tiger, device: torch.device) -> DRQN:
    agent = DRQN(
        cell=str(resolve_arg(args, train_args, "cell", "gru")),
        action_size=env.action_size,
        observation_size=env.observation_size,
        hidden_size=int(resolve_arg(args, train_args, "hidden_size", 32)),
        num_layers=int(resolve_arg(args, train_args, "num_layers", 2)),
    )
    agent.Q.to(device)
    agent.Q_tar.to(device)
    return agent


def load_checkpoint(
    agent: DRQN,
    run_id: str,
    checkpoint_episode: int,
    weights_dir: str,
    device: torch.device,
) -> None:
    pattern = agent._checkpoint_path(
        run_id,
        episode=checkpoint_episode,
        weights_dir=weights_dir,
    )
    q_path = pattern.format("Q")
    q_tar_path = pattern.format("Q_tar")

    if not os.path.isfile(q_path):
        raise FileNotFoundError(f"Missing checkpoint file: {q_path}")
    if not os.path.isfile(q_tar_path):
        raise FileNotFoundError(f"Missing checkpoint file: {q_tar_path}")

    agent.Q.load_state_dict(torch.load(q_path, map_location=device))
    agent.Q_tar.load_state_dict(torch.load(q_tar_path, map_location=device))
    agent.Q.eval()
    agent.Q_tar.eval()


def one_hot(size: int, index: int, device: torch.device) -> torch.Tensor:
    value = torch.zeros(size, dtype=torch.float32, device=device)
    value[index] = 1.0
    return value


def move_hidden_to_device(hidden, device: torch.device):
    if hidden is None:
        return None
    if isinstance(hidden, (tuple, list)):
        return tuple(part.to(device) for part in hidden)
    return hidden.to(device)


@torch.no_grad()
def rollout_agent_vs_planner(
    agent: DRQN,
    planner: BeliefPolicy,
    env: Tiger,
    device: torch.device,
    *,
    epsilon: float,
    num_steps: int,
) -> Dict[str, np.ndarray]:
    planner._prepare_environment(env)
    planner._reset_cache(env)
    episode_horizon = int(env.horizon())

    obs = env.reset().to(device).float()
    hidden = None
    prev_action = torch.zeros(env.action_size, dtype=torch.float32, device=device)
    last_input = torch.cat([prev_action, obs], dim=0)
    episode_step = 0

    belief_left: List[float] = []
    belief_right: List[float] = []
    true_state: List[int] = []
    obs_idx: List[int] = []
    agent_action: List[int] = []
    planner_action: List[int] = []
    reward: List[float] = []
    terminal_after_step: List[bool] = []

    for _ in range(num_steps):
        belief = planner.extract_planning_belief(env)
        belief_left.append(float(belief[0].item()))
        belief_right.append(float(belief[1].item()))
        true_state.append(0 if bool(env.tiger_left) else 1)
        obs_idx.append(int(obs.argmax().item()))

        tau_t = last_input.view(1, 1, -1)
        q_values, hidden_next = agent.Q(tau_t, move_hidden_to_device(hidden, device))
        q_values = q_values[0, 0]

        if random.random() < epsilon:
            chosen_action = int(env.exploration())
        else:
            chosen_action = int(torch.argmax(q_values).item())

        remaining_steps = planner._remaining_steps(env, episode_step)
        planner_q = planner.q_values(env, belief, remaining_steps)
        planned_action = int(torch.argmax(planner_q).item())

        agent_action.append(chosen_action)
        planner_action.append(planned_action)

        next_obs, step_reward, done = env.step(chosen_action)
        reward.append(float(step_reward))
        reached_horizon = (episode_step + 1) >= episode_horizon
        reset_after_step = bool(done or reached_horizon)
        terminal_after_step.append(reset_after_step)

        if reset_after_step:
            obs = env.reset().to(device).float()
            hidden = None
            prev_action = torch.zeros(env.action_size, dtype=torch.float32, device=device)
            episode_step = 0
        else:
            obs = next_obs.to(device).float()
            hidden = hidden_next
            prev_action = one_hot(env.action_size, chosen_action, device=device)
            episode_step += 1

        last_input = torch.cat([prev_action, obs], dim=0)

    return {
        "belief_left": np.asarray(belief_left, dtype=np.float32),
        "belief_right": np.asarray(belief_right, dtype=np.float32),
        "true_state": np.asarray(true_state, dtype=np.int64),
        "obs_idx": np.asarray(obs_idx, dtype=np.int64),
        "agent_action": np.asarray(agent_action, dtype=np.int64),
        "planner_action": np.asarray(planner_action, dtype=np.int64),
        "reward": np.asarray(reward, dtype=np.float32),
        "terminal_after_step": np.asarray(terminal_after_step, dtype=np.bool_),
    }


def discounted_return_trace(
    rewards: np.ndarray,
    terminal_after_step: np.ndarray,
    gamma: float,
) -> np.ndarray:
    result = np.zeros_like(rewards, dtype=np.float32)
    start = 0

    for idx, terminal in enumerate(terminal_after_step):
        if not terminal:
            continue
        running = 0.0
        for t in range(idx, start - 1, -1):
            running = float(rewards[t]) + float(gamma) * running
            result[t] = running
        start = idx + 1

    if start < len(rewards):
        running = 0.0
        for t in range(len(rewards) - 1, start - 1, -1):
            running = float(rewards[t]) + float(gamma) * running
            result[t] = running

    return result


def outcome_series(
    rollout: Dict[str, np.ndarray],
    gamma: float,
    mode: str,
) -> Tuple[np.ndarray, str]:
    if mode == "reward":
        return rollout["reward"], "reward"
    if mode == "discounted_return":
        values = discounted_return_trace(
            rollout["reward"],
            rollout["terminal_after_step"],
            gamma,
        )
        return values, "discounted return-to-go"
    raise ValueError(f"Unsupported outcome mode: {mode}")


def add_reset_markers(
    axis,
    terminal_after_step: np.ndarray,
) -> None:
    reset_x = np.flatnonzero(terminal_after_step).astype(np.float32) + 0.5
    for x in reset_x:
        axis.axvline(
            x,
            color="0.45",
            linestyle=":",
            linewidth=1.1,
            alpha=0.55,
            zorder=0,
        )


def reset_legend_handle() -> Line2D:
    return Line2D(
        [0],
        [0],
        color="0.45",
        linestyle=":",
        linewidth=1.1,
        alpha=0.55,
        label="environment reset",
    )


def style_axis(axis) -> None:
    axis.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONT_SIZE)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_rollout(
    rollout: Dict[str, np.ndarray],
    gamma: float,
    outcome_mode: str,
    save_path: Path,
) -> None:
    x = np.arange(len(rollout["belief_left"]))
    outcome_values, outcome_label = outcome_series(rollout, gamma, outcome_mode)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(13, 10.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 2.0, 1.2, 1.3]},
    )

    ax_obs, ax_belief, ax_action, ax_outcome = axes

    ax_obs.step(
        x,
        rollout["obs_idx"],
        where="post",
        color="tab:red",
        linewidth=2.0,
        alpha=0.45,
        label=r"$o_t$",
    )
    ax_obs.step(
        x,
        rollout["true_state"],
        where="post",
        color="tab:blue",
        linewidth=2.0,
        linestyle="--",
        alpha=0.45,
        label=r"$s_t$",
    )
    add_reset_markers(ax_obs, rollout["terminal_after_step"])
    ax_obs.set_ylim(-0.25, 1.25)
    ax_obs.set_yticks([0, 1])
    ax_obs.set_yticklabels([r"$o_{\mathrm{hearL}}$", r"$o_{\mathrm{hearR}}$"])
    ax_obs.set_ylabel(r"$o_t \, / \, s_t$", fontsize=AXIS_LABEL_FONT_SIZE)
    obs_handles, obs_labels = ax_obs.get_legend_handles_labels()
    obs_handles.append(reset_legend_handle())
    obs_labels.append("environment reset")
    ax_obs.legend(
        obs_handles,
        obs_labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )
    style_axis(ax_obs)

    ax_belief.plot(
        x,
        rollout["belief_left"],
        color="tab:blue",
        linewidth=2.0,
        alpha=0.9,
        label=r"$P(s_t=\mathrm{L})$",
    )
    add_reset_markers(ax_belief, rollout["terminal_after_step"])
    ax_belief.set_ylim(-0.05, 1.05)
    ax_belief.set_ylabel("belief", fontsize=AXIS_LABEL_FONT_SIZE)
    belief_handles, belief_labels = ax_belief.get_legend_handles_labels()
    belief_handles.append(reset_legend_handle())
    belief_labels.append("environment reset")
    ax_belief.legend(
        belief_handles,
        belief_labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )
    style_axis(ax_belief)

    ax_action.step(
        x,
        rollout["agent_action"],
        where="post",
        color="tab:green",
        linewidth=2.0,
        alpha=0.7,
        label=r"agent $a_t$",
    )
    ax_action.step(
        x,
        rollout["planner_action"],
        where="post",
        color="tab:orange",
        linewidth=2.0,
        linestyle="--",
        alpha=0.6,
        label=r"planner $a_t$",
    )
    add_reset_markers(ax_action, rollout["terminal_after_step"])
    ax_action.set_ylim(-0.25, 2.25)
    ax_action.set_yticks([0, 1, 2])
    ax_action.set_yticklabels(
        [r"$a_{\mathrm{listen}}$", r"$a_{\mathrm{openL}}$", r"$a_{\mathrm{openR}}$"]
    )
    ax_action.set_ylabel(r"$a_t$", fontsize=AXIS_LABEL_FONT_SIZE)
    ax_action.set_xlabel("step", fontsize=AXIS_LABEL_FONT_SIZE)
    action_handles, action_labels = ax_action.get_legend_handles_labels()
    action_handles.append(reset_legend_handle())
    action_labels.append("environment reset")
    ax_action.legend(
        action_handles,
        action_labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )
    style_axis(ax_action)

    ax_outcome.plot(
        x,
        outcome_values,
        color="tab:purple",
        linewidth=2.0,
        alpha=0.85,
        label=outcome_label,
    )
    add_reset_markers(ax_outcome, rollout["terminal_after_step"])
    ax_outcome.set_ylabel("reward", fontsize=AXIS_LABEL_FONT_SIZE)
    ax_outcome.set_xlabel("step", fontsize=AXIS_LABEL_FONT_SIZE)
    outcome_handles, outcome_labels = ax_outcome.get_legend_handles_labels()
    outcome_handles.append(reset_legend_handle())
    outcome_labels.append("environment reset")
    ax_outcome.legend(
        outcome_handles,
        outcome_labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )
    style_axis(ax_outcome)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def sanitize_label(text: str) -> str:
    cleaned = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "run"


def describe_reference(ref: str) -> str:
    return Path(ref).stem if Path(ref).exists() else ref


def ensure_tiger(train_args) -> None:
    if hasattr(train_args, "environment") and train_args.environment not in (None, "tiger"):
        raise ValueError(
            f"This script only supports Tiger checkpoints, but the resolved environment was "
            f"{train_args.environment!r}."
        )


def print_selected_checkpoints(episodes: Iterable[int]) -> None:
    items = ", ".join(str(ep) for ep in episodes)
    print(f"Checkpoint episodes: {items}", flush=True)


def main(args) -> None:
    device = select_device(args.device)
    set_seed(args.seed)

    run_id, train_args, weights_dir = resolve_agent_reference(args.agent_ref, args.weights_dir)
    args.weights_dir = weights_dir
    ensure_tiger(train_args)

    available = available_checkpoints(run_id, weights_dir)
    checkpoints = build_checkpoint_schedule(args, available)

    env_for_shapes = build_tiger_env(args, train_args)
    agent = build_agent(args, train_args, env_for_shapes, device)
    planner = BeliefPolicy(planning_horizon=args.planning_horizon)

    if args.output_dir is None:
        run_label = sanitize_label(describe_reference(run_id))
        output_dir = Path("report") / f"tiger_belief_evolution_{run_label}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Resolved run_id: {run_id}", flush=True)
    print(f"Weights dir: {weights_dir}", flush=True)
    print_selected_checkpoints(checkpoints)
    print(f"Saving figures to: {output_dir.resolve()}", flush=True)

    for checkpoint_episode in checkpoints:
        load_checkpoint(agent, run_id, checkpoint_episode, weights_dir, device)

        rollout_seed = args.seed + checkpoint_episode
        set_seed(rollout_seed)
        env = build_tiger_env(args, train_args)

        rollout = rollout_agent_vs_planner(
            agent,
            planner,
            env,
            device,
            epsilon=float(args.epsilon),
            num_steps=int(args.num_steps),
        )

        save_path = output_dir / f"ep_{checkpoint_episode:06d}_tiger_belief_evolution.png"
        plot_rollout(
            rollout,
            gamma=float(env.gamma),
            outcome_mode=args.outcome_mode,
            save_path=save_path,
        )
        print(f"Saved: {save_path}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Plot Tiger belief evolution, observation vs true state, and "
            "agent action vs belief-planner action for selected checkpoints."
        )
    )
    parser.add_argument(
        "agent_ref",
        type=str,
        help=(
            "WandB run id, pipeline run folder, train_run_info.json, "
            "resolved_config.json, weights directory, or checkpoint file."
        ),
    )

    parser.add_argument("--weights-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--end-episode", "--final-episode", dest="end_episode", type=int, default=-1)
    parser.add_argument("--period", type=int, default=500)
    parser.add_argument("-n", "--num-steps", type=int, default=60)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--planning-horizon", type=int, default=None)
    parser.add_argument(
        "--outcome-mode",
        choices=["reward", "discounted_return"],
        default="reward",
        help="Bottom panel: immediate reward (recommended) or discounted return-to-go.",
    )

    parser.add_argument("--cell", type=str, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)

    parser.add_argument("--listen-accuracy", type=float, default=None)
    parser.add_argument("--reward-listen", type=float, default=None)
    parser.add_argument("--reward-correct", type=float, default=None)
    parser.add_argument("--reward-wrong", type=float, default=None)
    parser.add_argument("--horizon", type=int, default=None)

    parsed_args = parser.parse_args()
    if parsed_args.num_steps <= 0:
        parser.error("--num-steps must be positive.")
    main(parsed_args)
