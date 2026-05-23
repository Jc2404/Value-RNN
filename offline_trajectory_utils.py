import csv
import json
import os
import re
from random import random
from typing import Dict, List, Sequence, Tuple

import torch


HIDDEN_PREVIEW_STEPS = 10


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_component(value: str) -> str:
    text = str(value)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return cleaned or "value"


def ordered_unique_ints(values: Sequence[int]) -> List[int]:
    seen = set()
    out = []
    for value in values:
        ivalue = int(value)
        if ivalue not in seen:
            seen.add(ivalue)
            out.append(ivalue)
    return out


def add_variant_test_flags(parser) -> None:
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
    parser.add_argument("--test_grid_size", action="store_true")
    parser.add_argument("--test_tprob", action="store_true")
    parser.add_argument("--test_reward_scheme", action="store_true")
    parser.add_argument("--test_reward_margin", action="store_true")
    parser.add_argument("--test_p_cry_if_hungry", action="store_true")
    parser.add_argument("--test_p_cry_if_full", action="store_true")


def add_offline_analysis_flags(parser) -> None:
    parser.add_argument("--run_mi", action="store_true")
    parser.add_argument("--run_regression", action="store_true")
    parser.add_argument("--run_softmax_linear_probe", action="store_true",
                        help="Run the 1-layer linear softmax belief probe.")
    parser.add_argument("--run_softmax_mlp_probe", action="store_true",
                        help="Run the MLP softmax belief probe.")
    parser.add_argument("--valid_size", type=float, default=0.2)

    parser.add_argument("--no-standardize", action="store_false", dest="standardize")
    parser.set_defaults(standardize=True)

    parser.add_argument("--no-float64", action="store_true",
                        help="Disable float64 in linreg (use float32 only).")

    parser.add_argument("--probe_epochs", type=int, default=300)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_batch_size", type=int, default=1024)
    parser.add_argument("--mlp_hidden_dim", type=int, default=128)
    parser.add_argument("--mlp_dropout", type=float, default=0.0)
    parser.add_argument("--belief_loss", choices=["kl", "mse"], default="kl")

    parser.add_argument("--mine_num_layers", type=int, default=2)
    parser.add_argument("--mine_hidden_size", type=int, default=256)
    parser.add_argument("--mine_alpha", type=float, default=0.01)
    parser.add_argument("--mine_num_epochs", type=int, default=400)
    parser.add_argument("--mine_batch_size", type=int, default=1024)
    parser.add_argument("--mine_learning_rate", type=float, default=1e-3)
    parser.add_argument("--mine_lambda", type=float, default=0.0)
    parser.add_argument("--representation_size", type=int, default=16)
    parser.add_argument("--belief_part", type=int, default=None)


def default_cache_dir(report_dir: str, train_id: str) -> str:
    return os.path.join(report_dir, "offline_trajectory_cache", train_id)


def default_artifact_dir(report_dir: str, train_id: str) -> str:
    return os.path.join(report_dir, "offline_replay", train_id)


def cache_file_path(cache_dir: str, generator_episode: int, variant_name: str) -> str:
    return os.path.join(
        cache_dir,
        f"gen_ep_{int(generator_episode)}",
        sanitize_component(variant_name),
        "trajectory_cache.pt",
    )


def pair_artifact_dir(artifact_root: str, generator_episode: int, variant_name: str, evaluator_episode: int) -> str:
    return os.path.join(
        artifact_root,
        f"gen_ep_{int(generator_episode)}",
        sanitize_component(variant_name),
        f"eval_ep_{int(evaluator_episode)}",
    )


def save_csv(path: str, rows: List[Dict]) -> None:
    ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            handle.write("")
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: str, payload) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _flatten_hidden(hidden_states) -> torch.Tensor:
    return hidden_states[0].detach().flatten().cpu().clone()


def collect_offline_trajectory_episode(agent, environment, epsilon: float = 0.0,
                                       preview_steps: int = HIDDEN_PREVIEW_STEPS) -> Dict:
    trajectory_inputs: List[torch.Tensor] = []
    observations: List[torch.Tensor] = []
    actions: List[int] = []
    rewards: List[float] = []
    dones: List[bool] = []
    beliefs_by_part = None
    hidden_preview: List[torch.Tensor] = []

    observation = environment.reset()
    device = next(agent.Q.parameters()).device
    hidden_states = None
    prev_action = torch.zeros(environment.action_size, dtype=observation.dtype)

    for _ in range(environment.horizon()):
        obs_t = observation.detach().clone().cpu()
        belief_t = tuple(part.detach().clone().cpu() for part in environment.get_belief())
        if beliefs_by_part is None:
            beliefs_by_part = [[] for _ in range(len(belief_t))]

        agent_input = torch.cat((prev_action, obs_t), dim=0)
        tau_t = agent_input.view(1, 1, -1).to(device)
        if hidden_states is not None:
            if isinstance(hidden_states, (tuple, list)):
                hidden_states = tuple(h.to(device) for h in hidden_states)
            else:
                hidden_states = hidden_states.to(device)

        with torch.no_grad():
            values, hidden_states = agent.Q(tau_t, hidden_states)

        if len(hidden_preview) < preview_steps:
            hidden_preview.append(_flatten_hidden(hidden_states))

        trajectory_inputs.append(agent_input)
        observations.append(obs_t)
        for part_idx, part in enumerate(belief_t):
            beliefs_by_part[part_idx].append(part)

        if random() < epsilon:
            action = environment.exploration()
        else:
            action = int(values.flatten().argmax().item())

        next_observation, reward, done = environment.step(action)
        actions.append(int(action))
        rewards.append(float(reward))
        dones.append(bool(done))

        prev_action = torch.zeros(environment.action_size, dtype=obs_t.dtype)
        prev_action[action] = 1.0
        observation = next_observation

        if done:
            break

    if not trajectory_inputs:
        raise RuntimeError("Collected an episode with zero decision steps.")

    assert beliefs_by_part is not None
    return {
        "episode_id": None,
        "length": len(trajectory_inputs),
        "agent_inputs": torch.stack(trajectory_inputs),
        "beliefs": tuple(torch.stack(parts) for parts in beliefs_by_part),
        "actions": torch.tensor(actions, dtype=torch.long),
        "observations": torch.stack(observations),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "dones": torch.tensor(dones, dtype=torch.bool),
        "generator_hidden_preview": torch.stack(hidden_preview),
    }


def replay_agent_inputs(agent, agent_inputs: torch.Tensor) -> torch.Tensor:
    hiddens = []
    hidden_states = None
    device = next(agent.Q.parameters()).device

    for step_idx in range(agent_inputs.size(0)):
        tau_t = agent_inputs[step_idx].view(1, 1, -1).to(device)
        if hidden_states is not None:
            if isinstance(hidden_states, (tuple, list)):
                hidden_states = tuple(h.to(device) for h in hidden_states)
            else:
                hidden_states = hidden_states.to(device)

        with torch.no_grad():
            _, hidden_states = agent.Q(tau_t, hidden_states)

        hiddens.append(_flatten_hidden(hidden_states))

    if not hiddens:
        raise RuntimeError("Replayed an episode with zero decision steps.")
    return torch.stack(hiddens)


def verify_hidden_preview(preview: torch.Tensor, replayed_hiddens: torch.Tensor, *,
                          atol: float = 1e-6, rtol: float = 1e-5) -> None:
    if preview.numel() == 0:
        return

    actual = replayed_hiddens[:preview.size(0)].to(preview.dtype)
    if actual.shape != preview.shape:
        raise ValueError(
            f"Hidden preview shape mismatch: preview={tuple(preview.shape)}, "
            f"replayed={tuple(actual.shape)}"
        )
    if not torch.allclose(actual, preview, atol=atol, rtol=rtol):
        max_abs = (actual - preview).abs().max().item()
        raise ValueError(
            f"Offline replay hidden preview mismatch (max_abs_diff={max_abs:.6e})."
        )


def flatten_cached_replay(episodes: Sequence[Dict], replayed_hiddens: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    if len(episodes) != len(replayed_hiddens):
        raise ValueError("Episode and replay counts do not match.")
    if not episodes:
        raise ValueError("Expected at least one cached episode.")

    num_parts = len(episodes[0]["beliefs"])
    belief_parts: List[List[torch.Tensor]] = [[] for _ in range(num_parts)]
    hidden_batches: List[torch.Tensor] = []

    for episode, hidden_seq in zip(episodes, replayed_hiddens):
        if int(episode["length"]) != int(hidden_seq.size(0)):
            raise ValueError(
                f"Replay length mismatch for episode {episode.get('episode_id')}: "
                f"cache={episode['length']}, replay={hidden_seq.size(0)}"
            )
        hidden_batches.append(hidden_seq)
        for part_idx in range(num_parts):
            belief_parts[part_idx].append(episode["beliefs"][part_idx])

    hiddens = torch.cat(hidden_batches, dim=0)
    beliefs = tuple(torch.cat(parts, dim=0) for parts in belief_parts)
    return hiddens, beliefs


def save_linreg_probe(path: str, probe: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save({"type": "linreg", "probe": probe}, path)


def save_softmax_probe(path: str, state: Dict, in_dim: int, out_dim: int, use_mlp: bool) -> None:
    ensure_dir(os.path.dirname(path))
    payload = {
        "type": "softmax_kl",
        "use_mlp": bool(use_mlp),
        "in_dim": int(in_dim),
        "out_dim": int(out_dim),
        "standardize": bool(state["mean"] is not None),
        "mean": state["mean"].detach().cpu() if state["mean"] is not None else None,
        "std": state["std"].detach().cpu() if state["std"] is not None else None,
        "probe_state_dict": {key: value.detach().cpu() for key, value in state["probe"].state_dict().items()},
    }
    torch.save(payload, path)


def save_mine_estimator(path: str, mine, beliefs: Tuple[torch.Tensor, ...], args) -> None:
    ensure_dir(os.path.dirname(path))
    representation_sizes = []
    belief_sizes = []
    for belief_part in beliefs:
        belief_sizes.append(int(belief_part.size(-1)))
        if belief_part.ndim == 2:
            representation_sizes.append(None)
        elif belief_part.ndim == 3:
            representation_sizes.append(int(args.representation_size))
        else:
            raise ValueError("Expected belief tensors to have 2 or 3 dims.")

    payload = {
        "type": "mine",
        "belief_part": args.belief_part,
        "belief_sizes": belief_sizes,
        "representation_sizes": representation_sizes,
        "mine_hidden_size": int(args.mine_hidden_size),
        "mine_num_layers": int(args.mine_num_layers),
        "mine_alpha": float(args.mine_alpha),
        "state_dict": {key: value.detach().cpu() for key, value in mine.state_dict().items()},
    }
    torch.save(payload, path)


def write_cache_manifest(cache_dir: str, rows: List[Dict]) -> str:
    path = os.path.join(cache_dir, "cache_manifest.json")
    save_json(path, rows)
    return path


def load_cache_manifest(cache_dir: str) -> List[Dict]:
    manifest_path = os.path.join(cache_dir, "cache_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
        return rows

    rows = []
    for root, _, files in os.walk(cache_dir):
        for filename in files:
            if filename != "trajectory_cache.pt":
                continue
            cache_path = os.path.join(root, filename)
            payload = torch.load(cache_path)
            meta = payload["metadata"]
            rows.append({
                "cache_path": cache_path,
                "generator_episode": int(meta["generator_episode"]),
                "variant": meta["variant"],
                "task_name": meta.get("task_name"),
                "task_value": meta.get("task_value"),
                "num_samples_collected": int(meta["num_samples_collected"]),
                "num_episodes": int(meta["num_episodes"]),
            })

    rows.sort(key=lambda row: (int(row["generator_episode"]), str(row["variant"])))
    return rows
