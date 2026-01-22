# regression_trace_probe.py
import os
import json
from argparse import ArgumentParser
from datetime import datetime

import torch
import wandb

from environments.tmaze import TMaze
from environments.hike import MountainHike
from environments.irrelevant import Irrelevant
from environments.starkweather import StarkweatherEnv
from environments.tiger import Tiger

from agents.drqn import DRQN
from utils import generate_hiddens_and_beliefs, get_run_statistic


# ----------------------------
# Linear probe (same math)
# ----------------------------
def fit_linear_probe(X, Y, add_bias=True, standardize=True):
    """
    X: [N, H] hidden
    Y: [N, K] belief
    """
    device = X.device
    if standardize:
        mean = X.mean(0, keepdim=True)
        std = X.std(0, keepdim=True) + 1e-6
        Xn = (X - mean) / std
    else:
        mean, std = None, None
        Xn = X

    if add_bias:
        ones = torch.ones(Xn.size(0), 1, device=device)
        Xn = torch.cat([Xn, ones], dim=1)

    res = torch.linalg.lstsq(Xn, Y)
    W = res.solution

    return {
        "W": W.detach().cpu(),
        "mean": None if mean is None else mean.detach().cpu(),
        "std": None if std is None else std.detach().cpu(),
        "add_bias": bool(add_bias),
        "standardize": bool(standardize),
        "x_dim": int(X.size(1)),
        "y_dim": int(Y.size(1)),
    }


def eval_linear_probe(X, Y, probe):
    device = X.device

    Xn = X
    if probe["mean"] is not None:
        mean = probe["mean"].to(device)
        std = probe["std"].to(device)
        Xn = (Xn - mean) / std

    if probe["add_bias"]:
        ones = torch.ones(Xn.size(0), 1, device=device)
        Xn = torch.cat([Xn, ones], dim=1)

    W = probe["W"].to(device)
    Yhat = Xn @ W

    # global R^2
    num = ((Y - Yhat) ** 2).sum()
    den = ((Y - Y.mean(0, keepdim=True)) ** 2).sum()
    rsq = 1 - (num / den)

    mse = torch.mean((Y - Yhat) ** 2)
    return rsq.item(), mse.item(), Yhat


# ----------------------------
# Env/agent builders (match your code)
# ----------------------------
def build_environment_from_train_args(train_args, bayes=True):
    if train_args.environment == "tmaze":
        env = TMaze(
            bayes=bayes,
            length=train_args.length,
            stochasticity=train_args.stochasticity,
        )
    elif train_args.environment == "hike":
        env = MountainHike(
            bayes=bayes,
            variations=train_args.variations,
        )
    elif train_args.environment == "starkweather":
        env = StarkweatherEnv(
            p_omission=train_args.p_omission,
            bin_size=train_args.bin_size,
            iti_hazard=train_args.iti_hazard,
            iti_min=train_args.iti_min,
            nITI_microstates=train_args.nITI_microstates,
        )
    elif train_args.environment == 'tiger':
        env = Tiger(
            listen_accuracy=train_args.listen_accuracy,
            reward_listen=train_args.reward_listen,
            reward_correct=train_args.reward_correct,
            reward_wrong=train_args.reward_wrong,
            horizon=train_args.horizon,
            bayes=bayes,
        )
    else:
        raise NotImplementedError(f"Unknown environment {train_args.environment}")

    if getattr(train_args, "irrelevant", 0) != 0:
        env = Irrelevant(env, state_size=train_args.irrelevant, bayes=bayes)

    return env


def build_agent_from_train_args(train_args, environment):
    if train_args.algorithm != "drqn":
        raise NotImplementedError(f"Unknown algorithm {train_args.algorithm}")

    agent = DRQN(
        cell=train_args.cell,
        action_size=environment.action_size,
        observation_size=environment.observation_size,
        num_layers=train_args.num_layers,
        hidden_size=train_args.hidden_size,
    )
    return agent


# ----------------------------
# Trace utilities
# ----------------------------
def unwrap_env(env):
    """
    Best-effort unwrap (Irrelevant wrapper in this repo stores underlying env in `.environment`).
    """
    cur = env
    seen = set()
    while True:
        if id(cur) in seen:
            return cur
        seen.add(id(cur))
        if hasattr(cur, "environment"):
            cur = getattr(cur, "environment")
            continue
        return cur


def extract_state_dict(env):
    """
    Best-effort 'state' snapshot for inspection.
    This is intentionally heuristic: we log known attributes when present.
    """
    base = unwrap_env(env)
    d = {}

    # TMaze-like
    for k in ["position", "last_position", "goal_up", "length", "stochasticity"]:
        if hasattr(base, k):
            v = getattr(base, k)
            # make JSON-friendly
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().tolist()
            d[k] = v

    # Tiger-like (if you add it)
    for k in ["tiger_left", "terminal", "listen_accuracy"]:
        if hasattr(base, k):
            v = getattr(base, k)
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().tolist()
            d[k] = v

    # fallback: include class name always
    d["_env_class"] = base.__class__.__name__
    return d


def belief_to_str(belief_tensor, topk=8):
    """
    belief_tensor: 1D torch tensor
    Prints top-k indices and probs; if very small dim, prints full.
    """
    b = belief_tensor.detach().float().cpu().flatten()
    K = b.numel()
    if K <= 16:
        return f"dim={K} full={b.tolist()}"
    vals, idx = torch.topk(b, k=min(topk, K))
    pairs = ", ".join([f"{i.item()}:{v.item():.4f}" for i, v in zip(idx, vals)])
    return f"dim={K} top{min(topk,K)}=[{pairs}] sum={b.sum().item():.4f}"


def action_name(env, a):
    """
    Optional action naming for readability for common envs.
    If unknown, returns str(a).
    """
    base = unwrap_env(env)
    cls = base.__class__.__name__.lower()

    if "tmaze" in cls:
        # from tmaze.py: 0=RIGHT,1=UP,2=LEFT,3=DOWN
        names = {0: "RIGHT", 1: "UP", 2: "LEFT", 3: "DOWN"}
        return names.get(int(a), str(a))

    if "tiger" in cls:
        names = {0: "LISTEN", 1: "OPEN_LEFT", 2: "OPEN_RIGHT"}
        return names.get(int(a), str(a))

    return str(a)


def observation_str(o):
    """
    o is one-hot tensor; log argmax + full vector for small dims.
    """
    if not torch.is_tensor(o):
        return f"{o}"
    o = o.detach().cpu().flatten()
    arg = int(o.argmax().item())
    if o.numel() <= 8:
        return f"argmax={arg} vec={o.tolist()}"
    return f"argmax={arg} (dim={o.numel()})"


def rollout_trace(agent, environment, trace_steps, epsilon, device, log_q_values=False):
    lines = []
    agent.Q.to(device)
    agent.Q.eval()

    # reset
    o = environment.reset()
    hidden_states = None
    prev_a = torch.zeros(environment.action_size, device=device)

    lines.append(f"TRACE: steps={trace_steps} epsilon={epsilon} device={device}")
    lines.append(f"RESET obs={observation_str(o)}")
    if hasattr(environment, "get_belief"):
        try:
            b = environment.get_belief()
            if isinstance(b, tuple) and len(b) > 0 and torch.is_tensor(b[0]):
                lines.append(f"RESET belief={belief_to_str(b[0])}")
        except Exception as e:
            lines.append(f"RESET belief=<error {type(e).__name__}: {e}>")
    lines.append("")

    t = 0
    ep = 0
    while t < trace_steps:
        # build DRQN input: [prev_action_onehot; obs]
        obs = o.to(device).view(-1)
        x = torch.cat([prev_a, obs], dim=0)
        tau_t = x.view(1, 1, -1)

        # forward
        with torch.no_grad():
            values, hidden_states = agent.Q(tau_t, hidden_states)
            q = values.flatten().detach()

        # choose action
        if torch.rand(()) < epsilon:
            a = environment.exploration()
            policy_tag = "explore"
        else:
            a = int(q.argmax().item())
            policy_tag = "greedy"

        # step env
        o2, r, d = environment.step(a)

        # update prev action one-hot
        prev_a = torch.zeros(environment.action_size, device=device)
        prev_a[a] = 1.0

        # snapshot for logging
        s = extract_state_dict(environment)

        belief_str = "<no belief>"
        if hasattr(environment, "get_belief"):
            try:
                b = environment.get_belief()
                if isinstance(b, tuple) and len(b) > 0 and torch.is_tensor(b[0]):
                    belief_str = belief_to_str(b[0])
                else:
                    belief_str = str(b)
            except Exception as e:
                belief_str = f"<error {type(e).__name__}: {e}>"

        q_str = ""
        if log_q_values:
            q_str = f" | Q={q.detach().cpu().tolist()}"

        lines.append(
            f"[t={t:03d} ep={ep:02d}] "
            f"state={json.dumps(s, ensure_ascii=False)} | "
            f"a={a}({action_name(environment,a)})[{policy_tag}] | "
            f"r={float(r):+.4f} done={bool(d)} | "
            f"obs={observation_str(o2)} | "
            f"belief={belief_str}"
            f"{q_str}"
        )

        # advance
        t += 1
        o = o2

        # handle terminal/reset
        if d:
            ep += 1
            o = environment.reset()
            hidden_states = None
            prev_a = torch.zeros(environment.action_size, device=device)
            lines.append(f"--- TERMINAL -> RESET obs={observation_str(o)} ---")

    lines.append("TRACE_END\n")
    return "\n".join(lines)



# ----------------------------
# Saving probes + writing txt
# ----------------------------
def save_probe(probe, outdir, train_id, episode, part_idx):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"probe_{train_id}_ep{episode:06d}_part{part_idx}.pt")
    torch.save(probe, path)
    latest = os.path.join(outdir, f"probe_{train_id}_latest_part{part_idx}.pt")
    torch.save(probe, latest)
    return path, latest


def append_to_txt(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def main(args):
    # Retrieve training args for the run
    train_args = get_run_statistic(args.train_id)

    # Merge config for wandb
    config = vars(train_args) | vars(args)

    wandb.init(
        project="belief-regression",
        name=args.name,
        config=config,
        save_code=True,
    )
    config = wandb.config

    wandb.save("*.py")
    wandb.save("agents/*.py")
    wandb.save("environments/*.py")

    # Build env + agent
    environment = build_environment_from_train_args(train_args, bayes=True)
    agent = build_agent_from_train_args(train_args, environment)

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    # Output directory
    run_id = wandb.run.id if wandb.run is not None else "no_wandb"
    outdir = os.path.join(args.outdir, args.train_id, run_id)
    os.makedirs(outdir, exist_ok=True)
    wandb.config.update({"regression_outdir": outdir}, allow_val_change=True)

    # Text log file (single file for entire run)
    txt_path = os.path.join(outdir, args.txt_name)
    append_to_txt(
        txt_path,
        f"=== Regression Trace+Probe Log ===\n"
        f"train_id={args.train_id}\n"
        f"wandb_run_id={run_id}\n"
        f"started={datetime.utcnow().isoformat()}Z\n"
        f"outdir={outdir}\n"
        f"=================================\n\n"
    )

    for episode in range(0, config.episodes + 1, args.period):
        # Load agent checkpoint
        agent.load(args.train_id, episode=episode)
        print("agent expects obs_size =", agent.Q.rnn.input_size)
        o = environment.reset()
        print("env.observation_size =", environment.observation_size)
        print("reset obs dim        =", o.numel())
        print(f"[episode {episode}] agent loaded")

        append_to_txt(
            txt_path,
            f"\n\n########################################\n"
            f"# EPISODE CHECKPOINT: {episode}\n"
            f"########################################\n"
        )

        # ---- 1) TRACE ROLLOUT (before probe train/eval) ----
        trace_text = rollout_trace(
            agent=agent,
            environment=environment,
            trace_steps=args.trace_steps,
            epsilon=args.trace_epsilon,
            device=device,
        )
        append_to_txt(txt_path, trace_text)

        # ---- 2) SAMPLE HIDDENS/BELIEFS ----
        hiddens, beliefs = generate_hiddens_and_beliefs(
            agent,
            environment,
            num_samples=args.num_samples,
            epsilon=args.epsilon,
            approximate=args.approximate,
        )
        append_to_txt(
            txt_path,
            f"SAMPLED: hiddens={tuple(hiddens.shape)} beliefs_parts={len(beliefs)}\n"
        )

        # Move to device
        hiddens = hiddens.to(device)
        beliefs = tuple(b.to(device) for b in beliefs)

        # Shuffle + split
        N = hiddens.size(0)
        perm = torch.randperm(N, device=device)
        hiddens = hiddens[perm]
        beliefs = tuple(b[perm] for b in beliefs)

        split = int(N * 0.8)
        X_train, X_test = hiddens[:split], hiddens[split:]

        # ---- 3) FIT/EVAL per belief part ----
        for part_idx, belief_part in enumerate(beliefs):
            Y_train, Y_test = belief_part[:split], belief_part[split:]

            probe = fit_linear_probe(
                X_train,
                Y_train,
                standardize=args.standardize,
                add_bias=True,
            )

            rsq_train, mse_train, _ = eval_linear_probe(X_train, Y_train, probe)
            rsq_test, mse_test, _ = eval_linear_probe(X_test, Y_test, probe)

            saved_path, latest_path = save_probe(
                probe, outdir, args.train_id, episode, part_idx
            )

            # wandb metrics
            wandb.log({
                "train/episode": episode,
                f"regression/rsq-train-{part_idx}": rsq_train,
                f"regression/mse-train-{part_idx}": mse_train,
                f"regression/rsq-test-{part_idx}": rsq_test,
                f"regression/mse-test-{part_idx}": mse_test,
            })

            # txt metrics
            append_to_txt(
                txt_path,
                f"METRICS part={part_idx}: "
                f"rsq_train={rsq_train:.6f} mse_train={mse_train:.6f} "
                f"rsq_test={rsq_test:.6f} mse_test={mse_test:.6f}\n"
                f"  saved_probe={saved_path}\n"
                f"  latest_probe={latest_path}\n"
            )

            print(
                f"[episode {episode}] part {part_idx} "
                f"rsq_test={rsq_test:.4f} rsq_train={rsq_train:.4f} "
                f"mse_train={mse_train:.6f}"
            )

            if args.wandb_save_weights:
                wandb.save(saved_path)
                wandb.save(latest_path)

    append_to_txt(txt_path, f"\nfinished={datetime.utcnow().isoformat()}Z\n")
    wandb.finish()
    print(f"Done. Trace+metrics log written to: {txt_path}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train linear probes with per-checkpoint rollout trace logging to txt",
    )

    parser.add_argument("name", type=str, nargs="?", default=None)
    parser.add_argument("train_id", type=str)

    # Sampling/probe loop
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--period", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--approximate", action="store_true")

    parser.add_argument("--no-standardize", action="store_false", dest="standardize")
    parser.set_defaults(standardize=True)

    # Trace settings
    parser.add_argument("--trace_steps", type=int, default=100)
    parser.add_argument("--trace_epsilon", type=float, default=0.0,
                        help="Exploration used ONLY for the trace rollout (0=greedy).")

    # Output
    parser.add_argument("--outdir", type=str, default="regression_weights")
    parser.add_argument("--txt_name", type=str, default="trace_and_metrics.txt")
    parser.add_argument("--wandb-save-weights", action="store_true", dest="wandb_save_weights")
    parser.set_defaults(wandb_save_weights=False)

    args = parser.parse_args()
    print("\n".join(f"\033[90m{k}=\033[0m{v}" for k, v in vars(args).items()))

    main(args)
