import os
import json
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from argparse import ArgumentParser

from environments.tmaze import TMaze
from environments.hike import MountainHike
from environments.irrelevant import Irrelevant
from environments.starkweather import StarkweatherEnv
from environments.tiger import Tiger

from agents.drqn import DRQN
from agents.memory import Trajectory
from utils import generate_hiddens_and_beliefs, get_run_statistic


# -------------------------
# Probe models
# -------------------------
class SoftmaxProbe(nn.Module):
    """Linear probe that outputs log-probabilities."""
    def __init__(self, in_dim, out_dim, add_bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=add_bias)

    def forward(self, x):
        logits = self.linear(x)
        return F.log_softmax(logits, dim=-1)


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
        logits = self.net(x)
        return F.log_softmax(logits, dim=-1)


def fit_softmax_probe(
    X,
    Y,
    add_bias=True,
    standardize=True,
    epochs=200,
    lr=1e-2,
    batch_size=1024,
    use_MLP=True,
):
    """
    X: [N, H] hidden states
    Y: [N, K] belief probabilities (rows sum to 1)
    """
    device = X.device
    N, H = X.shape
    K = Y.shape[1]

    if standardize:
        mean = X.mean(0, keepdim=True)
        std = X.std(0, keepdim=True) + 1e-6
        Xn = (X - mean) / std
    else:
        mean, std = None, None
        Xn = X

    if use_MLP:
        print("Using MLP for fitting")
        probe = MLPProbe(H, K, add_bias=add_bias).to(device)
    else:
        print("Using Linear fitting")
        probe = SoftmaxProbe(H, K, add_bias=add_bias).to(device)

    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction="batchmean")

    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        num_batches = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xn[idx]
            yb = Y[idx]

            log_probs = probe(xb)
            loss = criterion(log_probs, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            num_batches += 1

        if (ep + 1) % 50 == 0 or ep == 0:
            avg_kl = total_loss / max(num_batches, 1)
            print(f"[Epoch {ep+1}/{epochs}] train_KL={avg_kl:.4f}")

    return {
        "probe": probe,
        "mean": mean,
        "std": std,
        "standardize": standardize,
    }


def eval_softmax_probe(X, Y, state):
    """
    Returns KL and cross-entropy.
    """
    probe = state["probe"]
    if state["standardize"]:
        Xn = (X - state["mean"]) / state["std"]
    else:
        Xn = X

    with torch.no_grad():
        log_probs = probe(Xn)      # [N,K]
        probs = log_probs.exp()

    kl = F.kl_div(log_probs, Y, reduction="batchmean").item()
    ce = -(Y * log_probs).sum(dim=-1).mean().item()
    return kl, ce, probs, log_probs, Y


# -------------------------
# Env/agent builders
# -------------------------
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
    elif train_args.environment == "tiger":
        env = Tiger(
            bayes=bayes,
            listen_accuracy=getattr(train_args, "listen_accuracy", 0.85),
            reward_listen=getattr(train_args, "reward_listen", -1.0),
            reward_correct=getattr(train_args, "reward_correct", 10.0),
            reward_wrong=getattr(train_args, "reward_wrong", -100.0),
            horizon=getattr(train_args, "horizon", 20),
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


# -------------------------
# Trace logging
# -------------------------
def unwrap_env(env):
    cur = env
    seen = set()
    while True:
        if id(cur) in seen:
            return cur
        seen.add(id(cur))
        if hasattr(cur, "environment"):
            cur = getattr(cur, "environment")
        else:
            return cur


def extract_state(env):
    base = unwrap_env(env)
    d = {"_env_class": base.__class__.__name__}

    # TMaze-like
    for k in ["position", "last_position", "goal_up", "length", "stochasticity"]:
        if hasattr(base, k):
            v = getattr(base, k)
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().tolist()
            d[k] = v

    # Tiger-like
    for k in ["tiger_left", "terminal"]:
        if hasattr(base, k):
            v = getattr(base, k)
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().tolist()
            d[k] = v

    return d


def action_name(env, a):
    base = unwrap_env(env)
    name = base.__class__.__name__.lower()
    if "tmaze" in name:
        return {0: "RIGHT", 1: "UP", 2: "LEFT", 3: "DOWN"}.get(int(a), str(a))
    if "tiger" in name:
        return {0: "LISTEN", 1: "OPEN_LEFT", 2: "OPEN_RIGHT"}.get(int(a), str(a))
    return str(a)


def observation_str(o):
    if not torch.is_tensor(o):
        return str(o)
    o = o.detach().cpu().flatten()
    arg = int(o.argmax().item())
    if o.numel() <= 10:
        return f"argmax={arg} vec={o.tolist()}"
    return f"argmax={arg} (dim={o.numel()})"


def belief_str(env):
    if not hasattr(env, "get_belief"):
        return "<no get_belief()>"
    try:
        b = env.get_belief()
        # TMaze style: tuple of tensors
        if isinstance(b, tuple) and len(b) > 0 and torch.is_tensor(b[0]):
            bb = b[0].detach().cpu().flatten()
            if bb.numel() <= 16:
                return f"dim={bb.numel()} full={bb.tolist()}"
            vals, idx = torch.topk(bb, k=min(8, bb.numel()))
            pairs = ", ".join([f"{i.item()}:{v.item():.4f}" for i, v in zip(idx, vals)])
            return f"dim={bb.numel()} top=[{pairs}]"
        return f"type={type(b)}"
    except Exception as e:
        return f"<error {type(e).__name__}: {e}>"


def append_txt(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def rollout_trace_drqn(agent, environment, steps, epsilon, device):
    """
    Run a rollout for `steps` transitions and log state/action/reward/obs/belief.

    IMPORTANT: we build the DRQN input exactly like training expects:
    use Trajectory.get_last_observed() which contains [prev_action_onehot; obs].
    """
    agent.Q.to(device)
    agent.Q.eval()

    lines = []
    o = environment.reset()

    traj = Trajectory(environment.action_size, environment.observation_size)
    traj.add(None, None, o)

    hidden_states = None
    t = 0
    ep = 0

    while t < steps:
        tau_t = traj.get_last_observed().view(1, 1, -1)

        # Make sure input/hiddens on same device as model
        model_device = next(agent.Q.parameters()).device
        tau_t = tau_t.to(model_device)
        if hidden_states is not None:
            if isinstance(hidden_states, (tuple, list)):
                hidden_states = tuple(h.to(model_device) for h in hidden_states)
            else:
                hidden_states = hidden_states.to(model_device)

        with torch.no_grad():
            values, hidden_states = agent.Q(tau_t, hidden_states)

        # epsilon-greedy
        if random.random() < epsilon:
            a = environment.exploration()
            tag = "explore"
        else:
            a = int(values.flatten().argmax().item())
            tag = "greedy"

        o2, r, d = environment.step(a)

        # log BEFORE adding next step (so log reflects current transition)
        st = extract_state(environment)
        lines.append(
            f"[t={t:03d} ep={ep:02d}] "
            f"state={json.dumps(st, ensure_ascii=False)} | "
            f"a={a}({action_name(environment, a)})[{tag}] | "
            f"r={float(r):+.4f} done={bool(d)} | "
            f"obs={observation_str(o2)} | "
            f"belief={belief_str(environment)}"
        )

        traj.add(a, r, o2, terminal=d)
        t += 1

        if d:
            ep += 1
            o = environment.reset()
            traj = Trajectory(environment.action_size, environment.observation_size)
            traj.add(None, None, o)
            hidden_states = None
            lines.append(f"--- reset -> obs={observation_str(o)} belief={belief_str(environment)} ---")

    return "\n".join(lines) + "\n"


# -------------------------
# Main
# -------------------------
def main(args):
    train_args = get_run_statistic(args.train_id)
    config = vars(train_args) | vars(args)

    wandb.init(
        project="belief-softmax",
        name=args.name,
        config=config,
        save_code=True,
    )
    config = wandb.config

    wandb.save("*.py")
    wandb.save("agents/*.py")
    wandb.save("environments/*.py")

    environment = build_environment_from_train_args(train_args, bayes=True)
    agent = build_agent_from_train_args(train_args, environment)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    print(config.episodes)

    # output txt
    run_id = wandb.run.id if wandb.run is not None else "no_wandb"
    outdir = os.path.join(args.outdir, args.train_id, run_id)
    os.makedirs(outdir, exist_ok=True)
    txt_path = os.path.join(outdir, args.txt_name)
    wandb.config.update({"trace_outdir": outdir, "trace_txt": txt_path}, allow_val_change=True)

    append_txt(txt_path, f"=== softmax trace+probe run_id={run_id} train_id={args.train_id} ===\n")

    for episode in range(0, config.episodes + 1, args.mine_period):
        agent.load(args.train_id, episode=episode)
        print(f"[episode {episode}] agent loaded")

        append_txt(txt_path, f"\n\n########################################\n# CHECKPOINT EPISODE {episode}\n########################################\n")

        # 1) Trace rollout first
        trace = rollout_trace_drqn(
            agent=agent,
            environment=environment,
            steps=args.trace_steps,
            epsilon=args.trace_epsilon,
            device=device,
        )
        append_txt(txt_path, trace)

        # 2) Sample hidden states + beliefs
        hiddens, beliefs = generate_hiddens_and_beliefs(
            agent,
            environment,
            num_samples=args.mine_num_samples,
            epsilon=args.epsilon,
            approximate=args.approximate,
        )

        hiddens = hiddens.to(device)
        beliefs = tuple(b.to(device) for b in beliefs)

        # shuffle + split
        N = hiddens.size(0)
        perm = torch.randperm(N, device=device)
        hiddens = hiddens[perm]
        beliefs = tuple(b[perm] for b in beliefs)
        split = int(N * 0.8)
        X_train, X_test = hiddens[:split], hiddens[split:]

        # 3) Train + evaluate probe
        for part_idx, belief_part in enumerate(beliefs):
            Y_train, Y_test = belief_part[:split], belief_part[split:]

            probe_state = fit_softmax_probe(
                X_train,
                Y_train,
                add_bias=True,
                standardize=args.standardize,
                epochs=args.probe_epochs,
                lr=args.probe_lr,
                batch_size=args.probe_batch_size,
                use_MLP=args.use_MLP,
            )

            kl_test, ce_test, _, _, _ = eval_softmax_probe(X_test, Y_test, probe_state)
            kl_train, ce_train, _, _, _ = eval_softmax_probe(X_train, Y_train, probe_state)

            wandb.log({
                "train/episode": episode,
                f"probe/kl-{part_idx}": kl_test,
                f"probe/ce-{part_idx}": ce_test,
                f"probe/kl_train-{part_idx}": kl_train,
                f"probe/ce_train-{part_idx}": ce_train,
            })

            append_txt(
                txt_path,
                f"METRICS part={part_idx}: "
                f"KL_test={kl_test:.6f} CE_test={ce_test:.6f} "
                f"KL_train={kl_train:.6f} CE_train={ce_train:.6f}\n"
            )

            print(
                f"[episode {episode}] belief {part_idx}: "
                f"KL={kl_test:.4f}, CE={ce_test:.4f}, "
                f"train_KL={kl_train:.4f}, train_CE={ce_train:.4f}"
            )

    wandb.finish()
    print(f"Done. Wrote txt log to: {txt_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Softmax/KL probe with per-checkpoint trace logging")

    parser.add_argument("name", type=str, nargs="?", default=None)
    parser.add_argument("train_id", type=str)

    parser.add_argument("--mine_num_samples", type=int, default=10000)
    parser.add_argument("--mine_period", type=int, default=100)
    parser.add_argument("--approximate", action="store_true")
    parser.add_argument("--epsilon", type=float, default=0.0)

    parser.add_argument("--probe_epochs", type=int, default=300)
    parser.add_argument("--probe_lr", type=float, default=1e-2)
    parser.add_argument("--probe_batch_size", type=int, default=1024)

    parser.add_argument("--use_MLP", type=bool, default=True)

    parser.add_argument("--no-standardize", action="store_false", dest="standardize")
    parser.set_defaults(standardize=True)

    # Trace settings
    parser.add_argument("--trace_steps", type=int, default=100)
    parser.add_argument("--trace_epsilon", type=float, default=0.0)

    # Output
    parser.add_argument("--outdir", type=str, default="regression_weights")
    parser.add_argument("--txt_name", type=str, default="trace_and_metrics.txt")

    args = parser.parse_args()
    main(args)
