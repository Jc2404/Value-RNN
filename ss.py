import torch
import pandas as pd


# Change this path if needed
PT_PATH = r"D:\Personal folder\University\Projects\4th year\belief-rnn\cache\tmaze_stoch_05250226\gen_ep_0\tmaze_stochasticity_0.0\trajectory_cache.pt"
# Or use absolute path, e.g.
# PT_PATH = r"D:\Personal folder\University\Projects\belief-rnn\trajectory_cache.pt"


def load_cache(path):
    """
    Load a PyTorch .pt trajectory cache safely on CPU.
    """
    try:
        cache = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # For older PyTorch versions without weights_only
        cache = torch.load(path, map_location="cpu")
    return cache


def print_cache_summary(cache):
    print("Top-level type:", type(cache))
    print("Top-level keys:", list(cache.keys()))

    print("\n=== Metadata ===")
    metadata = cache.get("metadata", {})
    for k, v in metadata.items():
        print(f"{k}: {v}")

    episodes = cache["episodes"]
    print("\n=== Episodes ===")
    print("num episodes:", len(episodes))

    ep0 = episodes[0]
    print("\nFirst episode keys:")
    for k in ep0.keys():
        v = ep0[k]
        if torch.is_tensor(v):
            print(f"{k}: tensor {tuple(v.shape)}")
        elif isinstance(v, list):
            print(f"{k}: list length {len(v)}")
            if len(v) > 0 and torch.is_tensor(v[0]):
                print(f"  first item shape: {tuple(v[0].shape)}")
        else:
            print(f"{k}: {type(v)} = {v}")


def flatten_cache(cache):
    """
    Concatenate all episodes into one flat timestep dataset.

    Returns:
        X: agent inputs, shape [total_steps, input_dim]
        B: beliefs, shape [total_steps, belief_dim]
        A: actions, shape [total_steps]
        O: observations, shape [total_steps, obs_dim]
        R: rewards, shape [total_steps]
        D: done flags, shape [total_steps]
    """
    episodes = cache["episodes"]

    X = torch.cat([ep["agent_inputs"] for ep in episodes], dim=0)
    B = torch.cat([ep["beliefs"][0] for ep in episodes], dim=0)
    A = torch.cat([ep["actions"] for ep in episodes], dim=0)
    O = torch.cat([ep["observations"] for ep in episodes], dim=0)
    R = torch.cat([ep["rewards"] for ep in episodes], dim=0)
    D = torch.cat([ep["dones"] for ep in episodes], dim=0)

    return X, B, A, O, R, D


def print_first_trajectory(cache, max_steps=None):
    """
    Print the first cached trajectory timestep by timestep.
    """
    ep = cache["episodes"][0]
    T = int(ep["length"])

    if max_steps is not None:
        T = min(T, max_steps)

    print(f"\n=== First trajectory, first {T} steps ===")

    for t in range(T):
        print(f"\nt = {t}")
        print("agent_input:", ep["agent_inputs"][t].tolist())
        print("belief:", ep["beliefs"][0][t].tolist())
        print("action:", int(ep["actions"][t]))
        print("observation:", ep["observations"][t].tolist())
        print("reward:", float(ep["rewards"][t]))
        print("done:", bool(ep["dones"][t]))


def print_hidden_preview(cache):
    """
    Print saved generator hidden preview if present.
    """
    ep = cache["episodes"][0]

    if "generator_hidden_preview" not in ep:
        print("No generator_hidden_preview found in first episode.")
        return

    H = ep["generator_hidden_preview"]
    print("\n=== generator_hidden_preview ===")
    print("shape:", tuple(H.shape))
    print(H)


def export_first_trajectory_csv(cache, out_path="first_trajectory.csv"):
    """
    Export first trajectory to CSV for easier inspection.
    """
    ep = cache["episodes"][0]
    T = int(ep["length"])

    rows = []
    for t in range(T):
        row = {
            "t": t,
            "action": int(ep["actions"][t]),
            "reward": float(ep["rewards"][t]),
            "done": bool(ep["dones"][t]),
        }

        for i, x in enumerate(ep["agent_inputs"][t].tolist()):
            row[f"agent_input_{i}"] = x

        for i, b in enumerate(ep["beliefs"][0][t].tolist()):
            row[f"belief_{i}"] = b

        for i, o in enumerate(ep["observations"][t].tolist()):
            row[f"observation_{i}"] = o

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved first trajectory to: {out_path}")


def export_hidden_preview_csv(cache, out_path="first_trajectory_hidden_preview.csv"):
    """
    Export saved generator hidden preview to CSV.
    """
    ep = cache["episodes"][0]

    if "generator_hidden_preview" not in ep:
        print("No generator_hidden_preview found in first episode.")
        return

    H = ep["generator_hidden_preview"]
    df = pd.DataFrame(
        H.numpy(),
        columns=[f"h_{i}" for i in range(H.shape[1])]
    )
    df.insert(0, "t", range(H.shape[0]))
    df.to_csv(out_path, index=False)
    print(f"Saved hidden preview to: {out_path}")


if __name__ == "__main__":
    cache = load_cache(PT_PATH)

    print_cache_summary(cache)

    X, B, A, O, R, D = flatten_cache(cache)
    print("\n=== Flattened dataset shapes ===")
    print("X / agent_inputs:", X.shape)
    print("B / beliefs:", B.shape)
    print("A / actions:", A.shape)
    print("O / observations:", O.shape)
    print("R / rewards:", R.shape)
    print("D / dones:", D.shape)

    print_first_trajectory(cache, max_steps=20)
    print_hidden_preview(cache)