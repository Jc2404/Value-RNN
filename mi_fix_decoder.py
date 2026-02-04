import wandb
import torch
import pandas as pd
import os

from argparse import ArgumentParser
from copy import deepcopy

from environments.tmaze import TMaze
from environments.hike import MountainHike
from environments.irrelevant import Irrelevant
from environments.starkweather import StarkweatherEnv

from agents.drqn import DRQN
from mine.mine import MutualInformationNeuralEstimator
from utils import generate_hiddens_and_beliefs, get_run_statistic

def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def parse_variant(vname: str):
    """
    Examples:
      "tmaze_length=45" -> ("tmaze_length", 45.0)
      "starkweather_p_omission=0.1" -> ("starkweather_p_omission", 0.1)
    """
    k, v = vname.split("=")
    return k, float(v)

def build_environment(train_args, overrides=None):
    """
    Build an environment from train_args, optionally overriding a subset of params
    via overrides dict.
    """
    overrides = overrides or {}
    env_name = train_args.environment

    if env_name == "tmaze":
        length = overrides.get("length", train_args.length)
        stochasticity = overrides.get("stochasticity", train_args.stochasticity)
        env = TMaze(bayes=True, length=length, stochasticity=stochasticity)

    elif env_name == "hike":
        variations = overrides.get("variations", train_args.variations)
        env = MountainHike(bayes=True, variations=variations)

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
    """
    Returns: list of (variant_name, overrides_dict)

    Only one test flag is honored at a time; others are ignored.
    """
    env_name = train_args.environment
    variants = []

    if env_name == "tmaze":
        if args.test_length:
            grid = [45, 49, 50, 51, 55]
            for L in grid:
                variants.append((f"tmaze_length={L}", {"length": L}))
            return variants

        if args.test_stochasticity:
            grid = [0, 0.05, 0.1, 0.15]
            for s in grid:
                variants.append((f"tmaze_stochasticity={s}", {"stochasticity": s}))
            return variants

    if env_name == "hike":
        if args.test_variations:
            grid = [1, 2, 4, 8]
            for v in grid:
                variants.append((f"hike_variations={v}", {"variations": v}))
            return variants

    if env_name == "starkweather":
        if args.test_p_omission:
            grid = [0.2, 0.2, 0.2, 0.1, 0.1]
            #grid = [0.0, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2]
            for p in grid:
                variants.append((f"starkweather_p_omission={p}", {"p_omission": p}))
            return variants

        if args.test_bin_size:
            grid = [train_args.bin_size, max(1, train_args.bin_size // 2), train_args.bin_size * 2]
            seen = set()
            grid2 = []
            for x in grid:
                if x not in seen:
                    grid2.append(x)
                    seen.add(x)
            for b in grid2:
                variants.append((f"starkweather_bin_size={b}", {"bin_size": b}))
            return variants

        if args.test_iti_hazard:
            grid = [0.01, 0.05, 0.1, 0.2]
            for h in grid:
                variants.append((f"starkweather_iti_hazard={h}", {"iti_hazard": h}))
            return variants

        if args.test_iti_min:
            grid = [0, 5, 10, 20]
            for m in grid:
                variants.append((f"starkweather_iti_min={m}", {"iti_min": m}))
            return variants

        if args.test_nITI_microstates:
            grid = [1, 2, 4, 8]
            for n in grid:
                variants.append((f"starkweather_nITI_microstates={n}", {"nITI_microstates": n}))
            return variants
    return variants

def build_mine(hiddens, beliefs, args, device):
    belief_sizes = []
    representation_sizes = []

    for belief_part in beliefs:
        belief_sizes.append(belief_part.size(-1))
        if belief_part.ndim == 2:
            representation_sizes.append(None)
        elif belief_part.ndim == 3:
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

def main(args):
    train_args = get_run_statistic(args.train_id)
    device = select_device()
    print("Device:", device)

    variants = pick_variants(train_args, args)
    if not variants:
        print("No test flag selected (or env unsupported). Will only evaluate base env.")

    base_env = build_environment(train_args, overrides=None)

    cfg = vars(train_args) | vars(args)
    wandb.init(project="belief-mi-protocolA-frozen", name=args.name, config=cfg, save_code=True)
    wandb.save("*.py")
    wandb.save("agents/*.py")
    wandb.save("environments/*.py")
    os.makedirs("report", exist_ok=True)
    excel_path = f"report/mi_protocolA_{args.train_id}.xlsx"
    episode_rows = {}

    for episode in range(0, train_args.episodes + 1, args.mine_period):
        agent = DRQN(
            cell=train_args.cell,
            action_size=base_env.action_size,
            observation_size=base_env.observation_size,
            num_layers=train_args.num_layers,
            hidden_size=train_args.hidden_size,
        )
        agent.load(args.train_id, episode=episode)

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
            valid_size=args.valid_size,
        )

        # Evaluate on base env (held-out resample unless --train_set)
        if not args.train_set:
            h_eval, b_eval = generate_hiddens_and_beliefs(
                agent, base_env,
                num_samples=args.mine_num_samples,
                epsilon=args.epsilon,
                approximate=args.approximate,
            )
            h_eval = h_eval.to(device)
            b_eval = tuple(bb.to(device) for bb in b_eval)
        else:
            h_eval, b_eval = h_base, b_base

        mi_base = mine.estimate(h_eval, b_eval)
        
        episode_rows[episode] = []
        episode_rows[episode].append({
            "task_name": "base",
            "task_value": None,
            "mi": mi_base,
            "metric": "MI",
        })

        key_base = "mi/base"
        if args.belief_part is not None:
            key_base = f"mi/base-part{args.belief_part}"
        if args.epsilon != 0.0:
            key_base += f"-eps{args.epsilon}"

        wandb.log({"train/episode": episode, key_base: mi_base})
        print(f"[episode {episode}] base MI = {mi_base}")

        # -------------------------
        # Freeze MINE; evaluate on variants
        # -------------------------
        for vname, overrides in variants:
            venv = build_environment(train_args, overrides=overrides)

            h_v, b_v = generate_hiddens_and_beliefs(
                agent, venv,
                num_samples=args.mine_num_samples,
                epsilon=args.epsilon,
                approximate=args.approximate,
            )
            h_v = h_v.to(device)
            b_v = tuple(bb.to(device) for bb in b_v)

            mi_v = mine.estimate(h_v, b_v)

            task_name, task_value = parse_variant(vname)
            episode_rows[episode].append({
                "task_name": task_name,
                "task_value": task_value,
                "mi": mi_v,
                "metric": "MI",
            })

            key_v = f"mi/frozen_on_base__eval_on/{vname}"
            if args.belief_part is not None:
                key_v += f"-part{args.belief_part}"
            if args.epsilon != 0.0:
                key_v += f"-eps{args.epsilon}"

            wandb.log({"train/episode": episode, key_v: mi_v})
            print(f"[episode {episode}] {vname} MI (frozen) = {mi_v}")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for ep, rows in episode_rows.items():
            df = pd.DataFrame(rows)
            df_sorted = df.sort_values(by=["task_value"], na_position="first")
            wide = df_sorted.pivot_table(
                index=["task_value"],
                columns=["metric"],
                values="mi",
                aggfunc="mean"
            ).reset_index()
            sheet_name = f"ep_{ep}"
            wide.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    print(f"Saved Excel: {excel_path}")
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser(description="Protocol A: train MINE on base env, evaluate on modified envs with frozen estimator.")
    parser.add_argument("name", type=str, nargs="?", default=None)
    parser.add_argument("train_id", type=str)

    # sampling / schedule
    parser.add_argument("--mine_num_samples", type=int, default=10000)
    parser.add_argument("--mine_period", type=int, default=100)
    parser.add_argument("--approximate", action="store_true")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--train_set", action="store_true")
    parser.add_argument("--valid_size", type=float, default=0.2)

    # MINE params
    parser.add_argument("--mine_num_layers", type=int, default=2)
    parser.add_argument("--mine_hidden_size", type=int, default=256)
    parser.add_argument("--mine_alpha", type=float, default=0.01)
    parser.add_argument("--mine_num_epochs", type=int, default=200)
    parser.add_argument("--mine_batch_size", type=int, default=1024)
    parser.add_argument("--mine_learning_rate", type=float, default=1e-3)
    parser.add_argument("--mine_lambda", type=float, default=0.0)
    parser.add_argument("--representation_size", type=int, default=16)
    parser.add_argument("--belief_part", type=int, default=None)

    # ---- Test flags ----
    # TMaze
    parser.add_argument("--test_length", action="store_true")
    parser.add_argument("--test_stochasticity", action="store_true")
    # Hike
    parser.add_argument("--test_variations", action="store_true")
    # Starkweather
    parser.add_argument("--test_p_omission", action="store_true")
    parser.add_argument("--test_bin_size", action="store_true")
    parser.add_argument("--test_iti_hazard", action="store_true")
    parser.add_argument("--test_iti_min", action="store_true")
    parser.add_argument("--test_nITI_microstates", action="store_true")

    args = parser.parse_args()
    print("\n".join(f"{k}={v}" for k, v in vars(args).items()))
    main(args)
