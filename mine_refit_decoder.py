import wandb
import torch

from argparse import ArgumentParser

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


def build_environment(train_args, overrides=None):
    overrides = overrides or {}
    env_name = train_args.environment

    if env_name == "tmaze":
        env = TMaze(
            bayes=True,
            length=overrides.get("length", train_args.length),
            stochasticity=overrides.get("stochasticity", train_args.stochasticity),
        )
    elif env_name == "hike":
        env = MountainHike(
            bayes=True,
            variations=overrides.get("variations", train_args.variations),
        )
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
    env_name = train_args.environment
    variants = []

    if env_name == "tmaze":
        if args.test_length:
            for L in [45, 49, 50, 51, 55]:
                variants.append((f"tmaze_length={L}", {"length": L}))
            return variants
        if args.test_stochasticity:
            for s in [0, 0.05, 0.1, 0.15]:
                variants.append((f"tmaze_stochasticity={s}", {"stochasticity": s}))
            return variants

    if env_name == "hike":
        if args.test_variations:
            for v in [1, 2, 4, 8]:
                variants.append((f"hike_variations={v}", {"variations": v}))
            return variants

    if env_name == "starkweather":
        if args.test_p_omission:
            for p in [0.0, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2]:
                variants.append((f"starkweather_p_omission={p}", {"p_omission": p}))
            return variants
        if args.test_bin_size:
            grid = [train_args.bin_size, max(1, train_args.bin_size // 2), train_args.bin_size * 2]
            seen, grid2 = set(), []
            for x in grid:
                if x not in seen:
                    grid2.append(x); seen.add(x)
            for b in grid2:
                variants.append((f"starkweather_bin_size={b}", {"bin_size": b}))
            return variants
        if args.test_iti_hazard:
            for h in [0.01, 0.05, 0.1, 0.2]:
                variants.append((f"starkweather_iti_hazard={h}", {"iti_hazard": h}))
            return variants
        if args.test_iti_min:
            for m in [0, 5, 10, 20]:
                variants.append((f"starkweather_iti_min={m}", {"iti_min": m}))
            return variants
        if args.test_nITI_microstates:
            for n in [1, 2, 4, 8]:
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

    return MutualInformationNeuralEstimator(
        hs_sizes=hiddens.size(-1),
        belief_sizes=belief_sizes,
        hidden_size=args.mine_hidden_size,
        num_layers=args.mine_num_layers,
        alpha=args.mine_alpha,
        representation_sizes=representation_sizes,
        belief_part=args.belief_part,
        device=device,
    )


def main(args):
    train_args = get_run_statistic(args.train_id)
    device = select_device()
    print("Device:", device)

    variants = pick_variants(train_args, args)
    if not variants:
        print("No test flag selected (or env unsupported). Will only run base env as a single variant.")
        variants = [("base", {})]

    cfg = vars(train_args) | vars(args)
    wandb.init(project="belief-mi-protocolB-refit", name=args.name, config=cfg, save_code=True)
    wandb.save("*.py")
    wandb.save("agents/*.py")
    wandb.save("environments/*.py")

    for episode in range(0, train_args.episodes + 1, args.mine_period):
        # Build *any* env just to get action/obs sizes right (they should match across overrides)
        env0 = build_environment(train_args, overrides=variants[0][1])

        agent = DRQN(
            cell=train_args.cell,
            action_size=env0.action_size,
            observation_size=env0.observation_size,
            num_layers=train_args.num_layers,
            hidden_size=train_args.hidden_size,
        )
        agent.load(args.train_id, episode=episode)

        for vname, overrides in variants:
            venv = build_environment(train_args, overrides=overrides)

            # Train MINE on this variant
            h_tr, b_tr = generate_hiddens_and_beliefs(
                agent, venv,
                num_samples=args.mine_num_samples,
                epsilon=args.epsilon,
                approximate=args.approximate,
            )
            h_tr = h_tr.to(device)
            b_tr = tuple(bb.to(device) for bb in b_tr)

            mine = build_mine(h_tr, b_tr, args, device)

            mine.optimize(
                h_tr, b_tr,
                num_epochs=args.mine_num_epochs,
                logger=wandb.log,
                learning_rate=args.mine_learning_rate,
                batch_size=args.mine_batch_size,
                lambd=args.mine_lambda,
                valid_size=args.valid_size,
            )

            # Evaluate on same variant (resample unless --train_set)
            if not args.train_set:
                h_ev, b_ev = generate_hiddens_and_beliefs(
                    agent, venv,
                    num_samples=args.mine_num_samples,
                    epsilon=args.epsilon,
                    approximate=args.approximate,
                )
                h_ev = h_ev.to(device)
                b_ev = tuple(bb.to(device) for bb in b_ev)
            else:
                h_ev, b_ev = h_tr, b_tr

            mi = mine.estimate(h_ev, b_ev)

            key = f"mi/refit_on/{vname}"
            if args.belief_part is not None:
                key += f"-part{args.belief_part}"
            if args.epsilon != 0.0:
                key += f"-eps{args.epsilon}"

            wandb.log({"train/episode": episode, key: mi})
            print(f"[episode {episode}] {vname} MI (refit) = {mi}")

    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser(description="refit MINE separately on each modified env variant.")
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
