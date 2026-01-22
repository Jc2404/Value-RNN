import torch

from environments.tmaze import TMaze
from environments.hike import MountainHike
from environments.irrelevant import Irrelevant
from environments.starkweather import StarkweatherEnv

from agents.drqn import DRQN


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # keep your previous behavior for MPS
    return torch.device("cpu")


def build_environment(train_args):
    if train_args.environment == "tmaze":
        env = TMaze(
            bayes=True,
            length=train_args.length,
            stochasticity=train_args.stochasticity
        )
    elif train_args.environment == "hike":
        env = MountainHike(
            bayes=True,
            variations=train_args.variations
        )
    elif train_args.environment == "starkweather":
        env = StarkweatherEnv(
            p_omission=train_args.p_omission,
            bin_size=train_args.bin_size,
            iti_hazard=train_args.iti_hazard,
            iti_min=train_args.iti_min,
            nITI_microstates=train_args.nITI_microstates,
        )
    else:
        raise NotImplementedError(f"Unknown environment {train_args.environment}")

    if getattr(train_args, "irrelevant", 0) != 0:
        env = Irrelevant(env, state_size=train_args.irrelevant, bayes=True)

    return env


def build_agent(train_args, environment):
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
