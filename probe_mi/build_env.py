import torch

from environments.tmaze import TMaze
from environments.hike import MountainHike
from environments.irrelevant import Irrelevant
from environments.starkweather import StarkweatherEnv
from environments.tiger import Tiger
from environments.gridworld import GridWorld
from environments.crybaby import CryingBaby

from agents.drqn import DRQN


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # keep your previous behavior for MPS
    return torch.device("cpu")


def _get_attr(train_args, name, default=None):
    return getattr(train_args, name, default)


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
    elif train_args.environment == "tiger":
        env = Tiger(
            bayes=True,
            listen_accuracy=train_args.listen_accuracy,
            reward_listen=train_args.reward_listen,
            reward_correct=train_args.reward_correct,
            reward_wrong=train_args.reward_wrong,
            horizon=train_args.horizon,
        )
    elif train_args.environment == "gridworld":
        env = GridWorld(
            bayes=True,
            size=train_args.size,
            tprob=train_args.tprob,
            reward_scheme=train_args.reward_scheme,
            reward_margin=train_args.reward_margin,
            step_cost=train_args.step_cost,
        )
    elif train_args.environment == "crybaby":
        env = CryingBaby(
            bayes=True,
            p_hungry_if_full_wait=train_args.p_hungry_if_full_wait,
            p_stay_hungry_wait=train_args.p_stay_hungry_wait,
            p_full_if_feed=train_args.p_full_if_feed,
            p_cry_if_hungry=train_args.p_cry_if_hungry,
            p_cry_if_full=train_args.p_cry_if_full,
            p0_hungry=_get_attr(train_args, "p0_hungry", 0.5),
            reward_cry=train_args.reward_cry,
            cost_feed=train_args.cost_feed,
            reward_quiet=_get_attr(train_args, "reward_quiet", 0.0),
            horizon=_get_attr(train_args, "horizon", 50),
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
