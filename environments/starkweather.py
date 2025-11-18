import numpy as np
import torch
import scipy.stats
import random
from math import ceil

# from beliefs_starkweather.py

NULL = 0
STIM = 1
REW = 2


def transition_distribution(K, reward_times, reward_hazards,
                            p_omission, ITIhazard, iti_times=None):
    """
    T[i,j] = P(s'=j | s=i)
    """
    if iti_times is None:
        ITI_start = -1
        iti_times = []
    else:
        ITI_start = iti_times[0]
        assert iti_times[-1] == K - 1, "last iti time should be last state"

    T = np.zeros((K, K))

    # no probability of transitioning out of isi during this time
    for k in np.arange(min(reward_times)):
        T[k, k + 1] = 1.0

    # ISI states: may jump to ITI (reward delivery/omission) or progress in time
    for t, h in zip(reward_times, reward_hazards):
        T[t, t + 1] = 1 - h
        T[t, ITI_start] = h
    # After last reward time, must go to ITI
    T[reward_times.max(), ITI_start] = 1

    # ITI microstates (except last ITI): march forward deterministically
    for t in iti_times[:-1]:
        T[t, t + 1] = 1

    # transitions out of last ITI state:
    T[-1, -1] = 1 - ITIhazard
    T[-1, ITI_start] += ITIhazard * p_omission  # omission trial: go back into ITI
    T[-1, 0] = ITIhazard * (1 - p_omission)    # new trial: go to pre-stim state

    return T


def observation_distribution(K, reward_times, p_omission, ITIhazard, iti_times=None):
    """
    O[i,j,m] = P(x'=m | s=i, s'=j), for m in {NULL, STIM, REW}
    """
    O = np.zeros((K, K, 3))
    if iti_times is None:
        ITI_start = -1
    else:
        ITI_start = iti_times[0]
        assert iti_times[-1] == K - 1, "last iti time should be last state"

    # Progressed through time (non-ITI or non-reward transitions): observe NULL
    for k in np.arange(K - 1):
        O[k, k + 1, :] = [1, 0, 0]

    # Obtained reward: reward_times -> ITI_start with REW observation
    O[reward_times, ITI_start, :] = [0, 0, 1]

    # Stimulus onset: last ITI state -> state 0 with STIM observation
    O[-1, 0, :] = [0, 1, 0]

    # ITI self-loop and omission logic
    if np.arange(K)[ITI_start] == K - 1:
        # Only one ITI microstate (ITI_start == last state)
        O[-1, -1, NULL] = 1 - (ITIhazard * p_omission)  # stayed in ITI, saw NULL
        O[-1, -1, STIM] = ITIhazard * p_omission        # omission trial: see STIM
        O[-1, -1, REW] = 0                              # never see reward here
    else:
        # Multiple ITI microstates: last ITI always yields NULL
        O[-1, -1, :] = [1, 0, 0]
        if p_omission > 0:
            # On omission, jump from last ITI to ITI_start with STIM
            O[-1, ITI_start, :] = [0, 1, 0]

    return O


def pomdp(cue=0, p_omission=0.1, bin_size=0.2,
          ITIhazard=1 / 65.0, nITI_microstates=1):
    """
    Construct POMDP for the Starkweather task (cue 0).
    Returns:
        T: (K,K) transition matrix
        O: (K,K,3) observation kernel over {NULL, STIM, REW}
    """
    assert cue == 0

    # Reward times in seconds and discrete bins
    rts = np.arange(1.2, 3.0, 0.2)
    reward_times = (rts / bin_size).astype(int)

    # ISI distribution (Gaussian over times)
    ISIpdf = scipy.stats.norm.pdf(rts, rts.mean(), 0.5)
    ISIpdf = ISIpdf / ISIpdf.sum()

    # Number of hidden states: last reward bin + ITI microstates
    K = reward_times.max() + 1 + nITI_microstates

    # Hazard function for reward
    ISIcdf = np.cumsum(ISIpdf)
    ISIhazard = ISIpdf.copy()
    ISIhazard[1:] = ISIpdf[1:] / (1 - ISIcdf[:-1])
    reward_hazards = ISIhazard

    iti_times = np.arange(reward_times.max() + 1, K)

    T = transition_distribution(
        K, reward_times, reward_hazards,
        p_omission, ITIhazard,
        iti_times=iti_times
    )
    O = observation_distribution(
        K, reward_times,
        p_omission, ITIhazard,
        iti_times=iti_times
    )
    return T, O


def initial_belief(K, iti_min=0):
    """
    Start knowing we are in ITI at the beginning of a trial.
    """
    b = np.zeros(K)
    b[-(iti_min + 1)] = 1.0
    return b

# Environment

class StarkweatherEnv:
    """
    Passive Starkweather task as a POMDP environment with Bayes belief,
    in the style of TMaze (for DRQN).

    - One dummy action (action_size = 1) so actions do not affect transitions.
    - Observations are one-hot over {NULL, STIM, REW}.
    """

    gamma = 0.98
    observation_size = 3      # NULL / STIM / REW
    action_size = 1           # dummy
    belief_type = "exact"

    def __init__(
        self,
        bayes=True,
        p_omission=0.1,
        bin_size=0.2,
        iti_hazard=1 / 65.0,
        iti_min=0,
        nITI_microstates=10,
        max_steps=200,
    ):
        self.bayes = bayes
        self.p_omission = p_omission
        self.bin_size = bin_size
        self.iti_hazard = iti_hazard
        self.iti_min = iti_min
        self.nITI_microstates = nITI_microstates
        self.max_steps = max_steps

        # Build POMDP and convert to torch
        T_np, O_np = pomdp(
            cue=0,
            p_omission=self.p_omission,
            bin_size=self.bin_size,
            ITIhazard=self.iti_hazard,
            nITI_microstates=self.iti_min + 1,
        )
        self.T = torch.from_numpy(T_np).float()           # [K, K]
        self.O = torch.from_numpy(O_np).float()           # [K, K, 3]
        self.K = self.T.shape[0]


    def horizon(self):
        # Simple upper bound on episode length
        return self.max_steps

    def exploration(self):
        # Only one action; always return 0
        return 0

    def reset(self):
        """
        Reset to the initial hidden state distribution and return first observation.
        """
        # Sample initial hidden state from the prior
        b0_np = initial_belief(self.K, iti_min=self.iti_min)
        self.belief = torch.from_numpy(b0_np).float()     # [K]
        self.state = torch.distributions.Categorical(self.belief).sample().item()
        self.steps = 0

        obs = self._observation_from_state(self.state)

        if self.bayes:
            self._init_belief(obs)

        return obs

    def step(self, action: int):
        """
        Advance one time step. `action` is ignored for dynamics.
        """
        if action != 0:
            raise ValueError("Only action 0 is valid in this environment")

        self.steps += 1

        # Sample next state according to T
        T_s = self.T[self.state]                      # [K] = P(s' | s)
        next_state = torch.distributions.Categorical(T_s).sample().item()

        # Sample observation given (s, s')
        O_s = self.O[self.state, next_state]          # [3] over {NULL, STIM, REW}
        x = torch.distributions.Categorical(O_s).sample().item()

        self.state = next_state
        obs = torch.zeros(3)
        obs[x] = 1.0

        # reward: only when REW is observed
        reward = 1.0 if x == REW else 0.0

        # termination: either after max_steps or after a reward
        done = (self.steps >= self.max_steps) or (x == REW)

        if self.bayes:
            self._update_belief(action, obs)

        return obs, reward, done

    # --- Belief handling ---

    def _observation_from_state(self, state_idx: int):
        """
        Generate an observation given the current (initial) state.
        For the very first time step we approximate by sampling from
        staying in place (s -> s) with O[s,s,:].
        """
        O_ss = self.O[state_idx, state_idx]   # [3]
        x = torch.distributions.Categorical(O_ss).sample().item()
        obs = torch.zeros(3)
        obs[x] = 1.0
        return obs

    def _init_belief(self, observation: torch.FloatTensor):
        """
        Initialise belief b_0 based on the initial observation.
        """
        # start from prior
        b = torch.from_numpy(initial_belief(self.K, iti_min=self.iti_min)).float()

        # apply one-step update with 'pseudo' T*O for the first obs
        x = observation.argmax().item()
        T_eff = self.T * self.O[:, :, x]      # [K,K]
        b = b @ T_eff
        b = b / b.sum()
        self.belief = b

    def _update_belief(self, action: int, observation: torch.FloatTensor):
        """
        Standard Bayesian filtering step b' ∝ b^T (T * O_x).
        """
        x = observation.argmax().item()
        T_eff = self.T * self.O[:, :, x]      # [K,K]
        b = self.belief @ T_eff               # [K]
        b = b / b.sum()
        self.belief = b

    def get_belief(self):
        """
        Return the current belief in a tuple, like TMaze and Irrelevant.
        """
        return (self.belief.clone(),)
