import numpy as np
import torch
import scipy.stats
from math import ceil

NULL = 0
STIM = 1
REW  = 2


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

    T = np.zeros((K, K), dtype=np.float32)

    for k in np.arange(int(min(reward_times))):
        T[k, k + 1] = 1.0

    for t, h in zip(reward_times, reward_hazards):
        t = int(t)
        T[t, t + 1] = 1.0 - float(h)
        T[t, ITI_start] = float(h)

    # After last reward time, must go to ITI
    T[int(reward_times.max()), ITI_start] = 1.0

    # ITI microstates
    for t in iti_times[:-1]:
        t = int(t)
        T[t, t + 1] = 1.0

    # transitions out of last ITI state:
    # - stay in ITI with prob 1-ITIhazard
    # - with prob ITIhazard start a new trial (to state 0) unless omission
    # - on omission, go back into ITI_start
    T[-1, -1] = 1.0 - float(ITIhazard)
    T[-1, ITI_start] += float(ITIhazard) * float(p_omission)       # omission trial: go back into ITI
    T[-1, 0] = float(ITIhazard) * (1.0 - float(p_omission))        # new trial: go to pre-stim state

    row_sums = T.sum(axis=1, keepdims=True)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        T = T / np.clip(row_sums, 1e-12, None)

    return T


def observation_distribution(K, reward_times, p_omission, ITIhazard, iti_times=None):
    """
    O[i,j,m] = P(x'=m | s=i, s'=j), for m in {NULL, STIM, REW}
    """
    O = np.zeros((K, K, 3), dtype=np.float32)

    if iti_times is None:
        ITI_start = -1
    else:
        ITI_start = iti_times[0]
        assert iti_times[-1] == K - 1, "last iti time should be last state"

    # Progressed through time (non-ITI or non-reward transitions): observe NULL
    # This covers all deterministic "k -> k+1" edges (including ITI forward microsteps).
    for k in np.arange(K - 1):
        O[k, k + 1, :] = [1.0, 0.0, 0.0]

    # Obtained reward: reward_times -> ITI_start with REW observation
    # (These are the "hazard-triggered" jumps to ITI.)
    reward_times = np.asarray(reward_times, dtype=int)
    O[reward_times, ITI_start, :] = [0.0, 0.0, 1.0]

    # Stimulus onset: last ITI state -> state 0 with STIM observation
    O[-1, 0, :] = [0.0, 1.0, 0.0]

    # ITI self-loop and omission logic
    if np.arange(K)[ITI_start] == K - 1:
        # Only one ITI microstate (ITI_start == last state)
        # On "omission", we see STIM but remain in ITI (because ITI_start == last)
        O[-1, -1, NULL] = 1.0 - (float(ITIhazard) * float(p_omission))
        O[-1, -1, STIM] = float(ITIhazard) * float(p_omission)
        O[-1, -1, REW] = 0.0
    else:
        O[-1, -1, :] = [1.0, 0.0, 0.0]
        if p_omission > 0:
            O[-1, ITI_start, :] = [0.0, 1.0, 0.0]

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

    rts = np.arange(1.2, 3.0, 0.2)
    reward_times = (rts / bin_size).astype(int)

    ISIpdf = scipy.stats.norm.pdf(rts, rts.mean(), 0.5)
    ISIpdf = ISIpdf / ISIpdf.sum()

    # Hazard function for reward
    ISIcdf = np.cumsum(ISIpdf)
    ISIhazard = ISIpdf.copy()
    ISIhazard[1:] = ISIpdf[1:] / (1.0 - ISIcdf[:-1])
    reward_hazards = ISIhazard

    # Number of hidden states: last reward bin + ITI microstates
    K = int(reward_times.max()) + 1 + int(nITI_microstates)

    iti_times = np.arange(int(reward_times.max()) + 1, K, dtype=int)

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

    Convention (matches your existing intent):
      - if iti_min == 0: start at last ITI microstate (K-1)
      - if iti_min > 0: start at (K-1-iti_min)
        (i.e., "at least iti_min steps away from stimulus")
    """
    b = np.zeros(K, dtype=np.float32)
    idx = int(K - 1 - iti_min)
    idx = max(0, min(K - 1, idx))
    b[idx] = 1.0
    return b


class StarkweatherEnv:
    """
    Passive Starkweather task as a POMDP environment with Bayes belief,
    in the style of TMaze (for DRQN).

    - One dummy action (action_size = 1) so actions do not affect transitions.
    - Observations are one-hot over {NULL, STIM, REW}.
    """

    gamma = 0.98
    observation_size = 3      # NULL / STIM / REW
    action_size = 1
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


        T_np, O_np = pomdp(
            cue=0,
            p_omission=self.p_omission,
            bin_size=self.bin_size,
            ITIhazard=self.iti_hazard,
            nITI_microstates=self.nITI_microstates,
        )
        self.T = torch.from_numpy(T_np).float()           # [K, K]
        self.O = torch.from_numpy(O_np).float()           # [K, K, 3]
        self.K = int(self.T.shape[0])

        self.state = None
        self.belief = None
        self.steps = 0

    def horizon(self):
        return self.max_steps

    def exploration(self):
        return 0

    def reset(self):
        """
        Reset to the initial hidden state distribution and return first observation.

        IMPORTANT FIX: do NOT sample from O[s,s,:] (often all zeros).
        Instead sample an initial transition (s -> s') and then observation from O[s,s'].
        """
        b0_np = initial_belief(self.K, iti_min=self.iti_min)
        self.belief = torch.from_numpy(b0_np).float()     # [K]
        self.state = torch.distributions.Categorical(self.belief).sample().item()
        self.steps = 0

        # Sample initial transition and observation consistently
        T_s = self.T[self.state]                          # [K]
        next_state = torch.distributions.Categorical(T_s).sample().item()
        O_s = self.O[self.state, next_state]              # [3]
        x = torch.distributions.Categorical(O_s).sample().item()

        self.state = next_state
        obs = torch.zeros(3)
        obs[x] = 1.0

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
        T_s = self.T[self.state]                          # [K] = P(s' | s)
        next_state = torch.distributions.Categorical(T_s).sample().item()

        # Sample observation given (s, s')
        O_s = self.O[self.state, next_state]
        x = torch.distributions.Categorical(O_s).sample().item()

        self.state = next_state
        obs = torch.zeros(3)
        obs[x] = 1.0

        reward = 1.0 if x == REW else 0.0

        done = (self.steps >= self.max_steps) or (x == REW)

        if self.bayes:
            self._update_belief(action, obs)

        return obs, reward, done

    # --- Belief handling ---

    def _init_belief(self, observation: torch.FloatTensor):
        """
        Initialise belief b_0 based on the initial observation.

        We start from the prior initial_belief, then apply one standard filter update:
            b' ∝ b^T (T * O_x)
        """
        b = torch.from_numpy(initial_belief(self.K, iti_min=self.iti_min)).float()
        x = int(observation.argmax().item())
        T_eff = self.T * self.O[:, :, x]                  # [K, K]
        b = b @ T_eff                                     # [K]
        b = b / b.sum().clamp_min(1e-12)
        self.belief = b

    def _update_belief(self, action: int, observation: torch.FloatTensor):
        """
        Standard Bayesian filtering step b' ∝ b^T (T * O_x).
        """
        x = int(observation.argmax().item())
        T_eff = self.T * self.O[:, :, x]                  # [K, K]
        b = self.belief @ T_eff                           # [K]
        b = b / b.sum().clamp_min(1e-12)
        self.belief = b

    def get_belief(self):
        """
        Return the current belief in a tuple, like TMaze and Irrelevant.
        """
        return (self.belief.clone(),)
