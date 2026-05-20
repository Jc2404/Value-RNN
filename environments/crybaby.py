# environments/baby.py
import random
import torch

OBSERVATIONS = torch.eye(2)
O_CRY, O_QUIET = OBSERVATIONS

ACTIONS = (0, 1)
A_WAIT, A_FEED = ACTIONS


class CryingBaby:
    """
    Crying Baby POMDP (episodic, discrete, exact belief).

    Hidden state:
      - hungry ∈ {HUNGRY, FULL}

    Actions:
      - WAIT: do nothing
      - FEED: feed the baby (usually makes it FULL)

    Observations (emitted after transition to s'):
      - CRY or QUIET, with different probabilities depending on state

    Rewards (configurable):
      - default: penalize crying + penalize feeding effort

    Interface matches TMaze/Tiger in this repo style:
      - reset() -> one-hot observation
      - step(a) -> (one-hot observation, reward: float, done: bool)
      - exploration(), horizon()
      - if bayes=True, maintains exact belief and get_belief() -> (belief,)
    """

    gamma = 0.98
    observation_size = 2
    action_size = 2
    belief_type = "exact"

    # State indices
    HUNGRY = 0
    FULL = 1

    def __init__(
        self,
        # --- dynamics: P(s' | s, a)
        p_hungry_if_full_wait=0.10,   # P(HUNGRY at t+1 | FULL at t, WAIT)
        p_stay_hungry_wait=0.90,      # P(HUNGRY at t+1 | HUNGRY at t, WAIT)
        p_full_if_feed=0.95,          # P(FULL at t+1 | any state at t, FEED)

        # --- observation model: P(o | s')
        p_cry_if_hungry=0.90,         # P(CRY | HUNGRY)
        p_cry_if_full=0.10,           # P(CRY | FULL)

        # --- initial state prior
        p0_hungry=0.50,               # P(HUNGRY at reset)

        # --- reward shaping (simple + interpretable)
        reward_cry=-1.0,              # penalty if observation is CRY
        cost_feed=-0.2,               # action cost for FEED
        reward_quiet=0.0,             # reward if QUIET (default 0)

        # --- episode
        horizon=15,

        # --- belief tracking
        bayes=False,
    ):
        # params
        self.p_hungry_if_full_wait = float(p_hungry_if_full_wait)
        self.p_stay_hungry_wait = float(p_stay_hungry_wait)
        self.p_full_if_feed = float(p_full_if_feed)

        self.p_cry_if_hungry = float(p_cry_if_hungry)
        self.p_cry_if_full = float(p_cry_if_full)

        self.p0_hungry = float(p0_hungry)

        self.reward_cry = float(reward_cry)
        self.cost_feed = float(cost_feed)
        self.reward_quiet = float(reward_quiet)

        self._horizon = int(horizon)
        self.bayes = bool(bayes)

        # validation (light)
        for name, p in [
            ("p_hungry_if_full_wait", self.p_hungry_if_full_wait),
            ("p_stay_hungry_wait", self.p_stay_hungry_wait),
            ("p_full_if_feed", self.p_full_if_feed),
            ("p_cry_if_hungry", self.p_cry_if_hungry),
            ("p_cry_if_full", self.p_cry_if_full),
            ("p0_hungry", self.p0_hungry),
        ]:
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {p}")

        # derived transition tables (2x2 per action)
        # T[a][s, s'] = P(s' | s, a)
        self.T = {
            A_WAIT: torch.zeros(2, 2, dtype=torch.float32),
            A_FEED: torch.zeros(2, 2, dtype=torch.float32),
        }
        # WAIT
        # from HUNGRY: stay hungry with p_stay_hungry_wait else become FULL
        self.T[A_WAIT][self.HUNGRY, self.HUNGRY] = self.p_stay_hungry_wait
        self.T[A_WAIT][self.HUNGRY, self.FULL] = 1.0 - self.p_stay_hungry_wait
        # from FULL: become hungry with p_hungry_if_full_wait else stay FULL
        self.T[A_WAIT][self.FULL, self.HUNGRY] = self.p_hungry_if_full_wait
        self.T[A_WAIT][self.FULL, self.FULL] = 1.0 - self.p_hungry_if_full_wait

        # FEED: regardless of current state, become FULL with p_full_if_feed
        # and HUNGRY with 1-p_full_if_feed (e.g., feeding fails / not enough)
        for s in (self.HUNGRY, self.FULL):
            self.T[A_FEED][s, self.FULL] = self.p_full_if_feed
            self.T[A_FEED][s, self.HUNGRY] = 1.0 - self.p_full_if_feed

        # observation model O[s', o] = P(o | s')
        # order o: CRY(0), QUIET(1)
        self.O = torch.zeros(2, 2, dtype=torch.float32)
        self.O[self.HUNGRY, 0] = self.p_cry_if_hungry
        self.O[self.HUNGRY, 1] = 1.0 - self.p_cry_if_hungry
        self.O[self.FULL, 0] = self.p_cry_if_full
        self.O[self.FULL, 1] = 1.0 - self.p_cry_if_full

        # internal state
        self.state = None
        self.steps = 0
        self.terminal = False
        self.K = 2  # number of discrete hidden states

        # belief
        self.belief = None

    # -------- Core API (as in TMaze / Tiger) --------

    def horizon(self):
        return self._horizon

    def exploration(self):
        """
        Simple exploration policy.
        Bias a bit toward WAIT so the agent must handle persistence dynamics.
        """
        return random.choices(ACTIONS, weights=(0.6, 0.4))[0]

    def reset(self):
        """
        Reset env, sample initial hidden state and emit initial observation.
        Returns: one-hot observation
        """
        self.steps = 0
        self.terminal = False

        self.state = self._sample_initial_state()
        obs = self._sample_observation(self.state)

        if self.bayes:
            self._init_belief(obs)

        return obs

    def step(self, action: int):
        """
        Apply action, transition, emit observation, compute reward, done.
        Returns: (one-hot obs, reward: float, done: bool)
        """
        self._check_action(action)

        if self._terminal(last=False):
            # TMaze/Tiger-style: if already terminal, no-op
            return self._observation_from_last(), 0.0, True

        self.steps += 1

        # transition
        next_state = self._sample_next_state(self.state, action)
        self.state = next_state

        # observation (depends on new state)
        obs = self._sample_observation(self.state)

        # reward
        reward = self._reward(action, obs)

        # done (episodic truncation)
        done = self._terminal(last=False)

        # belief update
        if self.bayes:
            self._update_belief(action, obs)

        return obs, reward, done

    # -------- Helpers --------

    def _check_action(self, action):
        if action < 0 or self.action_size <= action:
            raise ValueError(f"The action should be in range [0, {self.action_size}[")

    def _terminal(self, last=False):
        # purely horizon-based termination
        return self.steps >= self._horizon

    def _sample_initial_state(self):
        return self.HUNGRY if (random.random() < self.p0_hungry) else self.FULL

    def _sample_next_state(self, s, a):
        p = self.T[a][s]  # [2]
        # categorical sample
        u = random.random()
        return 0 if u < float(p[0].item()) else 1

    def _sample_observation(self, s):
        # o ~ Cat(O[s, :])
        p_cry = float(self.O[s, 0].item())
        o = 0 if (random.random() < p_cry) else 1
        return O_CRY if o == 0 else O_QUIET

    def _reward(self, action, obs):
        r = 0.0
        # cry penalty / quiet reward
        if obs.argmax().item() == 0:
            r += self.reward_cry
        else:
            r += self.reward_quiet

        # feeding cost
        if action == A_FEED:
            r += self.cost_feed

        return float(r)

    def _observation_from_last(self):
        # if terminal, return QUIET by convention (doesn't matter for terminal)
        return O_QUIET

    # -------- Exact Bayes belief --------

    def _init_belief(self, observation):
        """
        Prior belief b0 over [P(HUNGRY), P(FULL)] then incorporate initial obs
        as if we "observed" the current state.
        """
        b = torch.tensor([self.p0_hungry, 1.0 - self.p0_hungry], dtype=torch.float32)
        # incorporate observation likelihood at current state
        o = int(observation.argmax().item())
        b = b * self.O[:, o]
        z = b.sum().clamp_min(1e-12)
        self.belief = (b / z)

    def _update_belief(self, action, observation):
        """
        Bayesian filter:
          b'(s') ∝ O(o | s') * sum_s T(s'|s,a) b(s)

        Here T[a] is [s, s'].
        """
        o = int(observation.argmax().item())

        # predict: b_pred(s') = sum_s b(s) T[a][s, s']
        b_pred = self.belief @ self.T[action]  # [2]

        # update with likelihood
        b_post = b_pred * self.O[:, o]
        z = b_post.sum().clamp_min(1e-12)
        self.belief = b_post / z

    def get_belief(self):
        """
        Returns belief in the same style as TMaze/Tiger: (belief_tensor,)
        """
        return (self.belief.clone(),)