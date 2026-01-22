import torch
import random


# Observation space (one-hot)
# 0: hear_left, 1: hear_right, 2: null (after opening door / terminal)
OBSERVATIONS = torch.eye(3)
O_HEAR_LEFT, O_HEAR_RIGHT, O_NULL = OBSERVATIONS

# Action space
# 0: listen, 1: open_left, 2: open_right
ACTIONS = (0, 1, 2)
A_LISTEN, A_OPEN_LEFT, A_OPEN_RIGHT = ACTIONS


class Tiger:
    """
    Classic Tiger POMDP (episodic).

    Hidden state: tiger ∈ {LEFT, RIGHT}
    Actions:
      - LISTEN: noisy observation, reward = -1
      - OPEN_LEFT / OPEN_RIGHT: terminal, reward = +10 if correct else -100
    Observations:
      - hear_left / hear_right after LISTEN (noisy)
      - null after OPEN (terminal)

    This matches the TMaze interface used by DRQN:
      - reset() -> one-hot observation
      - step(a) -> (one-hot observation, reward, done)
      - exploration(), horizon()
      - if bayes=True, maintains exact belief + get_belief() -> (belief,)
    """

    gamma = 0.98
    observation_size = 3
    action_size = 3
    belief_type = "exact"

    def __init__(
        self,
        listen_accuracy=0.85,
        reward_listen=-1.0,
        reward_correct=10.0,
        reward_wrong=-100.0,
        horizon=20,
        bayes=False,
    ):
        self.listen_accuracy = float(listen_accuracy)
        self.reward_listen = float(reward_listen)
        self.reward_correct = float(reward_correct)
        self.reward_wrong = float(reward_wrong)
        self._horizon = int(horizon)
        self.bayes = bayes

        if not (0.5 <= self.listen_accuracy <= 1.0):
            raise ValueError("listen_accuracy should be in [0.5, 1.0]")

        # internal state
        self.tiger_left = True
        self.terminal = False

    # -------- Core API (as in TMaze) --------

    def horizon(self):
        """Recommended truncation horizon."""
        return self._horizon

    def exploration(self):
        """
        Exploration policy.
        Bias toward LISTEN so the agent can infer the hidden state.
        """
        return random.choices(ACTIONS, weights=(0.6, 0.2, 0.2))[0]

    def reset(self):
        """
        Resets env; returns initial observation (uninformative).
        """
        self._init_state()
        observation = self._observation()

        if self.bayes:
            self._init_belief(observation)

        return observation

    def step(self, action):
        """
        Samples transition in POMDP according to chosen action.

        Returns:
        - observation: tensor (one-hot)
        - reward: float
        - done: bool
        """
        self._check_action(action)

        # If already terminal: match TMaze behavior (no-op, zero reward)
        if self._terminal(last=False):
            return self._observation(), 0.0, True

        self._transition(action)
        observation = self._observation(action=action)
        reward = self._reward(action)
        done = self._terminal(last=False)

        if self.bayes:
            self._update_belief(action, observation)

        return observation, reward, done

    # -------- Helpers --------

    def _check_action(self, action):
        if action < 0 or self.action_size <= action:
            size = self.action_size
            raise ValueError(f"The action should be in range [0, {size}[")

    def _init_state(self):
        """
        Samples initial hidden state p0(tiger_left)=0.5 and clears terminal.
        """
        self.tiger_left = (random.random() < 0.5)
        self.terminal = False

    def _terminal(self, last=False):
        """
        Tiger is terminal after opening a door.
        (The 'last' arg exists in TMaze; we keep it for stylistic symmetry.)
        """
        return bool(self.terminal)

    def _transition(self, action):
        """
        Transition dynamics:
          - LISTEN: hidden state unchanged
          - OPEN_*: go terminal (episode ends)
        """
        if action in (A_OPEN_LEFT, A_OPEN_RIGHT):
            self.terminal = True

    def _reward(self, action):
        """
        Reward function:
          - LISTEN: -1
          - OPEN_*: +10 if opened safe door, else -100
          - if previous was terminal, reward 0 (handled by early return in step)
        """
        if action == A_LISTEN:
            return self.reward_listen

        if action == A_OPEN_LEFT:
            return self.reward_wrong if self.tiger_left else self.reward_correct

        if action == A_OPEN_RIGHT:
            return self.reward_correct if self.tiger_left else self.reward_wrong

        raise ValueError("Unexpected action")

    def _observation(self, action=None):
        """
        Observation model:
          - After OPEN: always NULL
          - After LISTEN (or at reset): noisy 'hear_left/right' reflecting tiger position
        """
        if self.terminal:
            return O_NULL

        # If we just opened, also return NULL (even before terminal check above)
        if action in (A_OPEN_LEFT, A_OPEN_RIGHT):
            return O_NULL

        # LISTEN / reset: noisy hearing
        p = self.listen_accuracy
        if self.tiger_left:
            return O_HEAR_LEFT if (random.random() < p) else O_HEAR_RIGHT
        else:
            return O_HEAR_RIGHT if (random.random() < p) else O_HEAR_LEFT

    # -------- Exact Bayes belief --------

    def _init_belief(self, observation):
        """
        Initial belief b0 over tiger location: [0.5, 0.5].
        Conditioning on the initial observation is optional; since reset emits
        a noisy listen-like observation here, we update once for consistency.
        """
        self.belief = torch.tensor([0.5, 0.5], dtype=torch.float32)
        # If reset obs is informative (hear_left/right), incorporate it
        self._update_belief(action=A_LISTEN, observation=observation)

    def _update_belief(self, action, observation):
        """
        Bayes update only for LISTEN observations.
        Belief is over [P(tiger_left), P(tiger_right)].
        """
        if action != A_LISTEN:
            return

        p = self.listen_accuracy
        bL, bR = self.belief[0].item(), self.belief[1].item()

        if observation.argmax().item() == O_HEAR_LEFT.argmax().item():
            # P(o=HL | L)=p, P(o=HL | R)=1-p
            numL = bL * p
            numR = bR * (1.0 - p)
        elif observation.argmax().item() == O_HEAR_RIGHT.argmax().item():
            # P(o=HR | R)=p, P(o=HR | L)=1-p
            numL = bL * (1.0 - p)
            numR = bR * p
        else:
            # NULL carries no information
            return

        z = numL + numR
        if z <= 0.0:
            self.belief[:] = 0.5
        else:
            self.belief[0] = numL / z
            self.belief[1] = numR / z

    def get_belief(self):
        """
        Returns the current belief in the same shape/style as TMaze:
        a 1-tuple containing a cloned tensor.
        """
        return (self.belief.clone(),)
