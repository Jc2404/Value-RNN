# environments/gridworld_pomdp.py
# Discrete GridWorld turned into a POMDP by observing ONLY local walls (N/S/W/E).
# - No (-1,-1) terminal state (explicitly removed as requested).
# - Terminal condition uses self._terminal() (horizon) OR reaching terminate_from (reward cells).
# - Belief is exact categorical over grid positions.

import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import torch


# Action indices (match your style: ints)
A_UP, A_DOWN, A_LEFT, A_RIGHT = 0, 1, 2, 3
ACTIONS = (A_UP, A_DOWN, A_LEFT, A_RIGHT)

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _pos_from_margin(W, H, left=None, right=None, bottom=None, top=None):
    """
    Define a position by margins to the walls.
    Exactly one of left/right must be set; exactly one of bottom/top must be set.
    Margins are 0-based: margin=0 means on the wall; margin=1 means 1 cell away, etc.
    """
    assert (left is None) ^ (right is None)
    assert (bottom is None) ^ (top is None)

    if left is not None:
        x = 1 + left
    else:
        x = W - right

    if bottom is not None:
        y = 1 + bottom
    else:
        y = H - top

    # Clamp to valid grid (important if W/H small)
    x = _clamp(x, 1, W)
    y = _clamp(y, 1, H)
    return (x, y)

@dataclass(frozen=True)
class GridPos:
    x: int
    y: int


class GridWorld:
    """
    SimpleGridWorld as a discrete POMDP.

    Hidden state s = (x,y) in 1..W, 1..H.
    Observation o(s) is a 4-dim binary vector encoding WALLS around agent:
        o = [wall_up, wall_down, wall_left, wall_right]  each in {0,1}.
    - At corners, two entries are 1 (e.g., top-left => wall_up=1, wall_left=1).
    - Deterministic observation: P(o | s) is 1 if o == o(s), else 0.

    Transition:
      Intended move succeeds with prob tprob.
      Remaining prob distributed uniformly among the other 3 moves.
      If a move would go out of bounds, agent stays in place ("bounce back").

    Reward:
      R(s) = rewards.get(s, 0.0)
      (delivered when *arriving* in s_{t+1} after the transition; typical RL convention)

    Termination:
      done if steps >= max_steps OR state in terminate_from (default: reward cells).
      No artificial (-1,-1) terminal.

    Interface:
      - reset() -> obs (torch float tensor, shape [4])
      - step(a) -> (obs, reward: float, done: bool)
      - exploration() -> random action
      - horizon() -> max_steps
      - get_belief() -> (belief_vector,) if bayes=True
    """

    gamma = 0.95
    action_size = 4
    observation_size = 4
    belief_type = "exact"

    def __init__(
        self,
        size: int = 10,
        tprob: float = 0.7,
        discount: float = 0.95,
        max_steps: int = 200,
        bayes: bool = True,
        seed: Optional[int] = None,
        reward_scheme: str = "julia",
        reward_margin: int = 2,          # for symmetric/landmark schemes
        step_cost: float = 0.0,          # optional living cost
    ):
        self.W, self.H = size, size
        self.tprob = float(tprob)
        self.gamma = float(discount)
        self.max_steps = int(max_steps)
        self.bayes = bool(bayes)
        self.step_cost = float(step_cost)

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # default rewards (Julia)
        rewards = self._build_rewards(
            scheme=reward_scheme,
            margin=int(reward_margin))
        self.rewards: Dict[GridPos, float] = {GridPos(x, y): float(r) for (x, y), r in rewards.items()}

        terminate_from = [(p.x, p.y) for p in self.rewards.keys()]
        self.terminate_from = {GridPos(int(x), int(y)) for (x, y) in terminate_from}

        if not (0.0 <= self.tprob <= 1.0):
            raise ValueError(f"tprob must be in [0,1], got {self.tprob}")

        # enumerate states and index map
        self.states: List[GridPos] = [GridPos(x, y) for x in range(1, self.W + 1) for y in range(1, self.H + 1)]
        self.K = len(self.states)  # for your state decoder pipeline if needed
        self._idx: Dict[GridPos, int] = {s: i for i, s in enumerate(self.states)}

        # precompute deterministic wall-observation for each state: [K,4]
        self._obs_table = torch.stack([self._walls_obs(s) for s in self.states], dim=0)  # float32

        # precompute transition distributions: next_indices[a][i] -> (idxs, probs)
        self._transitions = self._precompute_transitions()

        # runtime
        self.state: Optional[GridPos] = None
        self.steps: int = 0
        self.belief: Optional[torch.Tensor] = None  # [K]

    def _build_rewards(self, scheme: str, margin: int):
        """
        Returns a dict {(x,y): reward}.
        All coordinates are 1-indexed.
        """
        W, H = self.W, self.H
        m = max(0, int(margin))

        scheme = scheme.lower()

        if scheme == "julia":
            # this is NOT size-robust; keep only if size is big enough.
            return {
                (4, 3): -10.0,
                (4, 6): -5.0,
                (9, 3): 10.0,
                (8, 8): 3.0,
            }

        if scheme == "symmetric":
            # Julia-like magnitudes but placed symmetrically by margin m from walls.
            # Positions (2 blocks from walls by default):
            # TL, TR, BL, BR with a margin m from both relevant walls
            TL = _pos_from_margin(W, H, left=m,  top=m)
            TR = _pos_from_margin(W, H, right=m, top=m)
            BL = _pos_from_margin(W, H, left=m,  bottom=m)
            BR = _pos_from_margin(W, H, right=m, bottom=m)

            # Assign Julia magnitudes symmetrically:
            # positives at top corners, negatives at bottom corners
            return {
                TL: 3.0,
                TR: 10.0,
                BL: -5.0,
                BR: -10.0,
            }

        if scheme == "center":
            # Center goal (+10) and symmetric traps placed by margin m from walls.
            cx = (W + 1) // 2
            cy = (H + 1) // 2

            # trap candidates at margin-from-walls "ring"
            TL = _pos_from_margin(W, H, left=m,  top=m)
            TR = _pos_from_margin(W, H, right=m, top=m)
            BL = _pos_from_margin(W, H, left=m,  bottom=m)
            BR = _pos_from_margin(W, H, right=m, bottom=m)

            rew = {(cx, cy): 10.0}
            # symmetric traps
            rew.update({TL: -10.0, TR: -10.0, BL: -10.0, BR: -10.0})
            return rew

        if scheme == "scaled":
            # Example: size-scaled landmarks at fixed fractions (keeps asymmetry but size-robust).
            # You can tune these fractions.
            pts = [
                (0.40, 0.30, -10.0),
                (0.40, 0.60, -5.0),
                (0.90, 0.30, 10.0),
                (0.80, 0.80, 3.0),
            ]
            out = {}
            for fx, fy, r in pts:
                x = _clamp(int(round(fx * (W - 1))) + 1, 1, W)
                y = _clamp(int(round(fy * (H - 1))) + 1, 1, H)
                out[(x, y)] = float(r)
            return out

        raise ValueError(f"Unknown reward_scheme={scheme}")

    def horizon(self):
        return self.max_steps

    def exploration(self):
        return random.choice(ACTIONS)

    def reset(self):
        self.steps = 0
        self.state = self._sample_initial_state()

        obs = self._obs_from_state(self.state)

        if self.bayes:
            self._init_belief(obs)

        return obs

    def step(self, action: int):
        self._check_action(action)

        # If already terminal, no-op
        if self._terminal() or (self.state in self.terminate_from):
            obs = self._obs_from_state(self.state) if self.state is not None else torch.zeros(self.observation_size)
            return obs, 0.0, True

        self.steps += 1

        # sample transition
        s_idx = self._idx[self.state]
        idxs, probs = self._transitions[action][s_idx]
        next_idx = torch.distributions.Categorical(probs=probs).sample().item()
        self.state = self.states[int(idxs[next_idx].item())]

        # observation is deterministic walls around new state
        obs = self._obs_from_state(self.state)

        # reward based on new state (arrival reward)
        reward = float(self.rewards.get(self.state, 0.0)) + self.step_cost

        done = self._terminal() or (self.state in self.terminate_from)

        if self.bayes:
            self._update_belief(action, obs)

        return obs, reward, done

    def get_belief(self):
        if self.belief is None:
            raise RuntimeError("Belief is not initialized (bayes=False or reset() not called).")
        return (self.belief.clone(),)

    def _terminal(self) -> bool:
        return self.steps >= self.max_steps

    def _check_action(self, action: int):
        if action < 0 or action >= self.action_size:
            raise ValueError(f"Action must be in [0, {self.action_size}), got {action}")

    def _inbounds(self, s: GridPos) -> bool:
        return 1 <= s.x <= self.W and 1 <= s.y <= self.H

    def _sample_initial_state(self) -> GridPos:
        while True:
            x = random.randint(1, self.W)
            y = random.randint(1, self.H)
            s = GridPos(x, y)
            if s not in self.terminate_from:
                return s

    def _move(self, s: GridPos, a: int) -> GridPos:
        if a == A_UP:
            return GridPos(s.x, s.y + 1)
        if a == A_DOWN:
            return GridPos(s.x, s.y - 1)
        if a == A_LEFT:
            return GridPos(s.x - 1, s.y)
        if a == A_RIGHT:
            return GridPos(s.x + 1, s.y)
        raise ValueError("bad action")

    def _walls_obs(self, s: GridPos) -> torch.Tensor:
        """
        Returns [wall_up, wall_down, wall_left, wall_right] as float32 tensor.
        Corner => two 1s, edges => one 1, interior => all 0.
        """
        wall_up = 1.0 if s.y == self.H else 0.0
        wall_down = 1.0 if s.y == 1 else 0.0
        wall_left = 1.0 if s.x == 1 else 0.0
        wall_right = 1.0 if s.x == self.W else 0.0
        return torch.tensor([wall_up, wall_down, wall_left, wall_right], dtype=torch.float32)

    def _obs_from_state(self, s: GridPos) -> torch.Tensor:
        return self._obs_table[self._idx[s]].clone()

    def _precompute_transitions(self):
        """
        For each action a and state index i, build a sparse categorical over next state indices.

        We implement:
          P(move intended) = tprob
          P(move other) = (1-tprob)/3
        If move would go out of bounds, it contributes to "stay in place".
        """
        trans = {a: [None] * self.K for a in ACTIONS}

        for a in ACTIONS:
            for i, s in enumerate(self.states):
                # probabilities mass by destination index
                mass = torch.zeros(self.K, dtype=torch.float32)

                for act in ACTIONS:
                    p = self.tprob if act == a else (1.0 - self.tprob) / 3.0
                    dest = self._move(s, act)
                    if self._inbounds(dest):
                        j = self._idx[dest]
                        mass[j] += p
                    else:
                        # bounce back -> stay
                        mass[i] += p

                # compress to sparse
                nz = torch.nonzero(mass > 0, as_tuple=False).view(-1)
                probs = mass[nz]
                probs = probs / probs.sum().clamp_min(1e-12)
                trans[a][i] = (nz.to(torch.int64), probs)

        return trans

    def _init_belief(self, obs: torch.Tensor):
        # uniform over states not in terminate_from
        b = torch.ones(self.K, dtype=torch.float32)
        for s in self.terminate_from:
            b[self._idx[s]] = 0.0
        b = b / b.sum().clamp_min(1e-12)

        b = self._apply_obs_likelihood(b, obs)
        self.belief = b

    def _update_belief(self, action: int, obs: torch.Tensor):
        """
        b'(s') ∝ 1{o(s') = obs} * sum_s b(s) T(s'|s,a)
        """
        # predict step: b_pred(s') = sum_s b(s) T(s'|s,a)
        # We'll compute via sparse transitions for efficiency.
        b_pred = torch.zeros_like(self.belief)

        for i, b_i in enumerate(self.belief):
            if b_i.item() == 0.0:
                continue
            idxs, probs = self._transitions[action][i]  # idxs are next-state indices
            b_pred[idxs] += b_i * probs

        b_post = self._apply_obs_likelihood(b_pred, obs)
        self.belief = b_post

    def _apply_obs_likelihood(self, b: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
        Deterministic observation:
          likelihood(s) = 1 if obs_table[s]==obs else 0
        """
        # exact match on all 4 bits
        match = (self._obs_table == obs.view(1, -1)).all(dim=1).float()
        b = b * match
        z = b.sum().clamp_min(1e-12)
        return b / z


if __name__ == "__main__":
    env = GridWorld(bayes=True, max_steps=20)
    obs = env.reset()
    print("reset obs:", obs.tolist(), "belief sum:", env.get_belief()[0].sum().item())

    total = 0.0
    for t in range(50):
        a = env.exploration()
        obs, r, done = env.step(a)
        total += r
        if t < 5:
            print(f"t={t} a={a} obs={obs.tolist()} r={r:.1f} done={done} "
                  f"belief entropy~={-(env.get_belief()[0].clamp_min(1e-12)*env.get_belief()[0].clamp_min(1e-12).log()).sum().item():.3f}")
        if done:
            break
    print("total reward:", total)
