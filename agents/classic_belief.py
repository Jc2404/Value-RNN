import math
import hashlib
from random import random
from typing import Dict, List, Tuple

import torch

from .memory import Trajectory


class BeliefPolicy:
    """
    Supported:
      - Tiger
      - CryingBaby
      - TMaze
      - StarkweatherEnv

      - eval(environment, num_rollouts)
      - play(environment, epsilon=0.0, return_beliefs=False)
    """

    def __init__(self, planning_horizon=None, belief_round_ndigits=10):
        self.planning_horizon = planning_horizon
        self.belief_round_ndigits = int(belief_round_ndigits)
        self._cache: Dict[Tuple[str, int, Tuple[float, ...]], float] = {}
        self._env_signature = None

    def eval(self, environment, num_rollouts):
        sum_returns, disc_returns = 0.0, 0.0

        for _ in range(num_rollouts):
            trajectory, = self.play(environment, epsilon=0.0)
            sum_returns += trajectory.get_cumulative_reward()
            disc_returns += trajectory.get_cumulative_reward(environment.gamma)

        mean_return = sum_returns / num_rollouts
        mean_disc_return = disc_returns / num_rollouts
        return mean_return, mean_disc_return

    def play(self, environment, epsilon=0.0, return_beliefs=False):
        self._prepare_environment(environment)
        self._reset_cache(environment)

        beliefs = []

        o = environment.reset()
        trajectory = Trajectory(environment.action_size, environment.observation_size)
        trajectory.add(None, None, o)

        for t in range(environment.horizon()):
            if return_beliefs:
                beliefs.append(environment.get_belief())

            if random() < epsilon:
                a = environment.exploration()
            else:
                remaining = self._remaining_steps(environment, t)
                a = self.act(environment, remaining_steps=remaining)

            o, r, d = environment.step(a)
            trajectory.add(a, r, o, terminal=d)

            if d:
                break

        return_values = (trajectory,)
        if return_beliefs:
            return_values += (beliefs,)
        return return_values

    def act(self, environment, remaining_steps=None):
        self._prepare_environment(environment)
        self._reset_cache(environment)

        if remaining_steps is None:
            remaining_steps = self._planning_horizon(environment)

        belief = environment.get_belief()[0].detach().clone().float()
        q_values = self.q_values(environment, belief, remaining_steps)
        return int(torch.argmax(q_values).item())

    def q_values(self, environment, belief, remaining_steps=None):
        self._prepare_environment(environment)
        self._reset_cache(environment)

        if remaining_steps is None:
            remaining_steps = self._planning_horizon(environment)

        qs = []
        for action in range(environment.action_size):
            q = self._q_value(environment, belief, action, remaining_steps)
            qs.append(q)
        return torch.tensor(qs, dtype=torch.float32)

    # Planning
    def _q_value(self, environment, belief, action, remaining_steps):
        if remaining_steps <= 0:
            return 0.0

        total = 0.0
        branches = self._branches(environment, belief, action)
        gamma = float(environment.gamma)

        for prob, reward, next_belief, terminal in branches:
            if prob <= 0.0:
                continue
            cont = 0.0
            if (not terminal) and remaining_steps > 1:
                cont = gamma * self._value(environment, next_belief, remaining_steps - 1)
            total += prob * (reward + cont)

        return float(total)

    def _value(self, environment, belief, remaining_steps):
        if remaining_steps <= 0: # end
            return 0.0

        key = (self._env_signature, int(remaining_steps), self._belief_key(belief))
        if key in self._cache:
            return self._cache[key]

        best = -float("inf")
        for action in range(environment.action_size):
            q = self._q_value(environment, belief, action, remaining_steps)
            if q > best:
                best = q

        self._cache[key] = float(best)
        return float(best)

    # Environment-specifics, will incorporate into environment classes
    def _branches(self, environment, belief, action):
        name = environment.__class__.__name__.lower()
        if name == "tiger":
            return self._tiger(environment, belief, action)
        if name == "cryingbaby":
            return self._crybaby(environment, belief, action)
        if name == "tmaze":
            return self._tmaze(environment, belief, action)
        if name == "starkweatherenv":
            return self._stark(environment, belief, action)
        raise NotImplementedError(
            f"Does not support {environment.__class__.__name__} yet"
        )

    def _tiger(self, env, belief, action):
        b = self._normalize(belief)
        b_left = float(b[0].item())
        b_right = float(b[1].item())

        # action 0 listen, 1 open_left, 2 open_right
        if action == 0:
            p = float(env.listen_accuracy)

            p_hear_left = b_left * p + b_right * (1.0 - p)
            p_hear_right = b_left * (1.0 - p) + b_right * p

            next_left = torch.tensor([
                b_left * p,
                b_right * (1.0 - p),
            ], dtype=torch.float32)
            next_left = self._normalize(next_left)

            next_right = torch.tensor([
                b_left * (1.0 - p),
                b_right * p,
            ], dtype=torch.float32)
            next_right = self._normalize(next_right)

            return [
                (p_hear_left, float(env.reward_listen), next_left, False),
                (p_hear_right, float(env.reward_listen), next_right, False),
            ]

        if action == 1:  # open_left
            reward = b_left * float(env.reward_wrong) + b_right * float(env.reward_correct)
            return [(1.0, reward, b.clone(), True)]

        if action == 2:  # open_right
            reward = b_left * float(env.reward_correct) + b_right * float(env.reward_wrong)
            return [(1.0, reward, b.clone(), True)]

    def _crybaby(self, env, belief, action):
        b = self._normalize(belief)
        T = env.T[action].float() # [s, s']
        O = env.O.float() # [s', o]

        b_pred = b @ T # [s']
        branches = []

        for o in range(env.observation_size):
            likelihood = O[:, o]
            post_unnorm = b_pred * likelihood
            p_obs = float(post_unnorm.sum().item())
            if p_obs <= 0.0:
                continue
            b_next = self._normalize(post_unnorm)
            reward = float(env.reward_cry) if o == 0 else float(env.reward_quiet)
            if action == 1:  # FEED
                reward += float(env.cost_feed)
            branches.append((p_obs, reward, b_next, False))

        return branches

    def _tmaze(self, env, belief, action):
        b = self._normalize(belief)
        T = env.T[action].float() # [s', s]
        O = env.O # dict[o] -> [s']

        # predicted next-state distribution
        pred = T @ b # [s']

        # joint mass over (s, s') for expected immediate reward
        joint = T * b.unsqueeze(0) # [s', s]

        branches = []
        for o in range(env.observation_size):
            obs_mask = O[o].float()        # [s']
            post_unnorm = obs_mask * pred
            p_obs = float(post_unnorm.sum().item())
            if p_obs <= 0.0:
                continue

            reward_num = 0.0
            for s_next in range(env.K):
                if float(obs_mask[s_next].item()) <= 0.0:
                    continue
                for s in range(env.K):
                    p_ssp = float(joint[s_next, s].item())
                    if p_ssp <= 0.0:
                        continue
                    reward_num += p_ssp * self._tmaze_reward(env, s, s_next)

            reward = reward_num / p_obs
            b_next = self._normalize(post_unnorm)
            branches.append((p_obs, reward, b_next, False))

        return branches

    def _stark(self, env, belief, action):
        b = self._normalize(belief)
        T = env.T.float() # [s, s']
        O = env.O.float() # [s, s', x]

        branches = []
        for x in range(env.observation_size):
            weighted = T * O[:, :, x] # [s, s']
            next_unnorm = b @ weighted # [s']
            p_obs = float(next_unnorm.sum().item())
            if p_obs <= 0.0:
                continue

            b_next = self._normalize(next_unnorm)
            reward = 1.0 if x == 2 else 0.0
            terminal = (x == 2)
            branches.append((p_obs, reward, b_next, terminal))

        return branches

    # Environment-specific helpers
    def _tmaze_reward(self, env, s, s_next):
        length = int(env.length)

        goal_up = (s < (length + 3))
        position = s % (length + 3)
        next_position = s_next % (length + 3)

        # If the previous state was terminal, reward is zero.
        if length + 1 <= position <= length + 2:
            return 0.0

        # Bumped into wall / no movement
        if position == next_position:
            return -0.1

        # Corridor / crossroad
        if 0 <= next_position <= length:
            return 0.0

        # Reached terminal branch.
        if length + 1 <= next_position <= length + 2:
            if goal_up and next_position == length + 1:
                return 4.0
            if (not goal_up) and next_position == length + 2:
                return 4.0
            return -0.1

        raise ValueError("Unexpected TMaze state transition")

    def _prepare_environment(self, environment):
        if not hasattr(environment, "get_belief"):
            raise ValueError("Not implemented get_belief()")
        if hasattr(environment, "belief_type") and environment.belief_type != "exact":
            raise NotImplementedError(
                f"Only supports discrete beliefs, got {environment.belief_type}"
            )
        if hasattr(environment, "bayes") and (not environment.bayes):
            environment.bayes = True

    def _planning_horizon(self, environment):
        if self.planning_horizon is not None:
            return int(self.planning_horizon)
        return int(environment.horizon())

    def _remaining_steps(self, environment, t):
        return max(0, self._planning_horizon(environment) - int(t))

    def _normalize(self, belief):
        belief = belief.detach().clone().float()
        z = belief.sum().clamp_min(1e-12)
        return belief / z

    def _belief_key(self, belief):
        b = self._normalize(belief)
        vals = [round(float(x), self.belief_round_ndigits) for x in b.tolist()]
        return tuple(vals)

    def _reset_cache(self, environment):
        sig = self._environment_config(environment)
        if sig != self._env_signature:
            self._cache = {}
            self._env_signature = sig

    def _environment_config(self, environment): # temporary gpt solution to fix bug for reusing the policy
        items = [environment.__class__.__name__, int(environment.action_size), int(environment.observation_size)]
        for k, v in sorted(vars(environment).items()):
            if k in {"belief", "state", "steps", "terminal", "position", "last_position", "tiger_left", "goal_up"}:
                continue
            if torch.is_tensor(v):
                items.append((k, tuple(v.shape), self._tensor_digest(v)))
            elif isinstance(v, dict):
                dict_items = []
                for dk, dv in sorted(v.items(), key=lambda x: x[0]):
                    if torch.is_tensor(dv):
                        dict_items.append((dk, tuple(dv.shape), self._tensor_digest(dv)))
                    else:
                        dict_items.append((dk, repr(dv)))
                items.append((k, tuple(dict_items)))
            elif isinstance(v, (int, float, bool, str)):
                items.append((k, v))
        text = repr(items).encode("utf-8")
        return hashlib.md5(text).hexdigest()

    def _tensor_digest(self, x):
        x = x.detach().cpu().contiguous()
        return hashlib.md5(x.numpy().tobytes()).hexdigest()
