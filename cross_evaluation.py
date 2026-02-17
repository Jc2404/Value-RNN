import torch

from agents.drqn import DRQN


def evaluate_trained_agent_on_env(
    *,
    run_id: str,
    episode: int,
    make_train_env,
    make_test_env,
    cell: str,
    hidden_size: int,
    num_layers: int,
    num_rollouts: int = 50,
    device: str = "cpu",
):
    """
    Evaluate a saved DRQN policy (weights/{run_id}-{episode}-Q.pth) on a different environment.

    IMPORTANT: This only works if the test environment has the SAME:
      - action_size
      - observation_size
    as the environment the agent was trained with (because the network I/O dims are fixed).

    Returns
    -------
    result : dict
        {
          'mean_return': float,
          'mean_disc_return': float,
          'train_env': str,
          'test_env': str,
          'num_rollouts': int,
        }
    """
    train_env = make_train_env()
    test_env = make_test_env()

    if train_env.action_size != test_env.action_size:
        raise ValueError(
            f"action_size mismatch: train={train_env.action_size}, test={test_env.action_size}. "
            "You need the same discrete action space to reuse the trained Q-network."
        )
    if train_env.observation_size != test_env.observation_size:
        raise ValueError(
            f"observation_size mismatch: train={train_env.observation_size}, test={test_env.observation_size}. "
            "Observation encoding dimensionality must match to reuse the trained Q-network."
        )

    agent = DRQN(
        cell=cell,
        action_size=train_env.action_size,
        observation_size=train_env.observation_size,
        num_layers=num_layers,
        hidden_size=hidden_size,
    )

    agent.load(run_id, episode=episode)  # loads weights/{run_id}-{episode}-Q.pth and Q_tar :contentReference[oaicite:3]{index=3}

    agent.Q.to(device)
    agent.Q_tar.to(device)
    mean_return, mean_disc_return = agent.eval(test_env, num_rollouts)  # :contentReference[oaicite:4]{index=4}

    return {
        "mean_return": float(mean_return),
        "mean_disc_return": float(mean_disc_return),
        "train_env": type(train_env).__name__,
        "test_env": type(test_env).__name__,
        "num_rollouts": int(num_rollouts),
    }
