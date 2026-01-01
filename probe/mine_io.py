import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
import json


@dataclass
class MineConfig:
    hs_sizes: int
    belief_sizes: list[int]
    representation_sizes: list[Optional[int]]
    hidden_size: int
    num_layers: int
    alpha: float
    belief_part: Optional[int] = None


def save_mine_config(mine_id: str, agent_episode: int, cfg: MineConfig, root: str = "weights"):
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"{mine_id}-{agent_episode}-cfg.json")
    with open(path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_mine_config(mine_id: str, agent_episode: int, root: str = "weights") -> MineConfig:
    path = os.path.join(root, f"{mine_id}-{agent_episode}-cfg.json")
    with open(path, "r") as f:
        d: Dict[str, Any] = json.load(f)
    return MineConfig(**d)
