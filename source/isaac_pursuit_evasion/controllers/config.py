from pathlib import Path
from typing import Dict

import yaml


def load_controller_config(controller_name: str, drone_name: str) -> Dict:
    """Load a YAML configuration for a given controller and drone."""
    cfg_dir = Path(__file__).with_name("cfg")
    drone_key = str(drone_name).lower()
    cfg_path = cfg_dir / f"{controller_name}_{drone_key}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Controller config '{controller_name}' for drone '{drone_name}' not found at {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
