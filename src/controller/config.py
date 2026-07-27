import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config.json")

DEFAULT_CONFIG = {
    "move_border": [0, 0, 0, 0],
    "shoot_border": [0, 0, 0, 0],
    "ult_center": [0, 0],
    "capture_resolution": [1920, 1080],
}


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        return dict(DEFAULT_CONFIG)
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def save_config(config: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
