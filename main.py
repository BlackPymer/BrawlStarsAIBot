MODE = "rl"  # "rl" | "game" — режим запуска: rl=обучение с подкреплением, game=игра

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from controller.config import load_config
from game_engine import GameEngine


def make_controller(cfg):
    ctrl_type = cfg.get("controller", "keyboard")
    if ctrl_type == "adb":
        from controller.game_controller import GameController
        controller = GameController()
        if cfg.get("move_border") == [0, 0, 0, 0]:
            print("Borders not configured. Starting interactive setup...")
            if not controller.setup_interactive():
                print("Setup failed. Exiting.")
                sys.exit(1)
            print("Config saved:", load_config())
        else:
            print("Config loaded:", cfg)
    else:
        from controller.keyboard_controller import KeyboardController
        controller = KeyboardController()
        print(f"Keyboard controller enabled ({ctrl_type})")
    return controller


if __name__ == '__main__':
    cfg = load_config()

    if MODE == "rl":
        from utilities.train_rl import main as rl_main
        rl_main()
        sys.exit(0)

    controller = make_controller(cfg)
    controller.setup_binds()

    game = GameEngine()
    game.start_game(controller)
    while game.is_game:
        game.update()