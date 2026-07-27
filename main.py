import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from controller.config import load_config
from controller.game_controller import GameController
from game_engine import GameEngine

if __name__ == '__main__':
    controller = GameController()
    cfg = load_config()
    if cfg["move_border"] == [0, 0, 0, 0]:
        print("Borders not configured. Starting interactive setup...")
        controller.setup_interactive()
    else:
        print("Config loaded:", cfg)

    game = GameEngine()
    game.start_game(controller)
    while game.is_game:
        game.update()
