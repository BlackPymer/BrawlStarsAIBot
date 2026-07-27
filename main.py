from controller.game_controller import GameController
from game_engine import GameEngine

if __name__ == '__main__':
    game = GameEngine()
    controller = GameController()
    # TODO get required coordinates for controller
    game.start_game(controller)
    while game.is_game:
        game.update()
