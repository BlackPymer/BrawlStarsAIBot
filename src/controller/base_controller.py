class BaseController:
    def start_game(self):
        pass

    def exit_game(self):
        pass

    def get_frame(self):
        return None

    def make_action(self, move: int, shoot: int, ult: bool):
        pass

    def setup_interactive(self):
        return False