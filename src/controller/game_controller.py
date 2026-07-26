from controller.adb_api import ADBShell


def to_screen(borders: list[int], changes: list[int]) -> list[int]:
    return [borders[0] + (borders[2] - borders[0]) * (changes[0] + 1),
            borders[1] + (borders[3] - borders[1]) * (changes[1] + 1)]


class GameController:
    def __init__(self):
        self.adb = ADBShell()
        self.move_border = [0, 0, 0, 0]  # x1, y1, x2 ,y2
        self.shoot_border = [0, 0, 0, 0]  # x1, y1, x2 ,y2
        self.ult_center = [0, 0]
        self.last_move = [0, 0]  # in screen axes

    def start_game(self):
        pass

    def exit_game(self):
        pass

    def get_frame(self):
        pass

    def make_action(self, move, shoot, ult: bool):
        cur_move = to_screen(self.move_border, move)
        self.adb.swipe(self.last_move[0], self.last_move[1], cur_move[0], cur_move[1])
        self.last_move = cur_move

        # TODO: create shoot swipes in 8 vectors

        if ult:
            self.adb.tap(self.ult_center[0], self.ult_center[1])
