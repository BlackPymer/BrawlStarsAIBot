from controller.adb_api import ADBShell
from controller.config import load_config
from controller.screen_capture import ScreenCapture


def to_screen(borders: list[int], changes: list[int]) -> list[int]:
    return [borders[0] + (borders[2] - borders[0]) * (changes[0] + 1),
            borders[1] + (borders[3] - borders[1]) * (changes[1] + 1)]


class GameController:
    def __init__(self):
        self.adb = ADBShell()
        self.last_move = None
        self.axes = []
        self.shoot_center = None
        self.move_border = [0, 0, 0, 0]
        self.shoot_border = [0, 0, 0, 0]
        self.ult_center = [0, 0]
        self._cap = ScreenCapture()
        self._load_config()

    def _load_config(self):
        cfg = load_config()
        if cfg["move_border"] != [0, 0, 0, 0]:
            self.set_move_border(cfg["move_border"])
        if cfg["shoot_border"] != [0, 0, 0, 0]:
            self.set_shoot_border(cfg["shoot_border"])
        if cfg["ult_center"] != [0, 0]:
            self.ult_center = cfg["ult_center"]

    def setup_interactive(self):
        from controller.border_setup import setup_borders
        cfg = setup_borders()
        if cfg:
            self.set_move_border(cfg["move_border"])
            self.set_shoot_border(cfg["shoot_border"])
            self.ult_center = cfg["ult_center"]

    def start_game(self):
        pass

    def exit_game(self):
        pass

    def get_frame(self):
        return self._cap.get_frame()

    def set_move_border(self, borders: list[int]):
        self.move_border = borders

    def set_shoot_border(self, borders: list[int]):
        self.shoot_border = borders
        self.axes = [
            (borders[0], borders[1]),
            ((borders[0] + borders[2]) // 2, borders[1]),
            (borders[2], borders[1]),
            (borders[2], (borders[1] + borders[3]) // 2),
            (borders[2], borders[3]),
            ((borders[0] + borders[2]) // 2, borders[3]),
            (borders[0], borders[3]),
            (borders[0], (borders[1] + borders[3]) // 2),
        ]
        self.shoot_center = ((borders[0] + borders[2]) // 2, (borders[1] + borders[3]) // 2)

    def make_action(self, move, shoot: int, ult: bool):
        cur_move = to_screen(self.move_border, move)
        if self.last_move is not None:
            self.adb.swipe(self.last_move[0], self.last_move[1], cur_move[0], cur_move[1])
        else:
            self.adb.swipe(cur_move[0], cur_move[1], cur_move[0], cur_move[1])
        self.last_move = cur_move

        if shoot < 8:
            self.adb.swipe(*self.shoot_center, *self.axes[shoot])
        elif shoot == 8:
            self.adb.tap(*self.shoot_center)

        if ult:
            self.adb.tap(self.ult_center[0], self.ult_center[1])
