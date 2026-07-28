import time

from controller.adb_api import ADBShell
from controller.config import load_config
from controller.screen_capture import ScreenCapture

TAP_OFFSET = 20
MATCH_LOAD_WAIT = 13


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
        self._move_held = False
        self._move_pos = None
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
            return True
        return False

    def start_game(self):
        w = self.adb._screen_w or 1600
        h = self.adb._screen_h or 900
        self.adb.tap(w - TAP_OFFSET, h - TAP_OFFSET)
        time.sleep(MATCH_LOAD_WAIT)
        self.adb._chain_touches()

    def exit_game(self):
        if self._move_held:
            self.adb._chain_touches()
            self._move_held = False
            self._move_pos = None

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
        if not self._move_held:
            self.adb._chain_touches(tuple(cur_move))
            self._move_held = True
            self._move_pos = cur_move
        elif cur_move != self._move_pos:
            self.adb._chain_touches(tuple(cur_move))
            self._move_pos = cur_move

        if shoot < 8:
            self.adb._cmd(" && ".join([
                *self.adb._to_chain(tuple(cur_move), tuple(self.shoot_center)),
                *self.adb._to_chain(tuple(cur_move), tuple(self.axes[shoot])),
                *self.adb._to_chain(tuple(cur_move)),
            ]))
        elif shoot == 8:
            self.adb._cmd(" && ".join([
                *self.adb._to_chain(tuple(cur_move), tuple(self.shoot_center)),
                *self.adb._to_chain(tuple(cur_move)),
            ]))

        if ult:
            self.adb._cmd(" && ".join([
                *self.adb._to_chain(tuple(cur_move), tuple(self.ult_center)),
                "sleep 0.015",
                *self.adb._to_chain(tuple(cur_move)),
            ]))

        self.last_move = cur_move
