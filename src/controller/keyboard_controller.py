import time

import keyboard

from controller.base_controller import BaseController
from controller.config import load_config
from controller.screen_capture import ScreenCapture

MATCH_LOAD_WAIT = 13

# 8 directions, N clockwise -> (primary, secondary) WASD keys
MOVE_KEYS = [
    ("w",),
    ("w", "d"),
    ("d",),
    ("s", "d"),
    ("s",),
    ("s", "a"),
    ("a",),
    ("w", "a"),
]
# shoot directions 0-7, 8 = tap center, 9 = no shot
SHOOT_KEYS = [
    ("up",),
    ("up", "right"),
    ("right",),
    ("down", "right"),
    ("down",),
    ("down", "left"),
    ("left",),
    ("up", "left"),
]
SHOOT_CENTER_KEY = "space"
ULT_KEY = "q"
SHOOT_HOLD_SECONDS = 0.15
ULT_HOLD_SECONDS = 0.15


class KeyboardController(BaseController):
    def __init__(self, config: dict | None = None):
        self._cap = ScreenCapture()
        self._config = config or load_config()
        self._held_keys = []

    def _release_all(self):
        for k in self._held_keys:
            keyboard.release(k)
        self._held_keys = []

    def _press_hold(self, keys):
        for k in keys:
            if k not in self._held_keys:
                keyboard.press(k)
                self._held_keys.append(k)

    def _quick_press(self, keys):
        self._release_all()
        for k in keys:
            keyboard.press(k)
        time.sleep(SHOOT_HOLD_SECONDS)
        for k in keys:
            keyboard.release(k)

    def start_game(self):
        battle = self._config.get("battle_click", [0, 0])
        if battle != [0, 0]:
            import pyautogui
            pyautogui.click(battle[0], battle[1])
        time.sleep(MATCH_LOAD_WAIT)

    def exit_game(self):
        self._release_all()

    def get_frame(self):
        return self._cap.get_frame()

    def make_action(self, move: int, shoot: int, ult: bool):
        # movement: continuous hold, release when new direction differs or stop
        if 0 <= move <= 7:
            target = MOVE_KEYS[move]
            if self._held_keys != list(target):
                self._release_all()
                self._press_hold(target)
        else:
            self._release_all()

        if shoot <= 7:
            self._quick_press(SHOOT_KEYS[shoot])
        elif shoot == 8:
            self._quick_press((SHOOT_CENTER_KEY,))

        if ult:
            self._quick_press((ULT_KEY,))

    def setup_interactive(self):
        return False
