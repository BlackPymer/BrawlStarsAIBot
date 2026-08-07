import math
import time

import keyboard

from controller.base_controller import BaseController
from controller.config import load_config
from controller.screen_capture import ScreenCapture

MATCH_LOAD_WAIT = 13

# key -> unit vector in screen coords (y grows down). N is up.
KEY_VECTORS = {
    "w": (0, -1),
    "s": (0, 1),
    "a": (-1, 0),
    "d": (1, 0),
    "num 8": (0, -1),
    "num 2": (0, 1),
    "num 4": (-1, 0),
    "num 6": (1, 0),
    "num 7": (-1, -1),
    "num 9": (1, -1),
    "num 1": (-1, 1),
    "num 3": (1, 1),
}
# N clockwise
DIRECTIONS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]


def direction_class(keys) -> int | None:
    vx = vy = 0.0
    for k in keys:
        v = KEY_VECTORS.get(k)
        if v is None:
            continue
        vx += v[0]
        vy += v[1]
    if vx == 0 and vy == 0:
        return None
    angle = math.atan2(vy, vx)
    idx = int(round((angle + math.pi / 2) / (math.pi / 4))) % 8
    return idx

# move class -> keys to hold (8 dirs + stop). N clockwise.
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
# shoot directions 0-7 (numpad arrows), 8 = auto-attack (space), 9 = no shot
SHOOT_KEYS = [
    ("num 8",),
    ("num 9",),
    ("num 6",),
    ("num 3",),
    ("num 2",),
    ("num 1",),
    ("num 4",),
    ("num 7",),
]
SHOOT_CENTER_KEY = "space"
ULT_KEY = "shift"
SHOOT_HOLD_SECONDS = 0.15
ULT_HOLD_SECONDS = 0.15


class KeyboardController(BaseController):
    def __init__(self, config: dict | None = None):
        self._cap = ScreenCapture()
        self._config = config or load_config()
        self._held_keys = []
        self._shot_keys = []

    def _release_move(self):
        for k in self._held_keys:
            keyboard.release(k)
        self._held_keys = []

    def _release_shot(self):
        for k in self._shot_keys:
            keyboard.release(k)
        self._shot_keys = []

    def _release_all(self):
        self._release_move()
        self._release_shot()

    def _press_hold(self, keys):
        for k in keys:
            if k not in self._held_keys:
                keyboard.press(k)
                self._held_keys.append(k)

    def _quick_press(self, keys):
        self._release_shot()
        for k in keys:
            keyboard.press(k)
            self._shot_keys.append(k)
        time.sleep(SHOOT_HOLD_SECONDS)
        self._release_shot()

    def start_game(self):
        battle = self._config.get("battle_click", [0, 0])
        if battle == [0, 0]:
            print("[BOT] Battle button not configured.")
            print("[BOT] Hover the mouse over the 'Play/Battle' button in BlueStacks and press Enter...")
            print("[BOT] Press Esc to cancel.")
            while True:
                if keyboard.is_pressed("esc"):
                    print("[BOT] Cancelled start. Exiting.")
                    return
                if keyboard.is_pressed("enter"):
                    break
                time.sleep(0.05)
            import pyautogui
            battle = list(pyautogui.position())
            self._config["battle_click"] = battle
            from controller.config import save_config
            save_config(self._config)
            print(f"[BOT] Battle button saved: {battle}")
        else:
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
                self._release_move()
                self._press_hold(target)
        else:
            self._release_move()

        if shoot <= 7:
            self._quick_press(SHOOT_KEYS[shoot])
        elif shoot == 8:
            self._quick_press((SHOOT_CENTER_KEY,))

        if ult:
            self._quick_press((ULT_KEY,))

    def setup_interactive(self):
        return False
