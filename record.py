import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cv2
import keyboard

from controller.screen_capture import ScreenCapture
from controller.keyboard_controller import direction_class

FPS = 10
OUT_DIR = os.path.join(os.path.dirname(__file__), "recorded_data")
MOVE_KEYS = ("w", "a", "s", "d")
SHOOT_KEYS = ("num 8", "num 2", "num 4", "num 6", "num 7", "num 9", "num 1", "num 3")
ULT_KEY = "shift"
AUTO_AIM_KEY = "space"


def read_action():
    move_keys = [k for k in MOVE_KEYS if keyboard.is_pressed(k)]
    shoot_keys = [k for k in SHOOT_KEYS if keyboard.is_pressed(k)]
    move = direction_class(move_keys)
    if keyboard.is_pressed(AUTO_AIM_KEY):
        shoot = 8  # класс "автоатака в центр"
    else:
        shoot = direction_class(shoot_keys)
    ult = keyboard.is_pressed(ULT_KEY)
    return (move if move is not None else 8), (shoot if shoot is not None else 9), int(ult)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    run = time.strftime("rec_%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUT_DIR, run)
    frames_dir = os.path.join(run_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cap = ScreenCapture()
    saved = 0
    start_dt = time.time()
    next_save = time.time()

    print(f"Recording to {run_dir}")
    print(f"FPS={FPS} | WASD=move, numpad=shoot, Space=auto-aim, Shift=ult | ESC to stop")
    try:
        while True:
            img = cap.get_frame()
            if img is None:
                time.sleep(0.05)
                continue

            now = time.time()
            if now < next_save:
                time.sleep(0.02)
                continue
            next_save = now + 1.0 / FPS

            move, shoot, ult = read_action()
            if move == 8 and shoot == 9 and not ult:
                continue  # ничего не нажато — пропускаем пустые кадры

            fname = f"frame_{saved:05d}.jpg"
            cv2.imwrite(os.path.join(frames_dir, fname), img)
            header = not saved
            with open(os.path.join(run_dir, "actions.csv"), "a", newline="") as f:
                w = csv.writer(f)
                if header:
                    w.writerow(["filename", "move", "shoot", "ult"])
                w.writerow([fname, move, shoot, ult])
            saved += 1
            if saved % 100 == 0:
                print(f"[{time.time() - start_dt:7.1f}s] saved {saved} frames")

            if keyboard.is_pressed("esc"):
                break
    except KeyboardInterrupt:
        pass

    print(f"\nDone. {saved} frames -> {run_dir}")


if __name__ == "__main__":
    main()