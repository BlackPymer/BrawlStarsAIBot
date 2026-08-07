import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import keyboard

from controller.screen_capture import ScreenCapture
from controller.keyboard_controller import direction_class
from hp_recognition.hp_recogniser import HP_PADDING
from hp_recognition.CRNN_hp_recogniser import CRNNHpRecogniser
from network.exceptions import PlayerNotFoundException
from network.network_engine import NetworkEngine
from objects_detection.object_detector import CLASS_NAMES
from objects_detection.yolo_object_detector import YoloObjectDetector
from ult_recognition.ult_classifier import UltClassifierRecogniser, extract_player_crop

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
        shoot = 8
    else:
        shoot = direction_class(shoot_keys)
    ult = keyboard.is_pressed(ULT_KEY)
    return (move if move is not None else 8), (shoot if shoot is not None else 9), int(ult)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    run = time.strftime("rec_%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUT_DIR, run)
    os.makedirs(run_dir, exist_ok=True)

    cap = ScreenCapture()
    detector = YoloObjectDetector()
    hp_recogniser = CRNNHpRecogniser()
    ult_recogniser = UltClassifierRecogniser()

    maps, hps, ults_inp = [], [], []
    moves, shoots, ults = [], [], []
    saved = 0
    skipped = 0
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
                continue  # ничего не нажато — пропускаем

            h, w = img.shape[:2]
            objects = detector.detect(img)

            hp_crops = []
            for obj in objects:
                for box in obj.boxes:
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    if CLASS_NAMES[cls_id] in ("player", "enemy"):
                        cx1 = max(0, x1 - HP_PADDING)
                        cy1 = max(0, y1 - HP_PADDING)
                        cx2 = min(w, x2 + HP_PADDING)
                        cy2 = min(h, y1 + 100)
                        crop = img[cy1:cy2, cx1:cx2]
                        if crop.size > 0:
                            hp_crops.append(crop)
            hp_texts = hp_recogniser.recognise(image=img, crops=hp_crops) if hp_crops else []
            hp_values = [int(t) if t.isdigit() else 0 for t in hp_texts]

            try:
                player_crop = extract_player_crop(img, objects)
                has_ult = ult_recogniser.recognise(player_crop)
                map_inp, player_hp_inp, ult_inp = NetworkEngine.build_state(
                    objects, hp_values, has_ult, w, h, "cpu")
            except PlayerNotFoundException:
                skipped += 1
                continue

            maps.append(map_inp[0].numpy())
            hps.append(player_hp_inp[0].numpy())
            ults_inp.append(ult_inp[0].numpy())
            moves.append(move)
            shoots.append(shoot)
            ults.append(ult)
            saved += 1
            if saved % 50 == 0:
                print(f"[{time.time() - start_dt:7.1f}s] saved {saved} samples (skipped {skipped})")

            if keyboard.is_pressed("esc"):
                break
    except KeyboardInterrupt:
        pass

    if saved == 0:
        print("Nothing recorded.")
        return

    np.savez(
        os.path.join(run_dir, "dataset.npz"),
        map_inp=np.stack(maps),
        player_hp=np.stack(hps),
        ult_inp=np.stack(ults_inp),
        move=np.array(moves),
        shoot=np.array(shoots),
        ult=np.array(ults),
    )
    print(f"\nDone. {saved} samples -> {run_dir}/dataset.npz")


if __name__ == "__main__":
    main()