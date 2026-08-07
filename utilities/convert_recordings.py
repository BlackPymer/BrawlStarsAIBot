import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from network.exceptions import PlayerNotFoundException
from network.network_engine import NetworkEngine
from objects_detection.object_detector import CLASS_NAMES
from objects_detection.yolo_object_detector import YoloObjectDetector
from hp_recognition.hp_recogniser import HP_PADDING
from hp_recognition.CRNN_hp_recogniser import CRNNHpRecogniser
from ult_recognition.ult_classifier import UltClassifierRecogniser, extract_player_crop

RECORDED_ROOT = BASE / "recorded_data"


def convert_recording(rec_dir: Path):
    csv_path = rec_dir / "actions.csv"
    frames_dir = rec_dir / "frames"
    if not csv_path.exists() or not frames_dir.is_dir():
        return None
    out_path = rec_dir / "dataset.npz"
    if out_path.exists():
        print(f"[ok] already converted: {out_path}")
        return

    detector = YoloObjectDetector()
    hp_recogniser = CRNNHpRecogniser()
    ult_recogniser = UltClassifierRecogniser()

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    maps, hps, ults_inp = [], [], []
    moves, shoots, ults = [], [], []
    skipped = 0

    for row in tqdm(rows, desc=f"convert {rec_dir.name}"):
        frame = cv2.imread(str(frames_dir / row["filename"]))
        if frame is None:
            skipped += 1
            continue
        h, w = frame.shape[:2]
        objects = detector.detect(frame)

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
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.size > 0:
                        hp_crops.append(crop)
        hp_texts = hp_recogniser.recognise(image=frame, crops=hp_crops) if hp_crops else []
        hp_values = [int(t) if t.isdigit() else 0 for t in hp_texts]

        try:
            player_crop = extract_player_crop(frame, objects)
            has_ult = ult_recogniser.recognise(player_crop)
            map_inp, player_hp_inp, ult_inp = NetworkEngine.build_state(
                objects, hp_values, has_ult, w, h, "cpu")
        except PlayerNotFoundException:
            skipped += 1
            continue

        maps.append(map_inp[0].numpy())
        hps.append(player_hp_inp[0].numpy())
        ults_inp.append(ult_inp[0].numpy())
        moves.append(int(row["move"]))
        shoots.append(int(row["shoot"]))
        ults.append(int(row["ult"]))

    if not maps:
        print(f"[skip] no valid samples in {rec_dir.name} (skipped {skipped})")
        return

    np.savez(
        out_path,
        map_inp=np.stack(maps),
        player_hp=np.stack(hps),
        ult_inp=np.stack(ults_inp),
        move=np.array(moves),
        shoot=np.array(shoots),
        ult=np.array(ults),
    )
    print(f"[ok] {len(maps)} samples, skipped {skipped} -> {out_path}")


def main():
    rec_dirs = sorted([d for d in RECORDED_ROOT.iterdir() if d.is_dir()])
    if not rec_dirs:
        raise SystemExit(f"No recordings found in {RECORDED_ROOT}")
    for rec_dir in rec_dirs:
        convert_recording(rec_dir)


if __name__ == "__main__":
    main()