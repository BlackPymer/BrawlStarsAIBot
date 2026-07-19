import csv
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

MODEL_PATH = "runs/train/v3/weights/best.pt"
VIDEO_PATH = "videos/Запись экрана 2026-07-07 121042.mp4"
OUT_DIR = Path("health_dataset")
PADDING = 10
CONF_THRESHOLD = 0.3
SKIP_FRAMES = 5

OUT_IMAGES = OUT_DIR / "images"
OUT_LABELS = OUT_DIR / "labels.csv"


def load_existing():
    if not OUT_LABELS.exists():
        return set(), {}
    existing_names = set()
    existing_labels = {}
    with open(OUT_LABELS, newline="") as f:
        for row in csv.DictReader(f):
            existing_names.add(row["filename"])
            hp = row["hp"].strip()
            if hp:
                existing_labels[row["filename"]] = hp
    return existing_names, existing_labels


def main():
    model = YOLO(MODEL_PATH)
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Failed to open {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_stem = Path(VIDEO_PATH).stem

    existing_names, existing_labels = load_existing()
    records = []
    new_count = 0

    player_count = 0
    enemy_count = 0
    frame_idx = 0

    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % (SKIP_FRAMES + 1) == 0:
                h, w = frame.shape[:2]
                results = model.predict(frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)

                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        is_player = cls_id == 0
                        is_enemy = cls_id == 1
                        if not (is_player or is_enemy):
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        x1 = max(0, x1 - PADDING)
                        y1 = max(0, y1 - PADDING)
                        x2 = min(w, x2 + PADDING)
                        y2 = min(h, y1 + 100)

                        crop = frame[y1:y2, x1:x2]

                        if crop.size == 0:
                            continue

                        if is_player:
                            crop_name = f"{video_stem}_frame{frame_idx}_player_{player_count}.jpg"
                            player_count += 1
                        else:
                            crop_name = f"{video_stem}_frame{frame_idx}_enemy_{enemy_count}.jpg"
                            enemy_count += 1

                        if crop_name in existing_names:
                            hp = existing_labels.get(crop_name, "")
                            records.append({"filename": crop_name, "hp": hp})
                            continue

                        cv2.imwrite(str(OUT_IMAGES / crop_name), crop)
                        records.append({"filename": crop_name, "hp": ""})
                        new_count += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()

    if records:
        existing_records = []
        if OUT_LABELS.exists():
            with open(OUT_LABELS, newline="") as f:
                for row in csv.DictReader(f):
                    existing_records.append(row)

        all_records = existing_records + records

        with open(OUT_LABELS, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "hp"])
            writer.writeheader()
            writer.writerows(all_records)

    n_filled = sum(1 for r in all_records if r["hp"])
    print(f"Added {new_count} new crops ({len(records)} total new records)")
    print(f"Label file: {OUT_LABELS} — {n_filled}/{len(all_records)} hp filled")


if __name__ == "__main__":
    main()
