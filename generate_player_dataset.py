from pathlib import Path
import csv

import cv2
from tqdm import tqdm
from ultralytics import YOLO

MODEL_PATH = "runs/train/v3/weights/best.pt"
IMAGES_DIR = Path("datasets/datasetV3/images/Train")
OUT_DIR = Path("health_dataset")
PADDING = 10
CONF_THRESHOLD = 0.3

OUT_IMAGES = OUT_DIR / "images"
OUT_LABELS = OUT_DIR / "labels.csv"

def load_existing_labels():
    if not OUT_LABELS.exists():
        return {}
    existing = {}
    with open(OUT_LABELS, newline="") as f:
        for row in csv.DictReader(f):
            hp = row["hp"].strip()
            if hp:
                existing[row["filename"]] = hp
    return existing


def main():
    model = YOLO(MODEL_PATH)
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)

    existing_labels = load_existing_labels()

    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))
    if not image_paths:
        print(f"No images found in {IMAGES_DIR}")
        return

    records = []
    for img_path in tqdm(image_paths, desc="Processing"):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        results = model.predict(frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)

        player_count = 0
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 = max(0, x1 - PADDING)
                y1 = max(0, y1 - PADDING)
                x2 = min(w, x2 + PADDING)
                y2 = min(h, y2 + PADDING)

                crop = frame[y1:y2, x1:x2]
                stem = img_path.stem
                crop_name = f"{stem}_player_{player_count}.jpg"
                cv2.imwrite(str(OUT_IMAGES / crop_name), crop)

                hp = existing_labels.get(crop_name, "")
                records.append({"filename": crop_name, "hp": hp})
                player_count += 1

    with open(OUT_LABELS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "hp"])
        writer.writeheader()
        writer.writerows(records)

    n_filled = sum(1 for r in records if r["hp"])
    print(f"Saved {len(records)} crops to {OUT_IMAGES}")
    print(f"Label file: {OUT_LABELS} — {n_filled}/{len(records)} hp filled")

if __name__ == "__main__":
    main()
