import random
from pathlib import Path

import cv2
import yaml

DS_DIR = Path("datasets/datasetV2")
IMG_DIR = DS_DIR / "images" / "train"
LBL_DIR = DS_DIR / "labels" / "train"
YAML_PATH = DS_DIR / "data.yaml"
NUM_SAMPLES = 10
OUT_DIR = Path("runs/annotation_check")


def load_classes(yaml_path):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", {})
    return [names[i] for i in range(len(names))]


def draw_yolo_bboxes(img_path, lbl_path, class_names):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    if not lbl_path.exists():
        return img

    for line in lbl_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = class_names[cls] if cls < len(class_names) else str(cls)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def main():
    if not YAML_PATH.exists():
        print(f"data.yaml not found at {YAML_PATH}")
        return
    if not IMG_DIR.exists():
        print(f"Image directory not found at {IMG_DIR}")
        return

    class_names = load_classes(YAML_PATH)
    print(f"Loaded {len(class_names)} classes: {class_names}")
    print(f"Total images: {len(list(IMG_DIR.iterdir()))}")
    print(f"Total labels: {len(list(LBL_DIR.glob('*.txt')))}")

    label_files = sorted(LBL_DIR.glob("*.txt"))
    if not label_files:
        print(f"No labels found in {LBL_DIR}")
        return

    valid = []
    for lbl in label_files:
        for ext in [".jpg", ".png", ".jpeg"]:
            if (IMG_DIR / lbl.with_suffix(ext).name).exists():
                valid.append(lbl)
                break

    if not valid:
        print(f"No matching images found for any labels in {IMG_DIR}")
        return

    samples = random.sample(valid, min(NUM_SAMPLES, len(valid)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for lbl_path in samples:
        candidates = [
            IMG_DIR / lbl_path.with_suffix(ext).name
            for ext in [".jpg", ".png", ".jpeg"]
        ]
        img_path = next((p for p in candidates if p.exists()), None)
        if img_path is None:
            print(f"Image not found for {lbl_path.name}, skipping")
            continue

        result = draw_yolo_bboxes(img_path, lbl_path, class_names)
        if result is not None:
            out_path = OUT_DIR / f"check_{lbl_path.stem}.jpg"
            cv2.imwrite(str(out_path), result)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
