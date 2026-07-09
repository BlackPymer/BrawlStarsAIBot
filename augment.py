import random
from pathlib import Path

import cv2
import albumentations as A
import numpy as np

BBOX_PARAMS = {
    "format": "yolo",
    "label_fields": ["class_labels"],
    "clip": True,
    "filter_invalid_bboxes": True,
    "min_visibility": 0.05,
}

TRAIN_TRANSFORM = A.Compose(
    [
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.15, p=0.7
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=20, p=0.7
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.1, 0.1),
            rotate=(-10, 10),
            p=0.8,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(0.02, 0.08),
            hole_width_range=(0.02, 0.08),
            fill=0,
            p=0.3,
        ),
    ],
    bbox_params=A.BboxParams(**BBOX_PARAMS),
)


def read_yolo_labels(lbl_path):
    boxes = []
    class_labels = []
    if not lbl_path.exists():
        return boxes, class_labels
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
        boxes.append([cx, cy, bw, bh])
        class_labels.append(cls_id)
    return boxes, class_labels


def augment_image(img, boxes, class_labels, transform=TRAIN_TRANSFORM):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed = transform(
        image=img_rgb,
        bboxes=boxes,
        class_labels=class_labels,
    )
    aug_img = cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR)
    return aug_img, transformed["bboxes"], transformed["class_labels"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preview augmentations on random samples"
    )
    parser.add_argument("--img-dir", default="datasets/datasetV2/images/train")
    parser.add_argument("--lbl-dir", default="datasets/datasetV2/labels/train")
    parser.add_argument("--out-dir", default="runs/aug_preview")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    lbl_dir = Path(args.lbl_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(img_dir.iterdir())
    random.seed(args.seed)
    samples = random.sample(images, min(args.num_samples, len(images)))

    for img_path in samples:
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        boxes, class_labels = read_yolo_labels(lbl_path)
        if not boxes:
            continue

        aug_img, aug_boxes, aug_labels = augment_image(
            img, boxes, class_labels, TRAIN_TRANSFORM
        )
        out_path = out_dir / f"{stem}_aug.jpg"
        cv2.imwrite(str(out_path), aug_img)
        print(
            f"Saved {out_path} "
            f"({len(boxes)} -> {len(aug_boxes)} boxes)"
        )
