import csv
import random
import sys
from pathlib import Path

import cv2

IMAGES_DIR = Path("health_dataset/images")
LABELS_CSV = Path("health_dataset/labels.csv")


def main():
    labeled = []
    with open(LABELS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            hp = row["hp"].strip()
            if hp:
                labeled.append((row["filename"], int(hp)))

    if not labeled:
        print("Нет размеченных записей!")
        return

    samples = random.sample(labeled, min(5, len(labeled)))

    for filename, hp in samples:
        img_path = IMAGES_DIR / filename
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Не удалось загрузить {filename}")
            continue

        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)

        label = f"HP: {hp}"
        cv2.putText(img, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Check label", img)
        print(f"[{filename}] -> HP = {hp}. Нажми любую клавишу для продолжения...")
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == 27:
            break


if __name__ == "__main__":
    main()
