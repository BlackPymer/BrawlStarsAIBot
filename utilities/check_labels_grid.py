import csv
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

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

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    fig.suptitle("Случайные 5 размеченных кадров — проверь HP!", fontsize=14)

    for ax, (filename, hp) in zip(axes, samples):
        img_path = IMAGES_DIR / filename
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax.imshow(img)
        ax.set_title(f"HP={hp}\n{filename}", fontsize=8, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
