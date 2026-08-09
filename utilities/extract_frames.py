import cv2
from pathlib import Path

VIDEO = Path("videos/Запись экрана 2026-07-07 121042.mp4")
OUT = Path("datasets/datasetV3/images/Train")
START_FRAME = 300
NUM_IMAGES = 306
INTERVAL = 6

OUT.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(VIDEO))

for i in range(NUM_IMAGES):
    frame_idx = START_FRAME + i * INTERVAL
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, img = cap.read()
    if not ret:
        break
    cv2.imwrite(str(OUT / f"frame_{frame_idx:06d}.jpg"), img)
    print(f"Saved {i+1}/{NUM_IMAGES} – frame {frame_idx}")

cap.release()
print(f"Done – {NUM_IMAGES} images saved to {OUT}")
