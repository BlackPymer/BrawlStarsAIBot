from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

VIDEO = "videos/Запись экрана 2026-07-07 122233.mp4"
MODEL = "runs/train/v3/weights/best.pt"
OUT_DIR = Path("runs/detect/video_test")
SKIP_FRAMES = 2  # process every 3rd frame → 10 FPS

NAMES = {
    0: "player",
    1: "enemy",
    2: "powercube",
    3: "powercube-box",
    4: "wall",
    5: "water",
    6: "bush",
    7: "zone",
}

COLORS = {
    0: (0, 255, 0),    # player — green
    1: (0, 0, 255),    # enemy — red
    2: (255, 255, 0),  # powercube — cyan
    3: (255, 0, 255),  # powercube-box — magenta
    4: (128, 128, 0),  # wall — olive
    5: (255, 0, 0),    # water — blue
    6: (0, 128, 0),    # bush — dark green
    7: (128, 128, 128),# zone — grey
}


def main():
    model = YOLO(MODEL)
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"Failed to open {VIDEO}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = fps / (SKIP_FRAMES + 1)
    out_w, out_h = 1280, 720

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video_stem = Path(VIDEO).stem
    out_path = str(OUT_DIR / f"{video_stem}_detected.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))

    print(f"Processing {total_frames} frames @ {fps:.0f} FPS → {out_fps:.0f} FPS output")
    print(f"Output: {out_path}")

    frame_idx = 0
    processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % (SKIP_FRAMES + 1) == 0:
            results = model.predict(frame, imgsz=640, conf=0.3, verbose=False)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    color = COLORS.get(cls_id, (255, 255, 255))
                    label = f"{NAMES.get(cls_id, str(cls_id))} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            resized = cv2.resize(frame, (out_w, out_h))
            writer.write(resized)
            processed += 1

            if processed % 500 == 0:
                print(f"  Processed {processed} frames ({frame_idx}/{total_frames})")

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"\nDone — {processed} frames written to {out_path}")


if __name__ == "__main__":
    main()
