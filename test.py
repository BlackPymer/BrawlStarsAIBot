from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
import time as t

from train_hp import CRNN, preprocess, decode

VIDEO = "videos/Запись экрана 2026-07-07 122233.mp4"
MODEL = "runs/train/v3/weights/best.pt"
OUT_DIR = Path("runs/detect/video_test")
SKIP_FRAMES = 2

HP_MODEL_PATH = "hp_crnn_best.pt"
HP_PADDING = 10
HP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    0: (0, 255, 0),
    1: (0, 0, 255),
    2: (255, 255, 0),
    3: (255, 0, 255),
    4: (128, 128, 0),
    5: (255, 0, 0),
    6: (0, 128, 0),
    7: (128, 128, 128),
}


def load_hp_model():
    model = CRNN()
    model.load_state_dict(torch.load(HP_MODEL_PATH, map_location=HP_DEVICE))
    model.to(HP_DEVICE)
    model.eval()
    return model


def preprocess_hp_batch(crops, target_h=48):
    tensors = []
    for crop in crops:
        tensor = torch.from_numpy(preprocess(crop, target_h))
        tensors.append(tensor)

    max_w = max(t.shape[2] for t in tensors)
    padded = []
    for t in tensors:
        w = t.shape[2]
        if w < max_w:
            pad = torch.zeros((1, 48, max_w - w), dtype=torch.float32)
            t = torch.cat([t, pad], dim=2)
        padded.append(t)
    return torch.stack(padded).to(HP_DEVICE)


def predict_hp_batch(model, crops):
    if not crops:
        return []
    batch = preprocess_hp_batch(crops)
    with torch.no_grad():
        logits = model(batch).permute(1, 0, 2)
    pred_ids = logits.argmax(dim=2).permute(1, 0)
    results = []
    for b in range(len(crops)):
        hp_text = decode(pred_ids[b].tolist())
        results.append(hp_text)
    return results


def main():
    det_model = YOLO(MODEL)
    hp_model = load_hp_model()

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
    time = t.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % (SKIP_FRAMES + 1) == 0:
            results = det_model.predict(frame, imgsz=640, conf=0.3, verbose=False)

            hp_crops = []
            hp_boxes = []
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

                    if cls_id in (0, 1):
                        h, w = frame.shape[:2]
                        cx1 = max(0, x1 - HP_PADDING)
                        cy1 = max(0, y1 - HP_PADDING)
                        cx2 = min(w, x2 + HP_PADDING)
                        cy2 = min(h, y1 + 100)
                        crop = frame[cy1:cy2, cx1:cx2]
                        if crop.size > 0:
                            hp_crops.append(crop)
                            hp_boxes.append((x1, y2))

            if hp_crops:
                hp_texts = predict_hp_batch(hp_model, hp_crops)
                for (x1, y2), hp_text in zip(hp_boxes, hp_texts):
                    if hp_text:
                        cv2.putText(frame, f"HP: {hp_text}", (x1, y2 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            resized = cv2.resize(frame, (out_w, out_h))
            writer.write(resized)
            processed += 1

            if processed % 500 == 0:
                print(f"  Processed {processed} frames ({frame_idx}/{total_frames})\t{500/(t.time() - time)}FPS")
                time = t.time()

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"\nDone — {processed} frames written to {out_path}")


if __name__ == "__main__":
    print(HP_DEVICE)
    main()
