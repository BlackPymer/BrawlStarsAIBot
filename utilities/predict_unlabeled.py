from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utilities.train_hp import CRNN, HPDataset, preprocess, decode, DEVICE

DATASET_ROOT = Path("health_dataset")
MODEL_PATH = "hp_crnn_best.pt"
OUTPUT_CSV = "health_dataset/labels_with_predictions.csv"


def predict_with_confidence(model, img, return_probs=False):
    inp = preprocess(img)
    inp = torch.from_numpy(inp).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(inp).permute(1, 0, 2)
        probs = torch.softmax(logits, dim=-1)
        max_probs, pred_ids = probs.max(dim=-1)
        pred_ids = pred_ids.squeeze(1)

    pred = decode(pred_ids.tolist())
    confidence = max_probs.mean().item()
    return pred, confidence


def main():
    model = CRNN().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {MODEL_PATH}")

    lines = (DATASET_ROOT / "labels.csv").read_text().strip().splitlines()

    header = lines[0]
    updated = [header]
    annotated = []
    unannotated = []

    for line in lines[1:]:
        parts = line.strip().split(",", 1)
        fname = parts[0]
        hp = parts[1].strip() if len(parts) > 1 else ""
        if hp:
            updated.append(line)
            annotated.append(fname)
        else:
            unannotated.append(fname)

    print(f"Annotated: {len(annotated)}, Unannotated: {len(unannotated)}")

    img_dir = DATASET_ROOT / "images"
    correct = 0
    total = 0
    threshold = 0.7

    for fname in tqdm(unannotated, desc="Predicting", unit="img"):
        img_path = img_dir / fname
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARN: cannot load {img_path}, skipping")
            updated.append(f"{fname},")
            continue

        pred_hp, conf = predict_with_confidence(model, img)
        updated.append(f"{fname},{pred_hp}")

        if conf < threshold:
            print(f"  LOW CONF: {fname} -> pred={pred_hp}, conf={conf:.3f}")

    out_path = Path(OUTPUT_CSV)
    out_path.write_text("\n".join(updated) + "\n")
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
