# Brawl Stars Gameplay Dataset & Models

This repository contains labeled gameplay footage of **Brawl Stars**, along with trained computer vision models for detection and HP (health points) OCR.

## Contents

### YOLO Detection Model

- **`best.pt`** — Trained YOLO model weights (yolo26s architecture). Detects: player, enemy, powercube, powercube-box, wall, water, bush, zone.

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict("screenshot.png")
```

### HP OCR Dataset

`health_dataset/` — A labeled dataset of **3259 cropped HP regions** extracted from gameplay frames:

- `images/` — 100×115 px cropped HP bars (3-channel)
- `labels.csv` — Ground truth HP values (regression targets)
- `labels_with_predictions.csv` — Labels augmented with model predictions for unlabelled entries

Each filename encodes metadata: `{source}_frame{number}_{player|enemy}_{instance_id}.jpg`

#### Distribution

| HP range | Labeled samples | Common values |
|----------|----------------|---------------|
| 0 (dead) | 176 | 0 |
| 4800–6800 | 230 | 4800, 5189, 5316, 5716, 5846, 6260, 6480, 6800 |
| 7200–8600 | 367 | 7200, 7600, 7649, 7800, 8076, 8173, 8200, 8476, 8600 |
| 9000–11400 | 248 | 9000, 9400, 9800, 10200, 10600, 10630, 11000, 11200 |
| 12600–13000 | 141 | 12600, 13000 |
| Other | 295 | Mixed partial-HP values |
| Unlabelled | 1800 | — |

### HP OCR Model

- **`hp_crnn_best.pt`** — CRNN (CNN + BiLSTM + CTC) for reading HP values from cropped health bars.
- **Accuracy: 93.75%** on the validation split.

![Training stats](accuracy.png)

#### Loading & Inference

```python
import cv2
import torch
from train_hp import CRNN, preprocess, decode

model = CRNN()
model.load_state_dict(torch.load("hp_crnn_best.pt", map_location="cpu"))
model.eval()

# Load a cropped HP region
img = cv2.imread("health_dataset/images/frame_000306_player_0.jpg")

# Preprocess (grayscale, resize to H=48, normalize)
inp = torch.from_numpy(preprocess(img)).unsqueeze(0)

# Inference
with torch.no_grad():
    logits = model(inp).permute(1, 0, 2)
pred_ids = logits.argmax(dim=2).permute(1, 0)[0]

hp_text = decode(pred_ids.tolist())
print(f"HP: {hp_text}")
```

#### Training

The model is a lightweight CRNN trained via CTC loss on the HP dataset. See `train_hp.py` for the full training pipeline.

## Video Demo

A video demonstrating the trained network in action is included in this repository.

## License

Apache 2.0
