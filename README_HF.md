# Brawl Stars Gameplay Dataset

This dataset contains labeled and unlabeled footage of **Brawl Stars** gameplay, intended for training computer vision models.

## Contents

- **`best.pt`** — Trained YOLO model weights (yolo26s architecture). The model detects in-game objects such as brawlers, projectiles, and UI elements.
- **`video/`** — Unlabeled raw gameplay recordings used as source material to create the labeled dataset.
- **`dataset/`** — Annotated images and labels in YOLO format (if present).

## Demo

A video demonstrating the trained network in action is included in this repository.

## Usage

Download the weights and use them with Ultralytics YOLO:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict("screenshot.png")
```

## License

Apache 2.0
