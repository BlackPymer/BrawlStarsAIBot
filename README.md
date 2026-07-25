# Brawl Stars Bot — Player Network

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Overview

This project builds a neural-network-driven **Player Network** — an autonomous agent that perceives, decides, and acts within **Brawl Stars** in real time. The agent's "brain" (`PlayerNetwork`) is a single PyTorch module that fuses spatial awareness (convnet over the minimap/game frame), scalar state (HP), and binary flags (ultimate readiness) to produce continuous movement, discrete shoot direction, and ultimate activation. The perception pipeline supplies live object detections and HP readings; the platform layer (stub) closes the loop.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        GameEngine                                │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │ GameController│  │  ObjectDetector   │  │   HPRecogniser    │  │
│  │  (abstract)   │  │  YoloObjectDetect.│  │ CRNNHpRecogniser  │  │
│  │  platform     │─▶│  YOLO v26s        │─▶│  CRNN + BiLSTM    │  │
│  │  adapter      │  │  8 classes        │  │  CTC decoder      │  │
│  └──────┬───────┘  │  640px, conf=0.3   │  │  ~93.75% acc      │  │
│         │          └──────────────────┘  └─────────┬───────────┘  │
│         │                                         │               │
│  ┌──────▼──────────────────────────────────────────▼───────────┐  │
│  │                NetworkEngine                                │  │
│  │  ┌──────────────────────────────────────────────────────┐   │  │
│  │  │                 PlayerNetwork (nn.Module)             │   │  │
│  │  │                                                       │   │  │
│  │  │  map_layer  ──┐                                       │   │  │
│  │  │  (3→64 conv)  │                                       │   │  │
│  │  │  28×18 frame  ├── fc1 (2242→512→256) ──┬─ move_head  │   │  │
│  │  │               │                        │  (256→128→2, │   │  │
│  │  │  hp (scalar)  ├────────────────────────┘   Tanh)      │   │  │
│  │  │               │                        ├─ shoot_head  │   │  │
│  │  │  has_ult (bin)┘                        │  (256→128→64 │   │  │
│  │  │                                        │   →10, log_  │   │  │
│  │  │                                        │   softmax)   │   │  │
│  │  │                                        └─ ult_head    │   │  │
│  │  │                                           (256→128→1, │   │  │
│  │  │                                            Sigmoid)   │   │  │
│  │  └──────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### Modules

| Module | Path | Responsibility |
|--------|------|---------------|
| **GameEngine** | `src/game_engine.py` | Orchestrates the main loop: bullet timer → frame capture → detection → HP OCR → decision → action |
| **GameController** | `src/controller/game_controller.py` | Abstract platform adapter (Android ADB, emulator, etc.) — stub |
| **ObjectDetector** | `src/objects_detection/object_detector.py` | Abstract base; 8 classes: player, enemy, powercube, powercube-box, wall, water, bush, zone |
| **YoloObjectDetector** | `src/objects_detection/yolo_object_detector.py` | Concrete YOLO v26s inference at 640px, conf=0.3 |
| **HPRecogniser** | `src/hp_recognition/hp_recogniser.py` | Abstract base with `HP_PADDING=10` |
| **CRNNHpRecogniser** | `src/hp_recognition/CRNN_hp_recogniser.py` | CRNN (Conv→BiLSTM→CTC) reading HP digits from cropped bars |
| **NetworkEngine** | `src/network/network_engine.py` | Stub wrapper around `PlayerNetwork` — `make_action(objects, hps)` |
| **PlayerNetwork** | `src/network/player_network.py` | PyTorch `nn.Module` — the agent's decision network |

### Player Network Concept

The **Player Network** is the core idea: a single neural network that ingests the game state through multiple modalities and outputs a complete action in one forward pass.

**Inputs:**
- `x` — 28×18 RGB frame (game minimap or downscaled view), processed by 3 conv layers (3→32→64→64) with stride-2 downsampling, flattened to 2240-dim vector
- `hp` — scalar health points (0–10000+), concatenated with map features
- `has_ult` — binary flag, gates the ultimate head (ult forced to 0 when unavailable)

**Outputs:**
- `move` — 2D continuous vector in [-1, 1] via Tanh (movement direction)
- `shoot` — 10-class discrete distribution via log-softmax (shoot angle bins)
- `ult_prob` — scalar in [0, 1] via Sigmoid (ultimate activation probability)

This design decouples perception (YOLO + CRNN) from decision (PlayerNetwork), making it straightforward to train the network with reinforcement learning or imitation learning once the full loop is operational.

## Current Status

| Component | Status |
|-----------|--------|
| YOLO detection (8 classes, `best.pt`) | **Complete** — trained, weights on HF |
| HP OCR CRNN (`hp_crnn_best.pt`) | **Complete** — 93.75% validation accuracy |
| HP linear regression baseline | **Complete** — `hp_linear_best.pt` |
| Dataset generation (frames, HP crops) | **Complete** — multiple pipelines |
| Labeling tool (`label_tool.py`) | **Complete** — GUI with model pre-fill |
| Video test pipeline (`test.py`) | **Complete** — annotated output videos |
| **PlayerNetwork** architecture | **Implemented** — model class ready, untrained |
| **NetworkEngine** integration | **Stub** — `make_action` not wired |
| **GameController** platform adapter | **Stub** — no concrete backend |
| **GameEngine** end-to-end loop | **Partial** — perception runs, action loop pending |
| RL / training loop for PlayerNetwork | **Not started** |

## Weights & Dataset

Trained model weights and the dataset used for training are available on Hugging Face:

[https://huggingface.co/datasets/BlackPymer2/BrawlStars-gameplay/tree/main](https://huggingface.co/datasets/BlackPymer2/BrawlStars-gameplay/tree/main)

## HP Dataset

`health_dataset/` contains ~3257 annotated HP crops with the following files:
- `labels.csv` — ground truth labels (filename, hp)
- `labels_with_predictions.csv` — labels with model predictions for unlabeled entries
- `images/` — cropped HP regions from gameplay frames
