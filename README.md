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

This project builds a neural-network-driven **Player Network** — an autonomous agent that perceives, decides, and acts within **Brawl Stars** in real time. The agent's "brain" (`PlayerNetwork`) is a single PyTorch module that fuses spatial awareness (convnet over the minimap/game frame), scalar state (HP), and binary flags (ultimate readiness) to produce movement, discrete shoot direction, and ultimate activation.

The agent plays on a **PC emulator (BlueStacks)**: the controller captures the screen via `mss`, detects objects with YOLO, reads HP bars with a CRNN OCR, recognises the ultimate indicator, and drives the game through a keyboard bind layer (movement keys + numpad shoot directions + space auto-attack + shift ultimate).

## ⚠️ IMPORTANT — CUDA and PyTorch

`requirements.txt` intentionally stays **cross-platform**: it does not pin a CUDA build of PyTorch, because the correct package (CPU vs CUDA wheel) depends on your OS, driver and GPU.

- **With an NVIDIA GPU (recommended):** install the CUDA build from the official PyTorch index, which **overrides** the plain `torch` from `requirements.txt`:

  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  ```

  Verify it took effect:

  ```bash
  python -c "import torch; print(torch.cuda.is_available())"   # must be True
  ```

  With CUDA active, the bot runs YOLO detection, HP OCR (CRNN) and ult classification **all on the GPU automatically** (the code uses `cuda if torch.cuda.is_available()`). On an RTX 4050 laptop, YOLO drops from ~106 ms to ~21 ms and HP OCR from ~12 ms to ~1.3 ms per frame vs CPU.

- **CPU-only / no NVIDIA GPU:** just `pip install -r requirements.txt` (CPU wheels of torch are pulled automatically).

> **IMPORTANT:** Do **not** use `onnxruntime-gpu` on Windows for these models — current GPU builds require CUDA 13 / cuDNN 9 and a newer driver than most setups have (driver-only CUDA 12.x is not enough). The code instead switches automatically: **CUDA available → torch (`.pt`), no CUDA → ONNX Runtime CPU fallback** (see `utilities/export_onnx.py` to build the `.onnx` models).

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        GameEngine                                │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │ GameController│  │  ObjectDetector   │  │   HPRecogniser    │  │
│  │  KeyboardCtrl │  │  YoloObjectDetect.│  │ CRNNHpRecogniser  │  │
│  │  mss capture  │─▶│  YOLO v26s        │─▶│  CRNN + BiLSTM    │  │
│  │  keyboard key │  │  8 classes        │  │  CTC decoder      │  │
│  └──────┬───────┘  │  640px, conf=0.3   │  │  ~93.75% acc      │  │
│         │          └──────────────────┘  └─────────┬───────────┘  │
│         │                                         │               │
│  ┌──────▼──────────────────────────────────────────▼───────────┐  │
│  │                NetworkEngine / RLEnv                         │  │
│  │  ┌──────────────────────────────────────────────────────┐   │  │
│  │  │                 PlayerNetwork (nn.Module)             │   │  │
│  │  │  map 28×18 ──conv──┐                                  │   │  │
│  │  │  hp (scalar)       ├── fc1 ──┬─ move_head (9 classes) │   │  │
│  │  │  has_ult (bin) ────┘         ├─ shoot_head (10)       │   │  │
│  │  │                             └─ ult_head (binary)      │   │  │
│  │  └──────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### Modules

| Module | Path | Responsibility |
|--------|------|---------------|
| **KeyboardController** | `src/controller/keyboard_controller.py` | Screen capture via `mss`, keyboard bind layer (WASD + numpad), battle/after-match clicks, match restart with full load wait |
| **ObjectDetector** | `src/objects_detection/object_detector.py` | Abstract base; 8 classes: player, enemy, powercube, powercube-box, wall, water, bush, zone |
| **YoloObjectDetector** | `src/objects_detection/yolo_object_detector.py` | Concrete YOLO v26s inference at 640px, conf=0.3 |
| **HPRecogniser** | `src/hp_recognition/hp_recogniser.py` | Abstract base with `HP_PADDING=10` |
| **CRNNHpRecogniser** | `src/hp_recognition/CRNN_hp_recogniser.py` | CRNN (Conv→BiLSTM→CTC) reading HP digits from cropped bars |
| **NetworkEngine** | `src/network/network_engine.py` | Builds the network state tensors (map / HP / ult) from detections |
| **PlayerNetwork** | `src/network/player_network.py` | PyTorch `nn.Module` — the agent's decision network |
| **RewardTracker** | `src/reward/reward_tracker.py` | Event-based reward attribution: hits, kills, power cubes, box destruction, misses, zone damage — with shot-direction attribution and missing-frame patience |
| **RLEnv** | `src/rl/env.py` | OpenAI-gym-style wrapper: applies an action, captures the next frame, computes reward and episode termination |

### Player Network Concept

The **Player Network** is the core idea: a single neural network that ingests the game state through multiple modalities and outputs a complete action in one forward pass.

**Inputs:**
- `map` — 28×18 spatial grid (objects on the map), processed by conv layers
- `hp` — scalar health points (0–10000+), concatenated with map features
- `has_ult` — binary flag, gates the ultimate head (ult forced to 0 when unavailable)

**Outputs:**
- `move` — 9-class discrete distribution (8 compass directions + stop) via log-softmax
- `shoot` — 10-class discrete distribution (8 directions + auto-attack + no shot)
- `ult_prob` — scalar in [0, 1] via Sigmoid (ultimate activation probability)

## Training Pipeline

```
record.py                     # Records live gameplay: frames + HP crops + actions to recorded_data/
utilities/
├── convert_recordings.py     # Converts recordings into ready-to-train network inputs (dataset.npz)
├── train_player.py           # Supervised imitation learning (cross-entropy on move/shoot/ult)
│                             #   with augmentation (mirroring, shifting) and class weights
└── train_rl.py               # RL (PPO-lite) on the live emulator via RLEnv
```

### Supervised pretraining (`train_player.py`)

Trains `PlayerNetwork` from recorded human/scripted gameplay:

```bash
python utilities/train_player.py
```

- Reads `recorded_data/*/dataset.npz` (only recordings converted with `convert_recordings.py` are used)
- Augmentation: horizontal / vertical / 180° mirroring (labels mirrored accordingly) + 1-px shifts
- Class weights compensate for the imbalanced move/shoot class distributions
- Best validation checkpoint is saved to `src/network/player_network_best.pt`

### RL fine-tuning (`train_rl.py`)

Improves the supervised policy with on-policy PPO-lite directly against the live game on the emulator:

```bash
python utilities/train_rl.py     # or python main.py (MODE="rl")
```

- Loads `src/network/player_network_best.pt` as the starting policy
- `RLEnv` wraps **KeyboardController** — every step executes move + shoot + ult on the game, then reads the next screen frame (≈5 fps effective)
- **Rewards**: hit 0.4, kill 0.7, power cube 0.5, box destroyed 0.3, miss −0.03, zone damage −0.2, win (3+ kills per match) +1.0, death −1.0
- **Episode termination**: the player leaving the frame for `MATCH_END_GRACE_FRAMES` — **win = 3+ kills in the match**, anything else counts as a loss
- After each match the controller clicks the battle / after-match buttons and waits the full map load (`MATCH_LOAD_WAIT = 13s`)
- Every improvement over the best episode reward is saved back to `player_network_best.pt`; the model is also saved on exit

## Current Status

| Component | Status |
|-----------|--------|
| YOLO detection (8 classes, `best.pt`) | **Complete** — trained, weights on HF |
| HP OCR CRNN (`hp_crnn_best.pt`) | **Complete** — 93.75% validation accuracy |
| HP linear regression baseline | **Complete** — `hp_linear_best.pt` |
| Dataset generation (frames, HP crops) | **Complete** — multiple pipelines |
| Labeling tool (`label_tool.py`) | **Complete** — GUI with model pre-fill |
| Video test pipeline (`test.py`) | **Complete** — annotated output videos |
| **PlayerNetwork** architecture | **Complete** — 9-class move / 10-class shoot / binary ult |
| **Supervised training** (`train_player.py`) | **Complete** — best val loss ≈ 2.68, weights in `src/network/player_network_best.pt` |
| **KeyboardController** (BlueStacks PC, mss capture) | **Complete** — movement, shooting, ult, battle clicks, match restart |
| **RewardTracker** / **RLEnv** | **Complete** — full event-based reward attribution + gym-style wrapper |
| **RL fine-tuning** (`train_rl.py`) | **In progress** — on-live PPO-lite runs against the emulator |

## Weights & Dataset

Trained model weights and the dataset used for training are available on Hugging Face:

[https://huggingface.co/datasets/BlackPymer2/BrawlStars-gameplay/tree/main](https://huggingface.co/datasets/BlackPymer2/BrawlStars-gameplay/tree/main)

## HP Dataset

`health_dataset/` contains ~3257 annotated HP crops with the following files:
- `labels.csv` — ground truth labels (filename, hp)
- `labels_with_predictions.csv` — labels with model predictions for unlabeled entries
- `images/` — cropped HP regions from gameplay frames
