# Brawl Stars Bot

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

This project aims to create a neural network-based bot for playing **Brawl Stars**. The bot uses computer vision to perceive the game state and make decisions in real-time.

## Current Status

- **YOLO detection model** (`yolo26s.pt`) — recognizes game elements (characters, items, UI) from screenshots.
- **HP OCR model** (`hp_crnn_best.pt`) — reads health values from player/enemy crops with ~93.75% accuracy.
- **Labeling tool** (`label_tool.py`) — GUI tool for manual annotation of the HP dataset with model predictions as defaults.

## Weights & Dataset

The trained model weights and the dataset used for training are available on Hugging Face:

[https://huggingface.co/datasets/BlackPymer2/BrawlStars-gameplay/tree/main](https://huggingface.co/datasets/BlackPymer2/BrawlStars-gameplay/tree/main)

## HP Dataset

`health_dataset/` contains ~3257 annotated HP crops with the following files:
- `labels.csv` — ground truth labels (filename, hp)
- `labels_with_predictions.csv` — labels with model predictions for unlabeled entries
- `images/` — cropped HP regions from gameplay frames
