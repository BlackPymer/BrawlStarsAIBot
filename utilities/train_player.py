import csv
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from network.exceptions import PlayerNotFoundException
from network.network_engine import NetworkEngine, MAX_HP_VALUE
from network.player_network import PlayerNetwork, INPUT_WIDTH, INPUT_HEIGHT
from objects_detection.object_detector import CLASS_NAMES
from objects_detection.yolo_object_detector import YoloObjectDetector
from hp_recognition.hp_recogniser import HP_PADDING
from hp_recognition.CRNN_hp_recogniser import CRNNHpRecogniser
from ult_recognition.ult_classifier import UltClassifierRecogniser

RECORDED_ROOT = BASE / "recorded_data"
MODEL_PATH = BASE / "src" / "network" / "player_network_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 60
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 42
TRAIN_SPLIT = 0.9


class ActionDataset(Dataset):
    def __init__(self, samples):
        self.map_inp = torch.stack([s["map"] for s in samples])
        self.player_hp = torch.stack([s["hp"] for s in samples])
        self.ult_inp = torch.stack([s["ult_inp"] for s in samples])
        self.move = torch.tensor([s["move"] for s in samples])
        self.shoot = torch.tensor([s["shoot"] for s in samples])
        self.ult = torch.tensor([s["ult"] for s in samples])

    def __len__(self):
        return len(self.move)

    def __getitem__(self, i):
        return (self.map_inp[i], self.player_hp[i], self.ult_inp[i],
                self.move[i], self.shoot[i], self.ult[i])


def build_samples():
    detector = YoloObjectDetector()
    hp_recogniser = CRNNHpRecogniser()
    ult_recogniser = UltClassifierRecogniser()
    samples = []
    skipped = 0

    rec_dirs = sorted([d for d in RECORDED_ROOT.iterdir() if d.is_dir()])
    if not rec_dirs:
        raise SystemExit(f"No recordings found in {RECORDED_ROOT}")

    for rec_dir in rec_dirs:
        csv_path = rec_dir / "actions.csv"
        frames_dir = rec_dir / "frames"
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        for row in tqdm(rows, desc=f"build {rec_dir.name}"):
            frame = cv2.imread(str(frames_dir / row["filename"]))
            if frame is None:
                skipped += 1
                continue
            h, w = frame.shape[:2]
            objects = detector.detect(frame)

            hp_crops = []
            for obj in objects:
                for box in obj.boxes:
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    if CLASS_NAMES[cls_id] in ("player", "enemy"):
                        cx1 = max(0, x1 - HP_PADDING)
                        cy1 = max(0, y1 - HP_PADDING)
                        cx2 = min(w, x2 + HP_PADDING)
                        cy2 = min(h, y1 + 100)
                        crop = frame[cy1:cy2, cx1:cx2]
                        if crop.size > 0:
                            hp_crops.append(crop)
            hp_texts = hp_recogniser.recognise(image=frame, crops=hp_crops) if hp_crops else []
            hp_values = [int(i) for i in hp_texts] if hp_texts else []

            try:
                has_ult = ult_recogniser.recognise(frame)
                map_inp, player_hp_inp, ult_inp = NetworkEngine.build_state(
                    objects, hp_values, has_ult, w, h, "cpu")
            except PlayerNotFoundException:
                skipped += 1
                continue
            samples.append({
                "map": map_inp[0],
                "hp": player_hp_inp[0],
                "ult_inp": ult_inp[0],
                "move": int(row["move"]),
                "shoot": int(row["shoot"]),
                "ult": int(row["ult"]),
            })
    print(f"samples: {len(samples)}, skipped: {skipped}")
    return samples


def class_weights(labels, num_classes):
    counts = np.bincount(labels.numpy(), minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        weights[c] = len(labels) / (num_classes * counts[c]) if counts[c] > 0 else 0.0
    return torch.tensor(weights, device=DEVICE)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    samples = build_samples()
    random.shuffle(samples)
    n_train = int(len(samples) * TRAIN_SPLIT)
    train_ds = ActionDataset(samples[:n_train])
    val_ds = ActionDataset(samples[n_train:])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = PlayerNetwork().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    move_w = class_weights(train_ds.move, 9)
    shoot_w = class_weights(train_ds.shoot, 10)
    loss_move = nn.NLLLoss(weight=move_w)
    loss_shoot = nn.NLLLoss(weight=shoot_w)
    loss_ult = nn.BCELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss, t_ml, t_sl, t_ul = 0.0, 0.0, 0.0, 0.0
        for map_inp, hp, ult_inp, move, shoot, ult in train_loader:
            map_inp = map_inp.to(DEVICE)
            hp = hp.to(DEVICE)
            ult_inp = ult_inp.to(DEVICE)
            move = move.to(DEVICE)
            shoot = shoot.to(DEVICE)
            ult = ult.float().to(DEVICE)

            m, s, u = model(map_inp, hp, ult_inp)
            l_m = loss_move(m, move)
            l_s = loss_shoot(s, shoot)
            l_u = loss_ult(u.squeeze(1), ult)
            loss = l_m + l_s + l_u
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = map_inp.size(0)
            t_loss += loss.item() * bs
            t_ml += l_m.item() * bs
            t_sl += l_s.item() * bs
            t_ul += l_u.item() * bs

        n = len(train_ds)
        model.eval()
        v_loss, v_ml, v_sl, v_ul = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for map_inp, hp, ult_inp, move, shoot, ult in val_loader:
                map_inp = map_inp.to(DEVICE)
                hp = hp.to(DEVICE)
                ult_inp = ult_inp.to(DEVICE)
                move = move.to(DEVICE)
                shoot = shoot.to(DEVICE)
                ult = ult.float().to(DEVICE)
                m, s, u = model(map_inp, hp, ult_inp)
                bs = map_inp.size(0)
                l_m = loss_move(m, move).item()
                l_s = loss_shoot(s, shoot).item()
                l_u = loss_ult(u.squeeze(1), ult).item()
                v_ml += l_m * bs
                v_sl += l_s * bs
                v_ul += l_u * bs
                v_loss += (l_m + l_s + l_u) * bs
        nv = len(val_ds)
        print(f"epoch {epoch:3d} | train loss {t_loss / n:7.4f} (m {t_ml / n:6.3f} s {t_sl / n:6.3f} u {t_ul / n:6.3f})"
              f" | val {v_loss / nv:7.4f} (m {v_ml / nv:6.3f} s {v_sl / nv:6.3f} u {v_ul / nv:6.3f})")

    torch.save(model.state_dict(), str(MODEL_PATH))
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()