import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from network.player_network import PlayerNetwork

RECORDED_ROOT = BASE / "recorded_data"
MODEL_PATH = BASE / "src" / "network" / "player_network_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 42
TRAIN_SPLIT = 0.9


# индексы направлений move/shoot (0..7 = компасы N, NE, E, SE, S, SW, W, NW)
MIRROR_H = [0, 7, 6, 5, 4, 3, 2, 1]  # отражение влево-вправо
MIRROR_V = [4, 3, 2, 1, 0, 7, 6, 5]  # отражение вверх-вниз
MIRROR_180 = [4, 5, 6, 7, 0, 1, 2, 3]  # поворот на 180


def transform_labels(shoot, move, mirror):
    def _t(idx, table):
        return table[idx] if idx < len(table) else idx

    if mirror == "h":
        return _t(shoot, MIRROR_H), _t(move, MIRROR_H)
    if mirror == "v":
        return _t(shoot, MIRROR_V), _t(move, MIRROR_V)
    if mirror == "180":
        return _t(shoot, MIRROR_180), _t(move, MIRROR_180)
    return shoot, move


def flip_map(map_inp, mirror):
    if mirror == "h":
        return torch.flip(map_inp, dims=[-1])
    if mirror == "v":
        return torch.flip(map_inp, dims=[-2])
    if mirror == "180":
        return torch.flip(map_inp, dims=[-2, -1])
    return map_inp


def shift_map(map_inp, dx, dy):
    h, w = map_inp.shape[-2], map_inp.shape[-1]
    out = torch.zeros_like(map_inp)
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    dst_y0, dst_y1 = src_y0 + dy, src_y1 + dy
    dst_x0, dst_x1 = src_x0 + dx, src_x1 + dx
    if map_inp.dim() == 4:
        out[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = map_inp[:, :, src_y0:src_y1, src_x0:src_x1]
    else:
        out[:, dst_y0:dst_y1, dst_x0:dst_x1] = map_inp[:, src_y0:src_y1, src_x0:src_x1]
    return out


class ActionDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.map_inp = torch.stack([s["map"] for s in samples])
        self.player_hp = torch.stack([s["hp"] for s in samples])
        self.ult_inp = torch.stack([s["ult_inp"] for s in samples])
        self.move = torch.tensor([s["move"] for s in samples])
        self.shoot = torch.tensor([s["shoot"] for s in samples])
        self.ult = torch.tensor([s["ult"] for s in samples])
        self.augment = augment

    def __len__(self):
        return len(self.move)

    def __getitem__(self, i):
        map_inp = self.map_inp[i]
        move, shoot, ult = self.move[i], self.shoot[i], self.ult[i]
        if self.augment:
            mirror = random.choice(("h", "v", "180", None))
            if mirror is not None:
                map_inp = flip_map(map_inp, mirror)
                shoot, move = transform_labels(int(shoot), int(move), mirror)
                shoot = torch.tensor(shoot)
                move = torch.tensor(move)
            if random.random() < 0.5:
                map_inp = shift_map(map_inp, random.randint(-1, 1), random.randint(-1, 1))
        return map_inp, self.player_hp[i], self.ult_inp[i], move, shoot, ult


def load_samples():
    samples = []
    skipped = 0

    rec_dirs = sorted([d for d in RECORDED_ROOT.iterdir() if d.is_dir()])
    if not rec_dirs:
        raise SystemExit(f"No recordings found in {RECORDED_ROOT}")

    for rec_dir in rec_dirs:
        npz_path = rec_dir / "dataset.npz"
        if not npz_path.exists():
            print(f"[skip] no dataset.npz in {rec_dir.name} (run utilities/convert_recordings.py)")
            skipped += 1
            continue
        data = np.load(npz_path)
        map_inp = data["map_inp"]
        player_hp = data["player_hp"]
        ult_inp = data["ult_inp"]
        move = data["move"]
        shoot = data["shoot"]
        ult = data["ult"]
        n = len(move)
        for i in range(n):
            samples.append({
                "map": torch.from_numpy(map_inp[i]),
                "hp": torch.from_numpy(player_hp[i]),
                "ult_inp": torch.from_numpy(ult_inp[i]),
                "move": int(move[i]),
                "shoot": int(shoot[i]),
                "ult": int(ult[i]),
            })
        print(f"[ok] {rec_dir.name}: {n} samples")
    print(f"samples: {len(samples)}, recordings without npz: {skipped}")
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

    samples = load_samples()
    random.shuffle(samples)
    n_train = int(len(samples) * TRAIN_SPLIT)
    train_ds = ActionDataset(samples[:n_train], augment=True)
    val_ds = ActionDataset(samples[n_train:])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = PlayerNetwork().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_val_loss = float("inf")
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
        val_loss = v_loss / nv
        print(f"epoch {epoch:3d} | train loss {t_loss / n:7.4f} (m {t_ml / n:6.3f} s {t_sl / n:6.3f} u {t_ul / n:6.3f})"
              f" | val {val_loss:7.4f} (m {v_ml / nv:6.3f} s {v_sl / nv:6.3f} u {v_ul / nv:6.3f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(MODEL_PATH))
            print(f"  best val loss -> saved {MODEL_PATH}")

    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()