import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DATASET_ROOT = Path("health_dataset")

BATCH_SIZE = 16
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHARS = "0123456789"
BLANK_IDX = 0


class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        self.lstm = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 11),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.classifier(x)
        return x


def encode_hp(hp):
    return [CHARS.index(c) + 1 for c in str(int(hp))]


def decode(ids):
    prev = None
    result = []
    for i in ids:
        if i != prev and i != BLANK_IDX:
            idx = i - 1
            if 0 <= idx < len(CHARS):
                result.append(CHARS[idx])
        prev = i
    return "".join(result)


def decode_gt(ids):
    return "".join(CHARS[i - 1] for i in ids if 0 <= i - 1 < len(CHARS))


def preprocess(img, target_h=48):
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    h, w = gray.shape[:2]
    scale = target_h / h
    new_w = max(int(w * scale), target_h)
    resized = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
    resized = resized.astype(np.float32) / 255.0
    resized = (resized - 0.5) / 0.5
    resized = resized[np.newaxis, :, :]
    return resized


def elastic_transform(img, alpha=20, sigma=3):
    h, w = img.shape[:2]
    dx = np.random.randn(h, w).astype(np.float32) * sigma
    dy = np.random.randn(h, w).astype(np.float32) * sigma
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def cutout(img, size=8):
    h, w = img.shape[:2]
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img[y:y+size, x:x+size] = 128
    return img


def augment(img):
    aug = img.astype(np.float32)

    if random.random() < 0.5:
        alpha = random.uniform(0.7, 1.3)
        beta = random.uniform(-30, 30)
        aug = aug * alpha + beta

    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)

    if random.random() < 0.2:
        noise = np.random.randn(*aug.shape).astype(np.float32) * random.uniform(5, 20)
        aug = aug + noise

    if random.random() < 0.3:
        h, w = aug.shape[:2]
        angle = random.uniform(-3, 3)
        scale = random.uniform(0.95, 1.05)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if random.random() < 0.2:
        h, w = aug.shape[:2]
        x_off = random.randint(0, w // 4)
        y_off = random.randint(0, h // 4)
        cw = random.randint(w // 4, w // 2)
        ch = random.randint(h // 4, h // 2)
        aug[y_off:y_off+ch, x_off:x_off+cw] = random.uniform(100, 150)

    if random.random() < 0.2:
        sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        aug = cv2.filter2D(aug, -1, sharpen)

    return np.clip(aug, 0, 255)


class HPDataset(Dataset):
    def __init__(self, root, samples=None, augment=False):
        self.root = Path(root)
        self.augment = augment

        if samples is None:
            labels_path = self.root / "labels.csv"
            lines = labels_path.read_text().strip().splitlines()
            self.samples = []
            for line in lines[1:]:
                parts = line.strip().split(",")
                if len(parts) != 2:
                    continue
                fname, hp = parts
                hp = hp.strip()
                if not hp:
                    continue
                img_path = self.root / "images" / fname
                if img_path.exists():
                    self.samples.append((str(img_path), int(hp)))
        else:
            self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, hp = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load {img_path}")

        if self.augment:
            img = augment(img).astype(np.uint8)

        inp = preprocess(img)
        target = torch.tensor(encode_hp(hp), dtype=torch.long)
        return inp, target


def collate_fn(batch):
    images, targets = zip(*batch)
    max_w = max(im.shape[2] for im in images)
    padded = []
    for im in images:
        w = im.shape[2]
        if w < max_w:
            pad = np.zeros((1, 48, max_w - w), dtype=np.float32)
            im = np.concatenate([im, pad], axis=2)
        padded.append(torch.from_numpy(im))
    images = torch.stack(padded)

    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_concat = torch.cat(targets)
    return images, targets_concat, target_lengths


def main():
    model = CRNN().to(DEVICE)
    model.train()

    all_samples = HPDataset(DATASET_ROOT).samples

    rng = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(all_samples), generator=rng).tolist()
    n_train = int(len(all_samples) * 0.85)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]

    train_ds = HPDataset(DATASET_ROOT, samples=train_samples, augment=True)
    val_ds = HPDataset(DATASET_ROOT, samples=val_samples, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for images, targets, target_lengths in pbar:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            target_lengths = target_lengths.to(DEVICE)

            logits = model(images).permute(1, 0, 2)
            log_probs = torch.log_softmax(logits, dim=-1)
            T, B, C = log_probs.shape
            input_lengths = torch.full((B,), T, dtype=torch.long, device=DEVICE)

            loss = nn.functional.ctc_loss(
                log_probs, targets, input_lengths, target_lengths,
                blank=BLANK_IDX, zero_infinity=True,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets, target_lengths in val_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)
                target_lengths = target_lengths.to(DEVICE)

                logits = model(images).permute(1, 0, 2)
                log_probs = torch.log_softmax(logits, dim=-1)
                T, B, C = log_probs.shape
                input_lengths = torch.full((B,), T, dtype=torch.long, device=DEVICE)

                val_loss += nn.functional.ctc_loss(
                    log_probs, targets, input_lengths, target_lengths,
                    blank=BLANK_IDX, zero_infinity=True,
                ).item()

                pred_ids = logits.argmax(dim=2).permute(1, 0)
                for b in range(B):
                    start = target_lengths[:b].sum()
                    end = target_lengths[:b + 1].sum()
                    pred = decode(pred_ids[b].tolist())
                    gt = decode_gt(targets[start:end].tolist())
                    if pred == gt:
                        correct += 1
                    total += 1

        avg_val = val_loss / len(val_loader)
        acc = correct / total if total > 0 else 0
        scheduler.step(avg_val)

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss: {train_loss/len(train_loader):.4f} | "
            f"val_loss: {avg_val:.4f} | "
            f"acc: {acc:.4f}"
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "hp_crnn_best.pt")
            print(f"  -> saved hp_crnn_best.pt (acc={acc:.4f})")

    print(f"\nBest accuracy: {best_acc:.4f}")

    model.load_state_dict(torch.load("hp_crnn_best.pt"))
    model.eval()
    with torch.no_grad():
        images, targets, target_lengths = next(iter(val_loader))
        images = images.to(DEVICE)
        logits = model(images).permute(1, 0, 2)
        pred_ids = logits.argmax(dim=2).permute(1, 0)
        print("\nSample predictions:")
        for b in range(min(16, len(target_lengths))):
            start = target_lengths[:b].sum()
            end = target_lengths[:b + 1].sum()
            pred = decode(pred_ids[b].tolist())
            gt = decode_gt(targets[start:end].tolist())
            print(f"  gt: {gt:>6s}  pred: {pred:>6s}")


if __name__ == "__main__":
    main()
