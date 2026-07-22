import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DATASET_ROOT = Path("health_dataset")

BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HP_NORM = 20000.0


class HPRegressor(nn.Module):
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
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.regressor(x)
        return x.squeeze(1)


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


def augment(img):
    aug = img.astype(np.float32)

    if random.random() < 0.8:
        alpha = random.uniform(0.5, 1.5)
        beta = random.uniform(-50, 50)
        aug = aug * alpha + beta

    if random.random() < 0.4:
        ksize = random.choice([3, 5, 7])
        aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)

    if random.random() < 0.4:
        noise = np.random.randn(*aug.shape).astype(np.float32) * random.uniform(5, 30)
        aug = aug + noise

    if random.random() < 0.5:
        h, w = aug.shape[:2]
        angle = random.uniform(-5, 5)
        scale = random.uniform(0.88, 1.12)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if random.random() < 0.3:
        h, w = aug.shape[:2]
        x_off = random.randint(0, w // 3)
        y_off = random.randint(0, h // 3)
        cw = random.randint(w // 6, w // 2)
        ch = random.randint(h // 6, h // 2)
        aug[y_off:y_off+ch, x_off:x_off+cw] = random.uniform(80, 180)

    if random.random() < 0.3:
        sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        aug = cv2.filter2D(aug, -1, sharpen)

    if random.random() < 0.3:
        h, w = aug.shape[:2]
        dw = random.randint(2, 6)
        aug = cv2.resize(aug, (max(w // dw, 8), h), interpolation=cv2.INTER_LINEAR)
        aug = cv2.resize(aug, (w, h), interpolation=cv2.INTER_LINEAR)

    if random.random() < 0.2:
        h, w = aug.shape[:2]
        s = random.uniform(0.5, 1.5)
        pts1 = np.float32([[0, 0], [w, 0], [0, h]])
        pts2 = np.float32([[0, 0], [w, 0], [random.uniform(-w*0.05, w*0.05), h]])
        M = cv2.getAffineTransform(pts1, pts2)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if random.random() < 0.2:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        aug = cv2.dilate(aug, kernel, iterations=1)

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
        target = torch.tensor(hp / HP_NORM, dtype=torch.float32)
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
    targets = torch.stack(targets)
    return images, targets


def main():
    model = HPRegressor().to(DEVICE)
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

    best_mae = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for images, targets in pbar:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            preds = model(images)
            loss = nn.functional.smooth_l1_loss(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_loss = 0.0
        abs_errors = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)

                preds = model(images)
                val_loss += nn.functional.smooth_l1_loss(preds, targets).item()

                pred_hp = (preds * HP_NORM).cpu().numpy()
                true_hp = (targets * HP_NORM).cpu().numpy()
                abs_errors.extend(np.abs(pred_hp - true_hp))

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        mae = float(np.mean(abs_errors))
        med_ae = float(np.median(abs_errors))
        rel_errors = np.array(abs_errors) / (np.array([s[1] for s in val_samples]) + 1)
        mean_rel = float(np.mean(rel_errors))
        med_rel = float(np.median(rel_errors))
        scheduler.step(avg_val)

        print(
            f"Epoch {epoch:3d} | "
            f"train: {avg_train:.4f} | "
            f"val: {avg_val:.4f} | "
            f"MAE: {mae:.0f} | "
            f"MedAE: {med_ae:.0f} | "
            f"Rel: {mean_rel:.3f}/{med_rel:.3f}"
        )

        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), "hp_linear_best.pt")
            print(f"  -> saved hp_linear_best.pt (MAE={mae:.0f})")

    print(f"\nBest MAE: {best_mae:.0f}")

    model.load_state_dict(torch.load("hp_linear_best.pt"))
    model.eval()
    with torch.no_grad():
        images, targets = next(iter(val_loader))
        images = images.to(DEVICE)
        preds = model(images)
        pred_hp = (preds * HP_NORM).cpu().numpy()
        true_hp = (targets * HP_NORM).cpu().numpy()
        print("\nSample predictions:")
        for b in range(min(16, len(pred_hp))):
            print(f"  gt: {true_hp[b]:>7.0f}  pred: {pred_hp[b]:>7.0f}  diff: {pred_hp[b] - true_hp[b]:>+7.0f}")


if __name__ == "__main__":
    main()
