from hp_recognition.hp_recogniser import HPRecogniser
import torch
import random
import torch.nn as nn
import numpy as np

HP_MODEL_PATH = "hp_crnn_best.pt"
HP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLANK_IDX = 0
CHARS = "0123456789"
import cv2


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
        aug[y_off:y_off + ch, x_off:x_off + cw] = random.uniform(80, 180)

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
        pts2 = np.float32([[0, 0], [w, 0], [random.uniform(-w * 0.05, w * 0.05), h]])
        M = cv2.getAffineTransform(pts1, pts2)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if random.random() < 0.2:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        aug = cv2.dilate(aug, kernel, iterations=1)

    return np.clip(aug, 0, 255)


def load_hp_model():
    model = CRNN()
    model.load_state_dict(torch.load(HP_MODEL_PATH, map_location=HP_DEVICE))
    model.to(HP_DEVICE)
    model.eval()
    return model


def preprocess_hp_batch(crops, target_h=48):
    tensors = []
    for crop in crops:
        tensor = torch.from_numpy(preprocess(crop, target_h))
        tensors.append(tensor)

    max_w = max(t.shape[2] for t in tensors)
    padded = []
    for t in tensors:
        w = t.shape[2]
        if w < max_w:
            pad = torch.zeros((1, 48, max_w - w), dtype=torch.float32)
            t = torch.cat([t, pad], dim=2)
        padded.append(t)
    return torch.stack(padded).to(HP_DEVICE)


class CRNNHpRecogniser(HPRecogniser):
    def __init__(self):
        super().__init__()
        self.model = load_hp_model()

    def recognise(self, image, crops) -> list:
        if not crops:
            return []
        batch = preprocess_hp_batch(crops)
        with torch.no_grad():
            logits = self.model(batch).permute(1, 0, 2)
        pred_ids = logits.argmax(dim=2).permute(1, 0)
        results = []
        for b in range(len(crops)):
            hp_text = decode(pred_ids[b].tolist())
            results.append(hp_text)
        return results
