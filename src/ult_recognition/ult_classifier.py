import os as _os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from ult_recognition.ult_recogniser import UltRecogniser

ULT_MODEL_PATH = _os.path.join(_os.path.dirname(__file__), "ult_model.pth")
ULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = 64
ULT_PADDING = 10


class UltClassifier(nn.Module):
    def __init__(self):
        super(UltClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 16 * 16)

        x = F.relu(self.fc1(x))
        return self.fc2(x)


def load_ult_model():
    model = UltClassifier()
    model.load_state_dict(torch.load(ULT_MODEL_PATH, map_location=ULT_DEVICE))
    model.to(ULT_DEVICE)
    model.eval()
    return model


def preprocess(image):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.astype("float32") / 255.0)
    tensor = tensor.permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.unsqueeze(0).to(ULT_DEVICE)
    return tensor


def extract_player_crop(frame, objects, padding=ULT_PADDING):
    if frame is None:
        return None
    h, w = frame.shape[:2]
    from objects_detection.object_detector import CLASS_NAMES
    for obj in objects:
        for box in obj.boxes:
            cls_id = int(box.cls[0])
            if CLASS_NAMES[cls_id] != "player":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx1 = max(0, x1 - padding)
            cy1 = max(0, y1 - padding)
            cx2 = min(w, x2 + padding)
            cy2 = min(h, y2 + padding)
            crop = frame[cy1:cy2, cx1:cx2]
            return crop if crop.size > 0 else None
    return None


class UltClassifierRecogniser(UltRecogniser):
    def __init__(self):
        super().__init__()
        self.model = load_ult_model()

    def recognise(self, image) -> bool:
        if image is None:
            return False
        inp = preprocess(image)
        with torch.no_grad():
            logit = self.model(inp)
        prob = torch.sigmoid(logit).item()
        return prob > 0.5