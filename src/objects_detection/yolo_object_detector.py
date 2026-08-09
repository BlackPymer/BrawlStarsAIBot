from objects_detection.object_detector import *
from ultralytics import YOLO

import os as _os
import torch as _torch
MODEL = _os.path.join(_os.path.dirname(__file__), "best.pt")
MODEL_ONNX = _os.path.join(_os.path.dirname(__file__), "best.onnx")


class YoloObjectDetector(ObjectDetector):
    def __init__(self):
        super(YoloObjectDetector, self).__init__()
        if _torch.cuda.is_available() or not _os.path.isfile(MODEL_ONNX):
            model_path = MODEL
        else:
            model_path = MODEL_ONNX
        self.det_model = YOLO(model_path)

    def detect(self, frame) -> list:
        return self.det_model.predict(frame, imgsz=640, conf=0.3, verbose=False)
