from objects_detection.object_detector import *
from ultralytics import YOLO

MODEL = "best.pt"


class YoloObjectDetector(ObjectDetector):
    def __init__(self):
        super(YoloObjectDetector, self).__init__()
        self.det_model = YOLO(MODEL)

    def detect(self, frame) -> list:
        return self.det_model.predict(frame, imgsz=640, conf=0.3, verbose=False)
