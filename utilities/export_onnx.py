import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def export_yolo():
    from ultralytics import YOLO

    model_path = os.path.join(BASE, "src", "objects_detection", "best.pt")
    model = YOLO(model_path)
    model.export(format="onnx", imgsz=640, opset=17)
    out_path = model_path.replace(".pt", ".onnx")
    print("YOLO exported to", out_path)


def export_hp():
    from hp_recognition.CRNN_hp_recogniser import CRNN

    ckpt_path = os.path.join(BASE, "src", "hp_recognition", "hp_crnn_best.pt")
    out_path = os.path.join(BASE, "src", "hp_recognition", "hp_crnn_best.onnx")

    model = CRNN()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    dummy = torch.zeros(1, 1, 48, 64)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch", 3: "width"}, "logits": {0: "batch", 1: "time"}},
    )
    print("HP CRNN exported to", out_path)


if __name__ == "__main__":
    export_yolo()
    export_hp()