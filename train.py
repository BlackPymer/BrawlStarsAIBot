from ultralytics import YOLO

DATA = "datasets/datasetV2/data.yaml"
MODEL = "yolo26n.pt"
EPOCHS = 60
BATCH = 16
IMSZ = 640
WORKERS = 4
DEVICE = 0
SEED = 42
PROJECT = "runs/train"
NAME = "v2"
PATIENCE = 20
LR0 = 0.01

HSV_H = 0.015
HSV_S = 0.5
HSV_V = 0.3
TRANSLATE = 0.1
SCALE = 0.4
FLIPLR = 0.5
MOSAIC = 1.0
MIXUP = 0.1
COPY_PASTE = 0.1
ERASING = 0.3
DEGREES = 10.0

model = YOLO(MODEL)
model.train(
    data=DATA,
    epochs=EPOCHS,
    imgsz=IMSZ,
    batch=BATCH,
    workers=WORKERS,
    device=DEVICE,
    seed=SEED,
    project=PROJECT,
    name=NAME,
    exist_ok=True,
    patience=PATIENCE,
    lr0=LR0,
    hsv_h=HSV_H,
    hsv_s=HSV_S,
    hsv_v=HSV_V,
    translate=TRANSLATE,
    scale=SCALE,
    fliplr=FLIPLR,
    mosaic=MOSAIC,
    mixup=MIXUP,
    copy_paste=COPY_PASTE,
    erasing=ERASING,
    degrees=DEGREES,
)
