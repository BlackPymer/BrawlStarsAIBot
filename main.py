from ultralytics import YOLO

model = YOLO("src/objects_detection/yolo26s.pt")

if __name__ == "__main__":
    results = model.predict("videos", save=True)
    print("Done")
