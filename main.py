from ultralytics import YOLO

model = YOLO("yolo26n.pt")

if __name__ == "__main__":
    results = model.predict("videos", save=True)
    print("Done")
