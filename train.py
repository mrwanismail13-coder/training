from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=10,
    imgsz=640,
    batch=16,
    project="runs",
    name="pool_detector"
)
