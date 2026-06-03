from ultralytics import YOLO
from pathlib import Path
import os

ROOT = Path(__file__).parent.resolve()

model = YOLO("yolo11n.pt")

results = model.train(
    data=str(ROOT / "dataset" / "data.yaml"),
    epochs=50,
    imgsz=640,
    batch=16,
    project=str(ROOT / "runs"),
    name="pool_detector",
    exist_ok=True
)

print("Training Finished")
print("Project:", ROOT)
print("Save Dir:", results.save_dir)
