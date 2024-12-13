import torch
import os
from ultralytics import YOLO

model_name = "yolov8_4_pbc"
model_path = os.path.join("ultralytics/cfg/models/v8/", model_name + ".yaml")
model = YOLO(model_path)

model.train(data="ultralytics/cfg/datasets/PBC2.yaml", batch=32, epochs=150, patience=150)