from ultralytics import YOLO

model = YOLO("runs/detect/yolov8_4_pbc/weights/best.pt")
metrics = model.val(split='val')