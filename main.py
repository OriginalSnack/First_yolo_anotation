from ultralytics import YOLO

model = YOLO("yolov8n.pt")

result = model.train(data="config.yaml", epochs=2)
