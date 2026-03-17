from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="bird-dataset",
    epochs=32,
    lr0=0.0001,
    imgsz=224
)

