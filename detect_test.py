from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolov8n.pt")

# Run detection on image
results = model("bird.jpg", show=True)

print("Detection completed")