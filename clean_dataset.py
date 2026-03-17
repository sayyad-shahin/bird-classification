from ultralytics import YOLO
import os
import cv2

model = YOLO("yolov8n.pt")   # pretrained detection model

dataset_path = "bird-dataset/train"

for class_folder in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, class_folder)

    for img in os.listdir(class_path):

        img_path = os.path.join(class_path, img)

        image = cv2.imread(img_path)

        if image is None:
            continue

        results = model(image)

        detected_objects = results[0].names

        # check if bird detected
        bird_found = False

        for box in results[0].boxes:
            cls_id = int(box.cls)
            if results[0].names[cls_id] == "bird":
                bird_found = True

        if not bird_found:
            print("Removing:", img_path)
            os.remove(img_path)