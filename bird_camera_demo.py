from ultralytics import YOLO
import cv2

# Load trained classification model
model = YOLO("runs/classify/train2/weights/best.pt")

# Display names mapping
display_names = {
    "fly": "Fly",
    "not fly": "Not Fly",
    "fly_swim": "Fly & Swim"
}

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run prediction
    results = model.predict(frame, verbose=False)

    # Get probabilities
    probs = results[0].probs

    if probs is not None:
        class_id = probs.top1
        confidence = probs.top1conf

        # Get class name from model
        class_name = results[0].names[class_id]
        
        # Use display name mapping
        display_name = display_names.get(class_name, class_name)
        
        # Format text with display name and confidence
        text = f"{display_name} ({confidence:.2f})"
    else:
        text = "No prediction"

    # Display label on frame
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # Show camera window
    cv2.imshow("Bird Classification", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()