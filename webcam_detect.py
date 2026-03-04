# File: D:/VEHICLE_DATASET/webcam_detect.py
from ultralytics import YOLO
import cv2

# ----- STEP 1: Load your trained model -----
model_path = "D:/VEHICLE_DATASET/runs/full_train/weights/best.pt"
model = YOLO(model_path)

# ----- STEP 2: Open webcam -----
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ----- STEP 3: Run inference in a loop -----
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLOv8 prediction on the frame
    results = model(frame)

    # Draw bounding boxes on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLOv8 Vehicle Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----- STEP 4: Release resources -----
cap.release()
cv2.destroyAllWindows()