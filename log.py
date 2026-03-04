from ultralytics import YOLO
import cv2
import pandas as pd
from datetime import datetime

# Load trained YOLO model
model = YOLO("D:/VEHICLE_DATASET/runs/full_train/weights/best.pt")

# Vehicle class
vehicle_classes = ["Vehicle"]

# Initialize webcam
cap = cv2.VideoCapture(0)

# DataFrame to store vehicle counts
df = pd.DataFrame(columns=["Timestamp", "Vehicle_Count"])

print("===================================")
print("VEHICLE DETECTION & LOGGING STARTED")
print("Press Q to exit")
print("===================================")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    vehicle_count = 0

    # Run YOLO detection
    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label in vehicle_classes:
                vehicle_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display vehicle count on frame
    cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    # Append count to DataFrame
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.concat([df, pd.DataFrame({"Timestamp": [timestamp], "Vehicle_Count": [vehicle_count]})], ignore_index=True)

    # Show frame
    cv2.imshow("Vehicle Detection & Logging", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save Excel log
excel_path = "D:/VEHICLE_DATASET/vehicle_counts.xlsx"
df.to_excel(excel_path, index=False)
print(f"Vehicle counts saved to {excel_path}")