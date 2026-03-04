from ultralytics import YOLO
import cv2
import winsound
import time

# Load YOLO vehicle model
model = YOLO("D:/VEHICLE_DATASET/runs/full_train/weights/best.pt")

# Vehicle class from your dataset
vehicle_classes = ["Vehicle"]

# Webcam
cap = cv2.VideoCapture(0)

last_alert_time = 0
alert_cooldown = 2  # seconds between buzzes

print("===================================")
print("VEHICLE DETECTION ALERT SYSTEM STARTED")
print("Press Q to exit")
print("===================================")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    alert_detected = False

    # Run YOLO detection
    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label in vehicle_classes:
                alert_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Play alarm if vehicle detected
    if alert_detected and time.time() - last_alert_time > alert_cooldown:
        cv2.putText(frame, "VEHICLE ALERT!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3)
        print("ALERT: Vehicle detected!")
        winsound.PlaySound("D:/VEHICLE_DATASET/alarm.wav",
                           winsound.SND_FILENAME | winsound.SND_ASYNC)
        last_alert_time = time.time()

    # Show webcam
    cv2.imshow("Vehicle Detection Alert", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
