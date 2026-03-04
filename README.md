# Vehicle-Detection-System
🚗 Real-Time Vehicle Detection & Traffic Logging System (YOLOv8)

A real-time vehicle detection and monitoring system built using YOLOv8, OpenCV, and Pandas. This project detects vehicles from a live webcam feed, displays bounding boxes with confidence scores, counts vehicles in real time, and logs timestamp-based vehicle count data into an Excel file for analysis.

📌 Overview

This system uses a custom-trained YOLOv8 model to:

Detect vehicles in real time

Draw bounding boxes with confidence scores

Display live vehicle count

Log vehicle count with timestamps into an Excel file

The project demonstrates practical deployment of computer vision models with real-time data logging and structured ML project development.

🎯 Features

Real-time vehicle detection

Bounding boxes with confidence score

Live vehicle count display

Automatic Excel logging

Lightweight YOLOv8n model

Clean project structure

🛠 Technologies Used

Python 3.10

Ultralytics YOLOv8

OpenCV

Pandas

OpenPyXL

Anaconda (pytorch_env)

📂 Project Structure

Vehicle-Detection-System/
│
├── vehicle_alert_system.py
├── vehicle_detection_excel.py
├── requirements.txt
├── README.md
└── .gitignore

Note: Dataset, training runs, and model weights are excluded due to GitHub size limitations.

🧠 Model Information

Model: YOLOv8n (custom trained)

Classes: 1 (Vehicle)

Output: Bounding boxes + confidence score

Logging: Timestamp + vehicle count

Model weights (best.pt) are not included in this repository due to file size restrictions.

⚙ Installation

Clone the repository:

git clone <your-repository-link>
cd Vehicle-Detection-System

Create environment:

conda create -n pytorch_env python=3.10
conda activate pytorch_env

Install dependencies:

pip install -r requirements.txt

▶ How to Run

Ensure the model path inside the script is correctly set:

model = YOLO("D:/VEHICLE_DATASET/runs/full_train/weights/best.pt")

Run the program:

python vehicle_alert_system.py

Press Q to exit.

📊 Output

Webcam window opens

Vehicles are detected and highlighted

Vehicle count updates in real time

On exit, an Excel file vehicle_counts.xlsx is generated

Excel format example:

Timestamp | Vehicle_Count
2026-03-04 18:10:02 | 2
2026-03-04 18:10:03 | 1

🔒 Limitations

Counts vehicles per frame (no object tracking)

No unique vehicle ID tracking

Excel file size increases over long sessions

🚀 Future Improvements

Add object tracking (DeepSORT / ByteTrack)

Add FPS monitoring

Add traffic density analytics

Deploy as web-based dashboard

Store logs in database instead of Excel

Add multi-class vehicle detection
