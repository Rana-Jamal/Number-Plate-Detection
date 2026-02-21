# vehicle_detection.py

import os
import cv2
from ultralytics import YOLO
from config import TRUCK_MODEL, TRUCK_CONF, TRUCK_CLASSES, TRUCK_FOLDER


class VehicleDetector:

    def __init__(self):
        print("Loading Vehicle Detection Model (GPU)...")
        self.model = YOLO(TRUCK_MODEL)

    def detect(self, frame, frame_idx):
        results = self.model(
            frame,
            classes=TRUCK_CLASSES,
            conf=TRUCK_CONF,
            device=0,
            verbose=False
        )[0]

        vehicles = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            filename = f"vehicle_f{frame_idx:06d}_c{conf:.2f}.jpg"
            path = os.path.join(TRUCK_FOLDER, filename)
            cv2.imwrite(path, crop)

            vehicles.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "crop": crop,
                "filename": filename
            })

        return vehicles
