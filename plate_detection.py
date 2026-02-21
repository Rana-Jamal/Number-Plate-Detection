# plate_detection.py

import os
import cv2
from ultralytics import YOLO
from pathlib import Path
from config import PLATE_MODEL, PLATE_CONF, PLATE_FOLDER


class PlateDetector:

    def __init__(self):
        print("Loading Plate Detection Model (GPU)...")
        self.model = YOLO(PLATE_MODEL)

    def detect(self, vehicle_crop, vehicle_filename):
        results = self.model(
            vehicle_crop,
            conf=PLATE_CONF,
            device=0,
            verbose=False
        )[0]

        plates = []

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)

            crop = vehicle_crop[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            plate_name = f"{Path(vehicle_filename).stem}_plate{i}_c{conf:.2f}.jpg"
            path = os.path.join(PLATE_FOLDER, plate_name)
            cv2.imwrite(path, crop)

            plates.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "crop": crop,
                "filename": plate_name
            })

        return plates
