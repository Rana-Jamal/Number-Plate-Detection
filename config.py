# config.py

VIDEO_PATH = "test_video/test2.mp4"

TRUCK_FOLDER = "crop_trucks"
PLATE_FOLDER = "crop_plates"
OCR_FOLDER   = "ocr_results"
FINAL_FOLDER = "final_results"

TRUCK_CONF   = 0.40
PLATE_CONF   = 0.40
TRUCK_CLASSES = [2, 7]  # car + truck
TARGET_FPS    = 10

TRUCK_MODEL = "models/yolo11m.pt"
PLATE_MODEL = "models/license_plate_detector.pt"
