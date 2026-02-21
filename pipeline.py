# pipeline.py

import os
import cv2
import time
from config import *
from vehicle_detection import VehicleDetector
from plate_detection import PlateDetector
from ocr_module import OCRReader


def setup_folders():
    for folder in [TRUCK_FOLDER, PLATE_FOLDER, OCR_FOLDER, FINAL_FOLDER]:
        os.makedirs(folder, exist_ok=True)


def draw_plate(img, x1, y1, x2, y2, text, plate_conf):

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(img,
                f"Plate {plate_conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2)

    if text == "No text detected":
        return

    font_scale = 0.7
    thickness = 2

    text_size = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness
    )[0]

    text_x = x1 + (x2 - x1 - text_size[0]) // 2
    text_y = y2 + 30

    cv2.putText(img, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2)

    cv2.putText(img, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness)


def main():

    setup_folders()

    vehicle_detector = VehicleDetector()
    plate_detector = PlateDetector()
    ocr_reader = OCRReader()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(fps / TARGET_FPS))

    frame_idx = 0
    total_time = 0
    processed = 0

    while cap.isOpened():

        start = time.time()

        # Frame skipping
        for _ in range(stride - 1):
            if not cap.grab():
                break
            frame_idx += 1

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        output = frame.copy()
        has_detection = False

        vehicles = vehicle_detector.detect(frame, frame_idx)

        for vehicle in vehicles:

            x1, y1, x2, y2 = vehicle["bbox"]
            v_conf = vehicle["conf"]

            plates = plate_detector.detect(
                vehicle["crop"],
                vehicle["filename"]
            )

            for plate in plates:

                px1, py1, px2, py2 = plate["bbox"]
                p_conf = plate["conf"]

                text, ocr_conf = ocr_reader.read(
                    plate["crop"],
                    plate["filename"]
                )

                ox1, oy1 = x1 + px1, y1 + py1
                ox2, oy2 = x1 + px2, y1 + py2

                draw_plate(output, ox1, oy1, ox2, oy2, text, p_conf)

                has_detection = True

            # Draw vehicle box
            cv2.rectangle(output, (x1, y1), (x2, y2),
                          (0, 0, 255), 2)

            cv2.putText(output,
                        f"Vehicle {v_conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2)

        cv2.imshow("Vehicle + Plate + OCR (GPU)", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if has_detection:
            cv2.imwrite(os.path.join(
                FINAL_FOLDER,
                f"frame_{frame_idx:06d}.jpg"),
                output)

        processed += 1
        total_time += time.time() - start

    cap.release()
    cv2.destroyAllWindows()

    if processed:
        print(f"\nAvg time/frame: {(total_time/processed)*1000:.1f} ms")


if __name__ == "__main__":
    main()
