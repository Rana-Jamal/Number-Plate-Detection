# ocr_module.py

import re
import os
from paddleocr import PaddleOCR
from pathlib import Path
from config import OCR_FOLDER


def clean_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


class OCRReader:

    def __init__(self):
        print("Loading PaddleOCR (GPU)...")
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=True
        )

    def read(self, plate_img, plate_filename):
        result = self.ocr.ocr(plate_img, cls=True)

        text = "No text detected"
        conf = 0.0

        if result and result[0]:
            texts = [line[1][0] for line in result[0]]
            confs = [line[1][1] for line in result[0]]
            text = clean_plate_text(" ".join(texts))
            conf = max(confs)

        # Save OCR result
        txt_name = f"{Path(plate_filename).stem}_ocr.txt"
        with open(os.path.join(OCR_FOLDER, txt_name),
                  "w", encoding="utf-8") as f:
            f.write(f"OCR Text: {text}\nConfidence: {conf:.2f}\n")

        return text, conf
