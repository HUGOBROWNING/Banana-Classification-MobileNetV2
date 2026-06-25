"""YOLOv8n banana detector — COCO class 46."""

import os
from ultralytics import YOLO

BANANA_COCO_CLASS = 46
DEFAULT_CONF = 0.40
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
YOLO_PATH = os.path.join(ASSETS_DIR, "yolov8n.pt")


class BananaDetector:
    def __init__(self, model_path: str = YOLO_PATH, conf: float = DEFAULT_CONF):
        if not os.path.exists(model_path):
            # Download once into assets/ on first run.
            YOLO("yolov8n.pt").save(model_path)
        self.model = YOLO(model_path)
        self.banana_class = BANANA_COCO_CLASS
        self.conf = conf
        # Warm up with a tiny dummy inference.
        import numpy as np

        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.model(dummy, verbose=False, classes=[self.banana_class], conf=self.conf)

    def detect_best(self, image, imgsz: int = 320):
        """Return best banana detection or None.

        Args:
            image: BGR numpy array or PIL Image.
        Returns:
            dict with keys x1, y1, x2, y2, conf — or None.
        """
        results = self.model(
            image,
            verbose=False,
            classes=[self.banana_class],
            conf=self.conf,
            imgsz=imgsz,
        )
        best = None
        for r in results:
            for box in r.boxes:
                c = float(box.conf[0])
                if best is None or c > best["conf"]:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    best = {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "conf": c}
        return best
