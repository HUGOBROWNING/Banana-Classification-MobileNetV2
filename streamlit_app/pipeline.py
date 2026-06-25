import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from classifier import CLASSES, BananaRipenessClassifier
from detector import BananaDetector, DEFAULT_CONF

UNCERTAIN_THRESHOLD = 0.75

GUIDE_TITLES = {
    "unripe":   "Unripe",
    "ripe":     "Ripe",
    "overripe": "Overripe",
    "rotten":   "Rotten",
}

COLOURS_BGR = {
    "unripe":   (34, 139, 34),
    "ripe":     (0, 215, 255),
    "overripe": (0, 140, 255),
    "rotten":   (0, 0, 139),
}


def _hex_to_bgr(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)


def _draw_label_bar(out, x1, y1, x2, y2, color_bgr, label):
    font       = cv2.FONT_HERSHEY_SIMPLEX
    scale      = 0.8
    thickness  = 2
    bar_h      = 30
    pad        = 5
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    lx = x1 + (x2 - x1 - tw) // 2
    ly = y1 + th + pad
    cv2.rectangle(out, (x1, y1), (x2, y1 + bar_h), color_bgr, -1)
    cv2.putText(out, label, (lx, ly), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_no_banana_overlay(bgr: np.ndarray) -> np.ndarray:
    h, w    = bgr.shape[:2]
    overlay = np.zeros_like(bgr)
    cv2.putText(
        overlay,
        "No banana detected — point camera at a banana",
        (max(0, w // 8), h // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA
    )
    return cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)


def draw_no_banana_pil(pil_img: Image.Image) -> Image.Image:
    out    = pil_img.copy().convert("RGBA")
    w, h   = out.size
    dimmer = Image.new("RGBA", (w, h), (0, 0, 0, 120))
    out    = Image.alpha_composite(out, dimmer).convert("RGB")
    draw   = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    msg    = "No banana detected — point camera at a banana"
    bbox   = draw.textbbox((0, 0), msg, font=font)
    x      = max(0, (w - (bbox[2] - bbox[0])) // 2)
    y      = h // 2
    draw.text((x + 2, y + 2), msg, fill="black", font=font)
    draw.text((x, y), msg, fill="white", font=font)
    return out


def draw_bbox_on_pil(pil_img, det, ripeness, colors_hex=None):
    if colors_hex is None:
        colors_hex = {
            "unripe": "#228B22", "ripe": "#FFD700",
            "overripe": "#FF8C00", "rotten": "#8B0000",
        }
    top        = ripeness["top_class"]
    confidence = ripeness["top_conf"]
    color      = colors_hex.get(top, "#FFFFFF")
    label      = f"{GUIDE_TITLES[top]}  {confidence * 100:.0f}%"
    out        = pil_img.copy()
    draw       = ImageDraw.Draw(out)
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    bb     = draw.textbbox((0, 0), label, font=font)
    text_w = bb[2] - bb[0]
    text_h = bb[3] - bb[1]
    bar_y1 = max(0, y1 - text_h - 10)
    draw.rectangle([(x1, bar_y1), (x1 + text_w + 10, y1)], fill=color)
    draw.text((x1 + 5, bar_y1 + 2), label, fill="white", font=font)
    if confidence < UNCERTAIN_THRESHOLD:
        all_probs = ripeness.get("all_probabilities", ripeness.get("probs", {}))
        sorted_p  = sorted(all_probs.items(), key=lambda kv: kv[1], reverse=True)
        if len(sorted_p) >= 2:
            warn = f"Uncertain: {GUIDE_TITLES[sorted_p[0][0]]} or {GUIDE_TITLES[sorted_p[1][0]]}?"
            draw.text((10, out.size[1] - 30), warn, fill="#FF5050", font=font)
    return out


def apply_overlay(frame, detector, classifier, colors_hex=None):
    if colors_hex is None:
        colors_hex = {
            "unripe": "#228B22", "ripe": "#FFD700",
            "overripe": "#FF8C00", "rotten": "#8B0000",
        }

    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    det   = detector.detect_best(small, imgsz=320)

    if det is None:
        return draw_no_banana_overlay(frame), {
            "banana_detected": False, "detection": None, "ripeness": None,
        }

    scale_x = frame.shape[1] / small.shape[1]
    scale_y = frame.shape[0] / small.shape[0]
    det = {
        "x1": int(det["x1"] * scale_x),
        "y1": int(det["y1"] * scale_y),
        "x2": int(det["x2"] * scale_x),
        "y2": int(det["y2"] * scale_y),
        "conf": det["conf"],
    }

    if det["conf"] < 0.50:
        return draw_no_banana_overlay(frame), {
            "banana_detected": False, "detection": None, "ripeness": None,
        }

    y1, y2 = det["y1"], det["y2"]
    x1, x2 = det["x1"], det["x2"]
    crop_bgr = frame[y1:y2, x1:x2]

    if crop_bgr.size == 0:
        return draw_no_banana_overlay(frame), {
            "banana_detected": False, "detection": None, "ripeness": None,
        }

    pil_crop = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    ripeness = classifier.predict(pil_crop, conf_threshold=0.0, uncertain_threshold=UNCERTAIN_THRESHOLD)
    ripeness["no_banana"] = False

    top       = ripeness["top_class"]
    color_bgr = _hex_to_bgr(colors_hex[top])
    confidence = ripeness["top_conf"]
    label     = f"{GUIDE_TITLES[top]}  {confidence * 100:.0f}%"

    out = frame.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 3)
    _draw_label_bar(out, x1, y1, x2, y2, color_bgr, label)

    if confidence < UNCERTAIN_THRESHOLD:
        probs    = ripeness.get("probs", {})
        sorted_p = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        if len(sorted_p) >= 2:
            warn = f"Uncertain: {GUIDE_TITLES[sorted_p[0][0]]} or {GUIDE_TITLES[sorted_p[1][0]]}?"
            cv2.putText(out, warn, (10, out.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 255), 2, cv2.LINE_AA)

    return out, {
        "banana_detected": True, "detection": det, "ripeness": ripeness,
    }


class TwoStagePipeline:

    def __init__(self, yolo_conf: float = DEFAULT_CONF):
        self.detector   = BananaDetector(conf=yolo_conf)
        self.classifier = BananaRipenessClassifier()
        self.yolo_conf  = yolo_conf

    def _crop_pil(self, pil_img, det):
        w, h = pil_img.size
        x1 = max(0, det["x1"])
        y1 = max(0, det["y1"])
        x2 = min(w, det["x2"])
        y2 = min(h, det["y2"])
        if x2 <= x1 or y2 <= y1:
            return pil_img
        return pil_img.crop((x1, y1, x2, y2))

    def process_pil(self, pil_img):
        det = self.detector.detect_best(pil_img)
        if det is None:
            return {"banana_detected": False, "detection": None, "ripeness": None}
        crop     = self._crop_pil(pil_img, det)
        ripeness = self.classifier.predict(crop, conf_threshold=0.0, uncertain_threshold=UNCERTAIN_THRESHOLD)
        ripeness["no_banana"] = False
        return {"banana_detected": True, "detection": det, "ripeness": ripeness}

    def process_frame(self, bgr, colors_hex):
        return apply_overlay(bgr, self.detector, self.classifier, colors_hex)