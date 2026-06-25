"""Banana Ripeness Classifier.

Wraps the trained MobileNetV2 transfer-learning model (.keras) and an
ImageNet MobileNetV2 used purely as a "is this actually a banana?" filter.

IMPORTANT: the model was trained with Keras ImageDataGenerator(rescale=1./255),
so inference MUST scale pixels to [0, 1] -- NOT mobilenet_v2.preprocess_input.

Class order is fixed and was passed explicitly during training
(classes=['unripe', 'ripe', 'overripe', 'rotten']) -> it overrides the
alphabetical default, so it is hardcoded here as ground truth.
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Ground-truth class order (index -> label) from training.
CLASSES = ["unripe", "ripe", "overripe", "rotten"]
IMG_SIZE = (224, 224)

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
MODEL_PATH = os.path.join(ASSETS_DIR, "final_banana_ripeness_model.keras")


class BananaRipenessClassifier:
    def __init__(self, model_path: str = MODEL_PATH):
        self.classes = CLASSES
        self.img_size = IMG_SIZE
        self.model = tf.keras.models.load_model(model_path)
        # ImageNet filter model is loaded lazily on first banana check.
        self._filter_model = None
        self._preprocess_input = None
        self._decode_predictions = None

    # ----- preprocessing -----
    def _to_array(self, pil_img: Image.Image) -> np.ndarray:
        img = pil_img.convert("RGB").resize(self.img_size)
        return np.array(img).astype("float32")

    # ----- banana detector (ImageNet MobileNetV2) -----
    def _ensure_filter(self):
        if self._filter_model is None:
            from tensorflow.keras.applications.mobilenet_v2 import (
                MobileNetV2,
                preprocess_input,
                decode_predictions,
            )
            self._filter_model = MobileNetV2(weights="imagenet")
            self._preprocess_input = preprocess_input
            self._decode_predictions = decode_predictions

    def is_banana(self, pil_img: Image.Image, threshold: float = 0.10):
        """Returns (is_banana: bool|None, confidence: float|None).

        None means the filter could not run (e.g. weights unavailable offline);
        callers should then fall back to the ripeness-confidence threshold only.
        """
        try:
            self._ensure_filter()
            arr = self._to_array(pil_img)
            x = self._preprocess_input(np.expand_dims(arr.copy(), axis=0))
            preds = self._filter_model.predict(x, verbose=0)
            decoded = self._decode_predictions(preds, top=5)[0]
            for _, label, conf in decoded:
                if "banana" in label.lower() and conf >= threshold:
                    return True, float(conf)
            return False, 0.0
        except Exception:
            return None, None

    # ----- ripeness prediction -----
    def predict(
        self,
        pil_img: Image.Image,
        conf_threshold: float = 0.40,
        uncertain_threshold: float = 0.75,
        banana_threshold: float = 0.10,
        run_banana_filter: bool = True,
    ) -> dict:
        arr = self._to_array(pil_img)
        x = np.expand_dims(arr / 255.0, axis=0)  # rescale=1./255 (matches training)
        preds = self.model.predict(x, verbose=0)[0]

        probs = {c: float(p) for c, p in zip(self.classes, preds)}
        ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        top_class, top_conf = ordered[0]
        second_class, second_conf = ordered[1]

        banana_detected, banana_conf = (None, None)
        if run_banana_filter:
            banana_detected, banana_conf = self.is_banana(pil_img, banana_threshold)

        # No banana if: confidence too low OR the ImageNet filter says it's not a banana.
        no_banana = (top_conf < conf_threshold) or (banana_detected is False)
        uncertain = (not no_banana) and (top_conf < uncertain_threshold)

        return {
            "probs": probs,
            "ordered": ordered,
            "top_class": top_class,
            "top_conf": top_conf,
            "second_class": second_class,
            "second_conf": second_conf,
            "no_banana": no_banana,
            "uncertain": uncertain,
            "banana_detected": banana_detected,
            "banana_conf": banana_conf,
        }
