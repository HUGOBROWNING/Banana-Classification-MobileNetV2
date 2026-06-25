"""Banana Ripeness Classifier.

Wraps the trained MobileNetV2 transfer-learning model (.keras).

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

CLASSES = ["unripe", "ripe", "overripe", "rotten"]
IMG_SIZE = (224, 224)

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
MODEL_PATH = os.path.join(ASSETS_DIR, "final_banana_ripeness_model.keras")


class BananaRipenessClassifier:
    def __init__(self, model_path: str = MODEL_PATH):
        self.classes = CLASSES
        self.img_size = IMG_SIZE
        self.model = tf.keras.models.load_model(model_path)
        self._warmup()

    def _warmup(self):
        """Compile the graph once so the first real photo is fast."""
        dummy = np.zeros((1, *self.img_size, 3), dtype=np.float32)
        self.model(dummy, training=False)

    # ----- preprocessing -----
    def _to_array(self, pil_img: Image.Image) -> np.ndarray:
        img = pil_img.convert("RGB").resize(self.img_size)
        return np.array(img).astype("float32")

    def _infer(self, x: np.ndarray) -> np.ndarray:
        return self.model(x, training=False).numpy()[0]

    # ----- ripeness prediction -----
    def predict(
        self,
        pil_img: Image.Image,
        conf_threshold: float = 0.40,
        uncertain_threshold: float = 0.75,
    ) -> dict:
        arr = self._to_array(pil_img)
        x = np.expand_dims(arr / 255.0, axis=0)  # rescale=1./255 (matches training)
        preds = self._infer(x)

        probs = {c: float(p) for c, p in zip(self.classes, preds)}
        ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        top_class, top_conf = ordered[0]
        second_class, second_conf = ordered[1]

        no_banana = top_conf < conf_threshold
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
        }
