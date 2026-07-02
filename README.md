# Banana Ripeness Classifier

App can be accessed here: https://banana-classification-mobilenetv2dv.streamlit.app

A two-stage computer vision pipeline that classifies banana ripeness in real 
time using a live camera feed. Built end-to-end: data collection via web 
scraping, model training in Google Colab, and deployment as an interactive 
Streamlit web app.

## How it works

1. **YOLOv8n** detects whether a banana is present in the camera frame
2. **MobileNetV2** (fine-tuned on 13,000+ images) classifies the ripeness 
   stage: Unripe, Ripe, Overripe, or Rotten
3. If no banana is detected, the app displays "No banana detected" rather 
   than producing a spurious classification

## Features

- Live camera inference with bounding box overlay and confidence score
- Image upload mode for static photo classification
- Per-class probability breakdown (confidence chart)
- Uncertainty warning when top confidence falls below 75%
- Training history, confusion matrix and sample predictions visualised 
  in-app
- 98.22% test accuracy on held-out test set

## Tech stack

| Layer | Tools |
|---|---|
| Data | DuckDuckGo image scraping, Kaggle dataset (13K images) |
| Training | TensorFlow, Keras, MobileNetV2, Google Colab (GPU) |
| Detection | YOLOv8n (Ultralytics), COCO class 46 |
| App | Streamlit, streamlit-webrtc, OpenCV, PIL |
| Deployment | Streamlit Community Cloud |

## Model performance

| Metric | Score |
|---|---|
| Test accuracy | 98.22% |
| Validation accuracy (peak) | 98.13% |
| Classes | Unripe, Ripe, Overripe, Rotten |
| Training images | 13,000+ |
| Architecture | MobileNetV2 (fine-tuned, transfer learning) |
