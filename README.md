# YOLOv1 Implementation in PyTorch

This project is a basic implementation of the YOLOv1 (You Only Look Once) object detection algorithm using PyTorch. It is designed to train on the Pascal VOC dataset and perform inference on images.

## Features

* **YOLOv1 Model:** A convolutional neural network based on the YOLOv1 architecture (`src/model.py`).
* **Custom Loss Function:** Implementation of the YOLOv1 loss, including coordinate, objectness, no-objectness, and classification losses (`src/loss.py`).
* **Pascal VOC Dataset Loader:** Data loader specifically for the Pascal VOC dataset format (`src/dataset.py`).
* **Training Script:** Script to train the model on the VOC dataset (`train.py`).
* **Inference Script:** Script to run object detection on new images using a trained model (`infer.py`).
* **Utilities:** Includes Non-Max Suppression (NMS) (`src/utils.py`).

## Requirements

* Python 3.x
* PyTorch
* Other dependencies listed in `requirements.txt`

To install the required packages, run:
```bash
pip install -r requirements.txt
