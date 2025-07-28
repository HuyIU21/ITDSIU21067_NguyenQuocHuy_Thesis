from ultralytics_module import load_yolo_model
import cv2
import numpy as np
# # import torch
# model = YOLO("yolov8n.pt")

def detect_objects(model_path,image_path):
    results = model_path(image_path)
    return results

def extract_bounding_boxes(results):
    return results[0].boxes