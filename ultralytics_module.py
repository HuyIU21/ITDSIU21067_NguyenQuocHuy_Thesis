import ultralytics
from ultralytics import YOLO


def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model