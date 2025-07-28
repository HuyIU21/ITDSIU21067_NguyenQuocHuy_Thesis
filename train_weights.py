import os
from ultralytics import YOLO

if __name__ == "__main__":  
    data_yaml_path = os.path.join(os.getcwd(), "data", "data_OCR.yaml")

    # Load model
    model = YOLO("yolov8n.pt")

    # Train model
    model.train(data=data_yaml_path, epochs=60, device=0,save_period=10, batch=5)