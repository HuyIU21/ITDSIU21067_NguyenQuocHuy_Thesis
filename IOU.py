import cv2
import numpy as np

#iou cho YOLO
def IOU(box1,box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area =(x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

# IOU cho vietOCR 
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area
    return iou

# Hàm loại bỏ bounding box trùng lặp dựa trên IoU
def remove_duplicate_boxes(boxes, iou_threshold=0.5):
    if not boxes:
        return []
    
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    keep_boxes = []
    
    while boxes:
        current_box = boxes.pop(0)
        keep_boxes.append(current_box)
        boxes = [box for box in boxes if calculate_iou(current_box, box) < iou_threshold]
    
    return keep_boxes

# Hàm kiểm tra xem vùng ảnh có chứa văn bản không
def has_text(image_region):
    gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    pixel_sum = np.sum(thresh) / 255
    area = image_region.shape[0] * image_region.shape[1]
    if pixel_sum < 0.01 * area:
        return False
    return True