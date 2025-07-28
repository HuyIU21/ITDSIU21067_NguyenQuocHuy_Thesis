import os
import cv2
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image, ImageOps
from typing import Dict, List, Tuple, Union, Any, Optional
from Praser import clean_text_blocks, extract_invoice_data
from ultralytics_module import load_yolo_model
from YOLO import detect_objects, extract_bounding_boxes
from IOU import IOU

# YOLO model path - update this to your model path
YOLO_MODEL_PATH = "62_best.pt"  # Update with your actual model path

# Explicitly define what should be imported when using 'from combined import *'
__all__ = ['process_invoice', 'InvoiceProcessor', 'BatchProcessor']

class InvoiceProcessor:
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.device = 'cuda' if self.use_gpu else 'cpu'
        
        # Initialize YOLO model for object detection
        self.yolo_model = load_yolo_model(YOLO_MODEL_PATH)
        
        # Initialize PaddleOCR for text detection
        self.paddle_ocr = PaddleOCR(
            use_angle_cls=True,
            lang='vi',
            use_gpu=self.use_gpu,
            show_log=False
        )
        
        # Initialize VietOCR for improved Vietnamese text recognition
        self.viet_cfg = Cfg.load_config_from_name('vgg_transformer')
        self.viet_cfg['device'] = self.device
        self.viet_cfg['predictor']['beamsearch'] = False
        self.viet_cfg['predictor']['beamwidth'] = 5
        self.viet_predictor = Predictor(self.viet_cfg)
    
    def detect_text_regions(self, img_np: np.ndarray) -> List[Dict[str, Union[np.ndarray, List[int]]]]:
        # Convert to BGR for YOLO if needed
        if img_np.shape[2] == 4:  # RGBA
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        elif img_np.shape[2] == 3:  # RGB
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:  # Already BGR or grayscale
            img_bgr = img_np
            
        # Detect objects using YOLO
        results = detect_objects(self.yolo_model, img_bgr)
        boxes = extract_bounding_boxes(results)
        
        # Process detected boxes
        cropped_regions = []
        processed_boxes = []
        iou_threshold = 0.5
        
        for box in boxes:
            xyxy = box.xyxy.cpu().numpy()[0]
            x_min, y_min, x_max, y_max = map(int, xyxy)
            
            # Skip if the region is too small
            if (x_max - x_min) < 10 or (y_max - y_min) < 10:
                continue
                
            # Check for duplicate regions using IOU
            is_duplicate = any(IOU([x_min, y_min, x_max, y_max], pb) > iou_threshold 
                             for pb in processed_boxes)
            if is_duplicate:
                continue
                
            # Store the box coordinates
            box_coords = [x_min, y_min, x_max, y_max]
            processed_boxes.append(box_coords)
            
            # Crop and save the region
            cropped_region = img_bgr[y_min:y_max, x_min:x_max]
            cropped_regions.append({
                'image': cropped_region,
                'box': box_coords  # [x_min, y_min, x_max, y_max] in original image
            })
            
        return cropped_regions, processed_boxes
    
    def process_image(self, image: Union[Image.Image, np.ndarray], use_vietocr: bool = True) -> Dict[str, Any]:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            img_np = np.array(image)
            # Convert RGBA to RGB if needed
            if img_np.ndim == 3 and img_np.shape[2] == 4:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            elif img_np.ndim == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_np = image.copy()
        
        # Detect text regions using YOLO
        text_regions, yolo_boxes = self.detect_text_regions(img_np)
        
        all_text_blocks = []
        all_viet_lines = []
        
        # Process each detected text region with its original position
        for region_idx, region_data in enumerate(text_regions):
            # Get the cropped region image and its coordinates
            region = region_data['image']
            yolo_box = region_data['box']
            yolo_x_min, yolo_y_min, yolo_x_max, yolo_y_max = yolo_box
            
            # Use PaddleOCR for text detection on the region
            result = self.paddle_ocr.ocr(region, cls=True)
            
            # Store all text lines in this YOLO region
            region_text_lines = []
        
            if result is not None and len(result) > 0 and result[0] is not None:
                for line in result[0]:
                    if not line or len(line) < 2:
                        continue
                        
                    # Extract bounding box and text with confidence
                    pts, (text, confidence) = line
                    
                    # Convert points to numpy array
                    pts_arr = np.array(pts, dtype=np.int32)
                    
                    # Safely get relative coordinates within the YOLO region
                    try:
                        if pts_arr.size == 0 or len(pts_arr.shape) < 2 or pts_arr.shape[1] < 2:
                            print(f"Warning: Invalid points array shape: {pts_arr.shape}, skipping...")
                            continue
                            
                        xs = pts_arr[:, 0]
                        ys = pts_arr[:, 1]
                        
                        if len(xs) == 0 or len(ys) == 0:
                            print("Warning: Empty coordinates array, skipping...")
                            continue
                            
                        x0, y0 = min(xs), min(ys)
                        x1, y1 = max(xs), max(ys)
                        
                        # Ensure coordinates are within the region bounds
                        x0 = max(0, min(x0, region.shape[1] - 1))
                        y0 = max(0, min(y0, region.shape[0] - 1))
                        x1 = max(0, min(x1, region.shape[1] - 1))
                        y1 = max(0, min(y1, region.shape[0] - 1))
                        
                        # Ensure x1 > x0 and y1 > y0
                        if x1 <= x0 or y1 <= y0:
                            print(f"Warning: Invalid bounding box coordinates: ({x0}, {y0}, {x1}, {y1}), skipping...")
                            continue
                        
                        # Calculate absolute coordinates in the original image
                        abs_x0 = yolo_x_min + x0
                        abs_y0 = yolo_y_min + y0
                        abs_x1 = yolo_x_min + x1
                        abs_y1 = yolo_y_min + y1
                        
                    except Exception as e:
                        print(f"Error processing coordinates: {str(e)}, skipping...")
                        continue
                    
                    # Extract text region for OCR
                    roi = region[y0:y1, x0:x1]
                    if roi.size > 0:
                        try:
                            if use_vietocr:
                                # Use VietOCR for improved text recognition
                                pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                                recognized_text = self.viet_predictor.predict(pil_roi)
                                # For VietOCR, we use the original confidence from PaddleOCR
                                text_confidence = float(confidence)
                            else:
                                # Use PaddleOCR's text recognition
                                recognized_text = text  # PaddleOCR already provides text in the result
                                text_confidence = float(confidence)
                            
                            # Create text block with both relative and absolute coordinates
                            text_block = {
                                'text': recognized_text,
                                'box': pts_arr.tolist(),  # Relative to YOLO region
                                'abs_box': [abs_x0, abs_y0, abs_x1, abs_y1],  # Absolute in original image
                                'yolo_region': region_idx,  # Which YOLO region this belongs to
                                'confidence': text_confidence,
                                'ocr_engine': 'vietocr' if use_vietocr else 'paddleocr'
                            }
                            
                            all_text_blocks.append(text_block)
                            region_text_lines.append(recognized_text)
                            all_viet_lines.append(recognized_text)
                            
                        except Exception as e:
                            print(f"Error processing text region: {str(e)}")
                            continue
                
                # Add a special block to mark the end of this YOLO region
                if region_text_lines:
                    all_text_blocks.append({
                        'text': '--- END OF REGION ---',
                        'box': [],
                        'abs_box': [yolo_x_min, yolo_y_min, yolo_x_max, yolo_y_max],
                        'yolo_region': region_idx,
                        'is_region_separator': True
                    })
        
        # If no text regions were found with YOLO, fall back to processing the whole image
        if not all_text_blocks:
            print("No text regions found with YOLO, falling back to full image processing")
            result = self.paddle_ocr.ocr(img_np, cls=True)
            if result is not None and len(result) > 0 and result[0] is not None:
                for line in result[0]:
                    if not line or len(line) < 2:
                        continue
                        
                    pts, (text, confidence) = line
                    pts_arr = np.array(pts, dtype=np.int32)
                    xs = pts_arr[:, 0]
                    ys = pts_arr[:, 1]
                    x0, y0 = xs.min(), ys.min()
                    x1, y1 = xs.max(), ys.max()
                    
                    roi = img_np[y0:y1, x0:x1]
                    if roi.size > 0:
                        try:
                            if use_vietocr:
                                # Use VietOCR for improved text recognition
                                pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                                recognized_text = self.viet_predictor.predict(pil_roi)
                            else:
                                # Use PaddleOCR's text recognition
                                recognized_text = text  # PaddleOCR already provides text in the result
                            
                            # For fallback case, create a single region with all text
                            all_text_blocks.append({
                                'text': recognized_text,
                                'box': pts_arr.tolist(),
                                'abs_box': [x0, y0, x1, y1],
                                'yolo_region': 0,  # All fallback text goes to region 0
                                'confidence': float(confidence),
                                'ocr_engine': 'vietocr' if use_vietocr else 'paddleocr'
                            })
                            all_viet_lines.append(recognized_text)
                        except Exception as e:
                            print(f"Error processing fallback text region: {str(e)}")
                            continue
        
        # Combine all text
        full_text = '\n'.join(all_viet_lines)
        
        # Clean the text using the function from Praser.py
        cleaned_text = clean_text_blocks(full_text)
        
        # Extract structured data using the function from Praser.py
        try:
            df_result = extract_invoice_data(cleaned_text)
        except Exception as e:
            print(f"Error extracting invoice data: {str(e)}")
            # Return an empty DataFrame with expected columns if extraction fails
            df_result = pd.DataFrame(columns=[
                'Tên hàng hóa', 'Số lượng', 'Đơn giá', 'Thành tiền',
                'Người bán', 'MST người bán', 'Người mua', 'MST người mua'
            ])
        
        return {
            'text_blocks': all_text_blocks,  # List of text blocks with metadata
            'full_text': cleaned_text,       # Combined cleaned text
            'dataframe': df_result           # Extracted structured data
        }

# Create a global instance of InvoiceProcessor
_processor = InvoiceProcessor()

def process_invoice(image: Union[Image.Image, np.ndarray, str], use_vietocr: bool = True) -> Dict[str, Any]:
    try:
        # If image is a file path, load it
        if isinstance(image, str):
            image = Image.open(image)
        
        # Process the image with the specified OCR engine
        return _processor.process_image(image, use_vietocr=use_vietocr)
    except Exception as e:
        print(f"Error in process_invoice: {str(e)}")
        return {
            'text_blocks': [],
            'full_text': f"Error processing image: {str(e)}",
            'dataframe': pd.DataFrame(columns=[
                'Tên hàng hóa', 'Số lượng', 'Đơn giá', 'Thành tiền',
                'Người bán', 'MST người bán', 'Người mua', 'MST người mua'
            ])
        }

# For backward compatibility
class BatchProcessor:
    def __init__(self, input_folder, output_img_folder, output_text_folder, lang='vi'):
        # Cấu hình thư mục đầu vào và đầu ra
        self.input_folder = input_folder
        self.output_img_folder = output_img_folder
        self.output_text_folder = output_text_folder
        
        # Khởi tạo PaddleOCR (Phát hiện các bounding boxes)
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        
        # Khởi tạo VietOCR (Nhận diện văn bản)
        viet_cfg = Cfg.load_config_from_name('vgg_transformer')
        viet_cfg['device'] = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
        viet_cfg['predictor']['batch_first'] = True
        self.viet_predictor = Predictor(viet_cfg)
        
        # Tạo các thư mục nếu chưa có
        os.makedirs(self.output_img_folder, exist_ok=True)
        os.makedirs(self.output_text_folder, exist_ok=True)

    def numpy_to_pil(self, img_bgr):
        """Chuyển đổi BGR numpy array thành PIL RGB."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    def process_image(self, fname):
        """Xử lý từng ảnh để phát hiện văn bản và nhận diện nội dung."""
        # Đọc ảnh gốc
        in_path = os.path.join(self.input_folder, fname)
        img_bgr = cv2.imread(in_path)
        if img_bgr is None:
            return None, None

        # PaddleOCR để phát hiện bounding boxes
        res = self.paddle_ocr.ocr(in_path, cls=True)
        if not res or not res[0]:
            print(f"No text boxes in {fname}")
            return None, None

        # Mảng lưu kết quả văn bản VietOCR
        viet_lines = []

        # Vẽ bounding box lên ảnh để output
        annotated = img_bgr.copy()

        for line in res[0]:
            pts, (raw_text, score) = line
            # Convert pts to int and draw
            pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts_arr], True, (0, 255, 0), 2)

            # Tính bbox từ polygon
            xs = pts_arr[:, :, 0].flatten()
            ys = pts_arr[:, :, 1].flatten()
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()

            # Crop và feed vào VietOCR
            roi = img_bgr[y0:y1, x0:x1]
            pil_roi = self.numpy_to_pil(roi)
            refined = self.viet_predictor.predict(pil_roi)

            viet_lines.append(refined)

        # Lưu ảnh annotation
        out_img_path = os.path.join(self.output_img_folder, f"ann_{fname}")
        cv2.imwrite(out_img_path, annotated)

        # Ghi văn bản VietOCR ra file .txt
        stem = os.path.splitext(fname)[0]
        out_txt_path = os.path.join(self.output_text_folder, f"{stem}.txt")
        with open(out_txt_path, 'w', encoding='utf-8') as f:
            for line in viet_lines:
                f.write(line + "\n")

        return out_img_path, out_txt_path

    def process_batch(self):
        """Xử lý tất cả các ảnh trong thư mục đầu vào."""
        for fname in os.listdir(self.input_folder):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Xử lý từng ảnh
            out_img_path, out_txt_path = self.process_image(fname)
            if out_img_path and out_txt_path:
                print(f"Processed {fname}:")
                print(f" - Annotated image: {out_img_path}")
                print(f" - VietOCR texts:  {out_txt_path}")