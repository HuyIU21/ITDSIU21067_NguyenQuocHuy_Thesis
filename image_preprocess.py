import cv2
import os
import numpy as np
from PIL import Image
from IOU import remove_duplicate_boxes, has_text

def correct_skew(image):
    """ Xoay ảnh về hướng đúng bằng cách tìm góc nghiêng của văn bản """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Tìm các đường thẳng trong ảnh
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta * 180 / np.pi) - 90  # Chuyển từ radian sang độ
            if -45 < angle < 45:  # Chỉ lấy các góc hợp lý
                angles.append(angle)

        if angles:
            median_angle = np.median(angles)  # Lấy góc trung bình
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
    
    return image  # Trả về ảnh gốc nếu không tìm thấy góc nghiêng

def enhance_image(image):
    """ Tăng chất lượng ảnh trước khi đưa vào OCR """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Tăng cường độ tương phản bằng CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Làm mịn nhưng giữ cạnh bằng Bilateral Filter
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return enhanced

def preprocess_image(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 2  # Upscale image
    gray_img = cv2.resize(gray_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_img
def preprocess_image(image):
    """ Gọi cả hai hàm trên để xoay và làm rõ ảnh """
    image = correct_skew(image)  # Xoay ảnh nếu bị nghiêng
    enhanced = enhance_image(image)  # Làm rõ ảnh trước khi OCR
    return enhanced


def process_image(image_path, output_dir, predictor, yolo_labels, scale_percent=50):
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return None

    # Sao chép ảnh để vẽ bounding box
    image_with_boxes = image.copy()
    
    # Resize ảnh
    width = int(image_with_boxes.shape[1] * scale_percent / 100)
    height = int(image_with_boxes.shape[0] * scale_percent / 100)
    image_with_boxes_resized = cv2.resize(image_with_boxes, (width, height), interpolation=cv2.INTER_AREA)

    # Tiền xử lý ảnh
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological để kết nối ký tự
    kernel = np.ones((5, 30), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Tìm contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 10:
            bounding_boxes.append((x, y, w, h))

    # Loại bỏ trùng lặp
    bounding_boxes = remove_duplicate_boxes(bounding_boxes, iou_threshold=0.5)
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[1])  # Sắp xếp theo y

    if not bounding_boxes:
        print(f"Không tìm thấy văn bản trong ảnh: {image_path}")
        return None

    # Xác định scale
    scale_x = width / image.shape[1]
    scale_y = height / image.shape[0]

    # Kết quả nhận diện
    results = []

    # Duyệt qua các bounding box và xử lý theo nhãn
    for (idx, (x, y, w, h)) in enumerate(bounding_boxes):
        label = yolo_labels.get(idx, "paragraph")  # Mặc định là paragraph nếu không có nhãn

        # Nếu là bảng (table), xử lý riêng
        if label == "table":
            table_region = image[y:y+h, x:x+w]
            table_cells = extract_table_cells(table_region)  # Tách ô trong bảng
            table_text = []
            
            for (cx, cy, cw, ch) in table_cells:
                cell_region = table_region[cy:cy+ch, cx:cx+cw]
                cell_region_rgb = cv2.cvtColor(cell_region, cv2.COLOR_BGR2RGB)
                cell_region_pil = Image.fromarray(cell_region_rgb)
                cell_text = predictor.predict(cell_region_pil)
                
                table_text.append(cell_text.strip())

            results.append(("Table", table_text, (x, y, w, h)))

        # Nếu là đoạn văn bản bình thường
        else:
            text_region = image[y:y+h, x:x+w]

            # Kiểm tra xem vùng ảnh có chứa văn bản không
            if not has_text(text_region):
                continue

            text_region_rgb = cv2.cvtColor(text_region, cv2.COLOR_BGR2RGB)
            text_region_pil = Image.fromarray(text_region_rgb)
            text = predictor.predict(text_region_pil)

            if not text.strip():
                continue
            
            results.append((text, (x, y, w, h)))

        # Vẽ bounding box lên ảnh đã resize
        x_scaled = int(x * scale_x)
        y_scaled = int(y * scale_y)
        w_scaled = int(w * scale_x)
        h_scaled = int(h * scale_y)
        color = (0, 0, 255) if label == "table" else (0, 255, 0)  # Bảng màu đỏ, đoạn văn màu xanh
        cv2.rectangle(image_with_boxes_resized, (x_scaled, y_scaled), 
                      (x_scaled + w_scaled, y_scaled + h_scaled), color, 2)

    # Nếu không có kết quả nào, bỏ qua lưu ảnh
    if not results:
        print(f"Không có văn bản nào được nhận diện trong ảnh: {image_path}")
        return None

    # Lưu ảnh kết quả
    output_image_path = os.path.join(output_dir, f"output_{os.path.basename(image_path)}")
    cv2.imwrite(output_image_path, image_with_boxes_resized)

    return results

def extract_table_cells(table_image):
    """Tách bảng thành từng ô bằng phương pháp contour detection."""
    gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morphological để làm nổi bật ô
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Tìm contours các ô
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cells = [cv2.boundingRect(c) for c in contours]

    return sorted(cells, key=lambda c: (c[1], c[0]))  # Sắp xếp theo hàng và cột

