import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Image_VietOCR_Folder = os.path.join(BASE_DIR, "cropped_images")
Image_OutPut_Folder = os.path.join(BASE_DIR, "output_images")
Output_Text_File = os.path.join(BASE_DIR, "ocr_results")
