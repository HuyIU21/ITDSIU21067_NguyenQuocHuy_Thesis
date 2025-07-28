import re
import pandas as pd
from typing import List
from pathlib import Path
import os

def extract_money_value(text: str) -> float:
    """Extract numeric value from money string"""
    if not text:
        return 0.0
    # Remove all non-digit characters except decimal point and comma
    cleaned = re.sub(r'[^\d,]', '', str(text))
    # Replace comma with dot if it's used as decimal separator
    if ',' in cleaned and len(cleaned) >= 3 and cleaned[-3] == ',':  # If comma is used as decimal separator (e.g., 1,23)
        cleaned = cleaned.replace(',', '.')
    else:  # If comma is used as thousand separator or not applicable
        cleaned = cleaned.replace(',', '')
    try:
        return float(cleaned) if cleaned else 0.0
    except (ValueError, TypeError):
        return 0.0

def extract_tax_code(text: str) -> str:
    """Extract tax code from text"""
    tax_match = re.search(r'(?:Mã số thuế|MST|Tax[\s:]*)([0-9\-\s]{10,15})', text, re.IGNORECASE)
    return tax_match.group(1).strip() if tax_match else ""

def extract_name(text: str, prefixes: list) -> str:
    """Extract name based on common prefixes"""
    for prefix in prefixes:
        if prefix.lower() in text.lower():
            # Extract text after the prefix
            parts = re.split(prefix, text, flags=re.IGNORECASE)
            if len(parts) > 1:
                return parts[1].strip(': ').strip()
    return ""

def extract_invoice_data(full_text: str) -> dict:
    lines = [line.strip() for line in full_text.strip().splitlines() if line.strip()]
    
    # Initialize default values with proper data types
    result = {
        "Người bán": "",
        "MST người bán": "",
        "Địa chỉ người bán": "",
        "Người mua": "",
        "MST người mua": "",
        "Địa chỉ người mua": "",
        "Mẫu số": "01GTKT0/001",  # Default mẫu số phổ biến
        "Ký hiệu": "",
        "Số hóa đơn": "",
        "Ngày hóa đơn": "",
        "Tổng tiền trước thuế": 0.0,
        "Thuế GTGT": 0.0,
        "Tổng tiền thanh toán": 0.0,
        "Hàng hóa dịch vụ": []
    }
    
    # --- 1. Extract seller information ---
    seller_keywords = ["Đơn vị bán hàng", "Công ty", "Tên đơn vị bán", "Bên bán", "Người bán", "Seller", "Company", "Tên đơn vị"]
    seller_tax_keywords = ["Mã số thuế", "MST", "Tax code", "Mã số doanh nghiệp"]
    seller_address_keywords = ["Địa chỉ", "Address", "Địa chỉ người bán"]
    
    seller_section = False
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # Detect seller section
        if any(keyword.lower() in line_lower for keyword in ["bán hàng", "bên bán", "người bán"]):
            seller_section = True
        
        # Extract seller name
        if not result["Người bán"]:
            seller_name = extract_name(line, seller_keywords)
            if seller_name and seller_section:
                result["Người bán"] = seller_name
        
        # Extract seller tax code
        if not result["MST người bán"] and any(keyword.lower() in line_lower for keyword in seller_tax_keywords):
            tax_code = extract_tax_code(line)
            if tax_code:
                result["MST người bán"] = tax_code
                # Next line might be the actual tax code if current line is just the label
                if i + 1 < len(lines) and re.match(r'^[0-9\-\s]+$', lines[i+1].strip()):
                    result["MST người bán"] = lines[i+1].strip()
        
        # Extract seller address
        if not result["Địa chỉ người bán"] and any(keyword.lower() in line_lower for keyword in seller_address_keywords):
            # Get the part after the address keyword
            for kw in seller_address_keywords:
                if kw.lower() in line_lower:
                    addr = line.split(':', 1)[-1].strip()
                    if not addr:  # If empty, try next line
                        addr = lines[i+1].strip() if i+1 < len(lines) else ""
                    result["Địa chỉ người bán"] = addr
                    break
    
    # --- 2. Extract buyer information ---
    buyer_keywords = ["Tên đơn vị mua", "Bên mua", "Người mua", "Khách hàng", "Customer", "Buyer", "Tên đơn vị"]
    buyer_tax_keywords = ["Mã số thuế", "MST", "Tax code", "Mã số doanh nghiệp"]
    buyer_address_keywords = ["Địa chỉ", "Address", "Địa chỉ người mua"]
    
    buyer_section = False
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # Detect buyer section
        if any(keyword.lower() in line_lower for keyword in ["mua hàng", "bên mua", "người mua"]):
            buyer_section = True
        
        # Extract buyer name
        if not result["Người mua"]:
            buyer_name = extract_name(line, buyer_keywords)
            if buyer_name and buyer_section:
                result["Người mua"] = buyer_name
        
        # Extract buyer tax code
        if not result["MST người mua"] and any(keyword.lower() in line_lower for keyword in buyer_tax_keywords):
            tax_code = extract_tax_code(line)
            if tax_code:
                result["MST người mua"] = tax_code
                # Next line might be the actual tax code if current line is just the label
                if i + 1 < len(lines) and re.match(r'^[0-9\-\s]+$', lines[i+1].strip()):
                    result["MST người mua"] = lines[i+1].strip()
        
        # Extract buyer address
        if not result["Địa chỉ người mua"] and any(keyword.lower() in line_lower for keyword in buyer_address_keywords):
            # Get the part after the address keyword
            for kw in buyer_address_keywords:
                if kw.lower() in line_lower:
                    addr = line.split(':', 1)[-1].strip()
                    if not addr:  # If empty, try next line
                        addr = lines[i+1].strip() if i+1 < len(lines) else ""
                    result["Địa chỉ người mua"] = addr
                    break
    
    # --- 3. Extract invoice metadata ---
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # Extract invoice number
        if not result["Số hóa đơn"]:
            if any(keyword in line_lower for keyword in ["số hóa đơn", "số", "no.", "số:", "số ", "mã hóa đơn"]):
                # Try different patterns to extract invoice number
                patterns = [
                    r'(?:số|no\.?|số hóa đơn|mã hóa đơn)[:\s]*(?:số)?[\s:]*(\w[\w-]+\d+)',
                    r'(?:hóa đơn|hđ|số)[\s:]+(\w[\w-]+\d+)',
                    r'^(?:hđ|số|no\.?)[\s:]*(\w[\w-]+\d+)'
                ]
                
                for pattern in patterns:
                    num_match = re.search(pattern, line, re.IGNORECASE)
                    if num_match and num_match.group(1):
                        result["Số hóa đơn"] = num_match.group(1).strip()
                        break
        
        # Extract date
        if not result["Ngày hóa đơn"]:
            if any(keyword in line_lower for keyword in ["ngày", "date", "ngày ", "ngày:"]):
                date_patterns = [
                    r'(?:ngày|date)[\s:]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})',
                    r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
                    r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b'
                ]
                
                for pattern in date_patterns:
                    date_match = re.search(pattern, line, re.IGNORECASE)
                    if date_match:
                        result["Ngày hóa đơn"] = date_match.group(1).strip()
                        break
        
        # Extract mẫu số and ký hiệu
        if not result["Mẫu số"] and any(keyword in line_lower for keyword in ["mẫu số", "mẫu hóa đơn"]):
            pattern = r'(?:mẫu số|mẫu hóa đơn)[\s:]*([\w/]+)'
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                result["Mẫu số"] = match.group(1).strip()
        
        if not result["Ký hiệu"] and any(keyword in line_lower for keyword in ["ký hiệu", "kí hiệu"]):
            pattern = r'(?:ký hiệu|kí hiệu)[\s:]*([\w/]+)'
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                result["Ký hiệu"] = match.group(1).strip()
    
    # --- 4. Extract products/items ---
    items = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Look for product lines (usually starts with a number or has quantity/price patterns)
        if (re.match(r'^\d+[\.\)]?\s*$', line) or  # Line with just a number
            re.search(r'\d+\s*x\s*\d+', line) or     # Quantity x Unit price pattern
            re.search(r'\d+\s+\d+\s+\d+', line)):   # Columns of numbers (qty, price, amount)
            
            # Try to extract product info from current and next lines
            product = {
                "Tên hàng hóa": "",
                "Đơn vị tính": "cái",  # Default unit
                "Số lượng": 1.0,
                "Đơn giá": 0.0,
                "Thành tiền": 0.0
            }
            
            # Current line might be product name or STT
            if re.match(r'^\d+[\.\)]?\s*$', line):
                # This is likely an STT, next line is product name
                if i + 1 < len(lines):
                    product["Tên hàng hóa"] = lines[i+1]
                    i += 1
            else:
                product["Tên hàng hóa"] = line
            
            # Look for quantity, unit price, and amount in next lines
            for j in range(1, 4):
                if i + j >= len(lines):
                    break
                
                current_line = lines[i+j]
                
                # Check for quantity x unit price pattern (e.g., "2 x 1,500,000")
                qty_price_match = re.search(r'(\d+)\s*x\s*([\d,.]+)', current_line)
                if qty_price_match:
                    product["Số lượng"] = float(qty_price_match.group(1).replace(',', ''))
                    product["Đơn giá"] = extract_money_value(qty_price_match.group(2))
                    if product["Số lượng"] and product["Đơn giá"]:
                        product["Thành tiền"] = product["Số lượng"] * product["Đơn giá"]
                    break
                
                # Check for separate quantity and price columns
                numbers = [extract_money_value(n) for n in re.findall(r'[\d,.]+', current_line)]
                if len(numbers) >= 3:  # At least qty, unit price, and amount
                    product["Số lượng"] = numbers[0]
                    product["Đơn giá"] = numbers[1]
                    product["Thành tiền"] = numbers[2] if len(numbers) > 2 else numbers[0] * numbers[1]
                    break
                elif len(numbers) == 2:  # Maybe just quantity and amount
                    product["Số lượng"] = numbers[0]
                    product["Thành tiền"] = numbers[1]
                    if product["Số lượng"]:
                        product["Đơn giá"] = product["Thành tiền"] / product["Số lượng"]
                    break
            
            # Only add if we have a product name
            if product["Tên hàng hóa"]:
                items.append(product)
                i += 1  # Move to next line after processing product
        
        i += 1  # Move to next line
    
    # --- 5. Extract totals ---
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # Extract total before tax
        if any(keyword in line_lower for keyword in ["tổng tiền trước thuế", "tiền hàng", "tổng tiền hàng"]):
            numbers = [extract_money_value(n) for n in re.findall(r'[\d,.]+', line)]
            if numbers:  # Only proceed if we found numbers
                result["Tổng tiền trước thuế"] = max(numbers) if numbers else 0.0
        
        # Extract tax amount
        elif any(keyword in line_lower for keyword in ["thuế gtgt", "vat", "thuế suất"]):
            numbers = [extract_money_value(n) for n in re.findall(r'[\d,.]+', line)]
            if numbers:  # Only proceed if we found numbers
                result["Thuế GTGT"] = max(numbers) if numbers else 0.0
        
        # Extract total amount
        elif any(keyword in line_lower for keyword in ["tổng cộng", "cộng tiền thanh toán", "tổng thanh toán"]):
            numbers = [extract_money_value(n) for n in re.findall(r'[\d,.]+', line)]
            if numbers:  # Only proceed if we found numbers
                result["Tổng tiền thanh toán"] = max(numbers) if numbers else 0.0
    
    # Calculate any missing totals
    if not result["Tổng tiền thanh toán"] and result["Tổng tiền trước thuế"] and result["Thuế GTGT"]:
        result["Tổng tiền thanh toán"] = result["Tổng tiền trước thuế"] + result["Thuế GTGT"]
    
    result["Hàng hóa dịch vụ"] = items
    return result

def clean_text_blocks(text: str) -> str:
    if not text:
        return ""
        
    # Remove special characters but keep Vietnamese characters and common invoice symbols
    text = re.sub(r'[^\w\sÀ-ỹ.,:;!?\-+*/=()\[\]{}%$€¥£¢₫]', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Fix line breaks in words
    text = re.sub(r'([a-zA-ZÀ-ỹ])\s*-\s*([a-zA-ZÀ-ỹ])', r'\1\2', text)  # Join hyphenated words
    
    # Remove lines with only numbers or special characters
    lines = [line for line in text.split('\n') if line.strip() and not re.match(r'^[\s\d\W]+$', line)]
    
    # Normalize the text
    clean_text = '\n'.join(lines)
    
    # Fix number formatting (e.g., 1 000 000 -> 1000000)
    clean_text = re.sub(r'(\d)\s+(?=\d{3}\b)', r'\1', clean_text)  # Fix thousand separators
    clean_text = re.sub(r'(\d)\s*[,.]\s*(\d{3})\b', r'\1\2', clean_text)  # Fix decimal points
    
    # Common OCR corrections
    replacements = {
        '|': '1',  # Common OCR error
        'O': '0',  # Common OCR error
        'o': '0',  # Common OCR error
        'l': '1',  # Common OCR error
        'I': '1',  # Common OCR error
        'B': '8',  # Common OCR error
        'S': '5',  # Common OCR error
        's': '5',  # Common OCR error
        'Z': '2',  # Common OCR error
        'z': '2'   # Common OCR error
    }
    
    for old, new in replacements.items():
        clean_text = clean_text.replace(old, new)
    
    return clean_text

def process_invoice_file(input_file: str, output_file: str = None) -> str:
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Clean the text
    cleaned_text = clean_text_blocks(text)
    
    # Extract structured data
    invoice_data = extract_invoice_data(cleaned_text)
    
    # Create a DataFrame for the items
    items_df = pd.DataFrame(invoice_data["Hàng hóa dịch vụ"])
    
    # Create a summary DataFrame
    summary_data = {
        "Thông tin": [
            "Người bán", "MST người bán", 
            "Người mua", "MST người mua",
            "Số hóa đơn", "Ngày hóa đơn",
            "Tổng tiền trước thuế", "Thuế GTGT", "Tổng tiền thanh toán"
        ],
        "Giá trị": [
            invoice_data["Người bán"], 
            invoice_data["MST người bán"],
            invoice_data["Người mua"],
            invoice_data["MST người mua"],
            invoice_data["Số hóa đơn"],
            invoice_data["Ngày hóa đơn"],
            invoice_data["Tổng tiền trước thuế"],
            invoice_data["Thuế GTGT"],
            invoice_data["Tổng tiền thanh toán"]
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Determine output file path if not provided
    if not output_file:
        input_path = Path(input_file)
        output_file = str(input_path.with_suffix('.xlsx'))
    
    # Create a Pandas Excel writer
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Write summary sheet
        summary_df.to_excel(writer, sheet_name='Thông tin hóa đơn', index=False)
        
        # Write items sheet if there are any items
        if not items_df.empty:
            items_df.to_excel(writer, sheet_name='Hàng hóa dịch vụ', index=False)
        
        # Write raw text sheet
        pd.DataFrame({"Văn bản gốc": [cleaned_text]}).to_excel(
            writer, sheet_name='Văn bản gốc', index=False)
    
    return output_file

def process_directory(input_dir: str, output_dir: str = None) -> List[str]:
    if not output_dir:
        output_dir = input_dir
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all .txt files in the directory
    result_files = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.txt'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.xlsx")
            try:
                output_path = process_invoice_file(input_file, output_file)
                result_files.append(output_path)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    return result_files

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        if os.path.isfile(input_path) and input_path.lower().endswith('.txt'):
            output_file = process_invoice_file(input_path)
            print(f"Processed file saved to: {output_file}")
        elif os.path.isdir(input_path):
            output_dir = sys.argv[2] if len(sys.argv) > 2 else None
            result_files = process_directory(input_path, output_dir)
            print(f"Processed {len(result_files)} files.")
            for f in result_files:
                print(f"- {f}")
        else:
            print(f"Error: {input_path} is not a valid text file or directory")
    else:
        print("Usage: python Praser.py <input_file.txt or directory> [output_directory]")
