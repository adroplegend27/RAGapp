import os
import json
import uuid
import tempfile
import subprocess
from typing import Dict, List, Optional, Tuple
import io
import requests
import time

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans

# Configuration
OUTPUT_DIR = "extracted_text"
TESSERACT_CMD = "tesseract"  # Update this path if needed
# For Windows: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# API Configuration
OPENAI_API_KEY = "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"  # Will be updated via user input
AZURE_VISION_KEY = "your_azure_vision_key"  # Will be updated via user input
AZURE_VISION_ENDPOINT = "https://your-resource-name.cognitiveservices.azure.com/"  # Will be updated via user input

def ensure_directory(directory_path: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def sanitize_text_for_json(text: str) -> str:
    """
    Sanitize text to make it safe for JSON serialization.
    This prevents the "No closing quotation" error.
    """
    if not text:
        return ""
        
    # Replace problematic characters
    text = text.replace('\\', '\\\\')  # Escape backslashes first
    text = text.replace('"', '\\"')    # Escape double quotes
    text = text.replace('\n', ' ')     # Replace newlines with spaces
    text = text.replace('\r', ' ')     # Replace carriage returns
    text = text.replace('\t', ' ')     # Replace tabs
    text = text.replace('\b', ' ')     # Replace backspace
    text = text.replace('\f', ' ')     # Replace form feed
    
    # Remove control characters
    text = ''.join(ch if ord(ch) >= 32 or ch in ['\n', '\r', '\t'] else ' ' for ch in text)
    
    # Handle other potentially problematic unicode characters
    result = ""
    for ch in text:
        if ord(ch) < 65536:  # Basic Multilingual Plane is safe
            result += ch
        else:
            result += ' '  # Replace with space
    
    return result

def extract_text_from_pdf(pdf_path: str) -> Tuple[str, bool]:
    """
    Extract text from PDF using PyMuPDF.
    Returns text and a flag indicating if OCR is needed.
    """
    document = fitz.open(pdf_path)
    text = ""
    needs_ocr = True

    # Try to extract native text first
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        page_text = page.get_text()
        text += page_text
    
    # If we got meaningful text, we might not need OCR
    if len(text.strip()) > 100:  # Arbitrary threshold
        needs_ocr = False
    
    document.close()
    return sanitize_text_for_json(text), needs_ocr

def perform_ocr_on_pdf(pdf_path: str) -> str:
    """Use Tesseract to OCR a PDF document."""
    document = fitz.open(pdf_path)
    full_text = ""
    
    for page_num in range(len(document)):
        # Get the page
        page = document.load_page(page_num)
        
        # Convert to image
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        
        # Save as temporary image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_image_path = temp_file.name
            pix.save(temp_image_path)
        
        # OCR the image
        try:
            image = Image.open(temp_image_path)
            page_text = pytesseract.image_to_string(image)
            full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        except Exception as e:
            full_text += f"\n--- Page {page_num + 1} Error: {str(e)} ---\n"
        
        # Clean up temp file
        os.unlink(temp_image_path)
    
    document.close()
    return sanitize_text_for_json(full_text)

def process_image_with_ocr(image_path: str) -> str:
    """Process an image file with Tesseract OCR."""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return sanitize_text_for_json(text)
    except Exception as e:
        return f"Error processing image: {str(e)}"

def process_image_with_whisper(image_path: str) -> str:
    """
    Process an image with handwritten content using OpenAI's Whisper API
    or Microsoft Azure's Computer Vision API for handwriting recognition.
    
    For handwritten content, we use Azure's OCR since Whisper is primarily
    for audio. This function handles both approaches.
    """
    # Determine if we should use Azure's specialized handwriting OCR
    use_azure = True  # Set to False if you prefer to use local processing
    
    if use_azure and AZURE_VISION_KEY != "your_azure_vision_key":
        try:
            return sanitize_text_for_json(process_image_with_azure_vision(image_path))
        except Exception as e:
            print(f"Azure Vision API error: {str(e)}. Falling back to local processing.")
    
    # If Azure isn't available or failed, use local processing
    return sanitize_text_for_json(improved_local_handwriting_ocr(image_path))

def process_image_with_azure_vision(image_path: str) -> str:
    """
    Process an image with handwritten content using Microsoft Azure's 
    Computer Vision API for handwriting recognition.
    """
    # Prepare the image
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    
    # API endpoint for Read operation (OCR)
    vision_url = f"{AZURE_VISION_ENDPOINT}vision/v3.2/read/analyze"
    
    # Set request headers
    headers = {
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': AZURE_VISION_KEY
    }
    
    # Send the request
    response = requests.post(vision_url, headers=headers, data=image_data)
    
    if response.status_code != 202:
        raise Exception(f"Request failed with status code: {response.status_code}, {response.text}")
    
    # Get the operation location (URL to get results)
    operation_location = response.headers["Operation-Location"]
    
    # Poll for results
    headers = {'Ocp-Apim-Subscription-Key': AZURE_VISION_KEY}
    result = None
    max_retries = 10
    retry_delay = 1  # seconds
    
    for _ in range(max_retries):
        response = requests.get(operation_location, headers=headers)
        result = response.json()
        
        if result.get("status") not in ["notStarted", "running"]:
            break
        
        time.sleep(retry_delay)
        retry_delay *= 1.5  # Exponential backoff
    
    # Extract text from the result
    extracted_text = ""
    if result and result.get("status") == "succeeded":
        for read_result in result.get("analyzeResult", {}).get("readResults", []):
            for line in read_result.get("lines", []):
                extracted_text += line.get("text", "") + "\n"
    
    return extracted_text.strip()

def improved_local_handwriting_ocr(image_path: str) -> str:
    """
    Improved local processing for handwritten content using OpenCV and Tesseract
    with specialized preprocessing.
    """
    import pytesseract
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return "Error loading image"
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to handle varying illumination
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours to identify text regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size to remove noise
    min_area = 100  # Minimum area to consider as text
    text_regions = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            text_regions.append((x, y, w, h))
    
    # Sort regions by y-coordinate (top to bottom)
    text_regions.sort(key=lambda r: r[1])
    
    # If no meaningful regions found, process the entire image
    if not text_regions:
        text_regions = [(0, 0, image.shape[1], image.shape[0])]
    
    # Process each region
    extracted_text = ""
    for x, y, w, h in text_regions:
        # Extract region with padding
        padding = 10
        x_min = max(0, x - padding)
        y_min = max(0, y - padding)
        x_max = min(image.shape[1], x + w + padding)
        y_max = min(image.shape[0], y + h + padding)
        
        region = thresh[y_min:y_max, x_min:x_max]
        
        # Noise removal
        kernel = np.ones((1, 1), np.uint8)
        region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel)
        region = cv2.morphologyEx(region, cv2.MORPH_OPEN, kernel)
        
        # Invert back to black text on white background for OCR
        region = cv2.bitwise_not(region)
        
        # Use Tesseract with PSM 6 (assumes a single block of text)
        # and OEM 1 (LSTM neural net mode)
        config = '--psm 6 --oem 1'
        if detect_if_handwritten(image_path):
            # Additional config for handwritten text
            config += ' -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?()-:;\'\" "'
        
        try:
            text = pytesseract.image_to_string(region, config=config)
            extracted_text += text + "\n"
        except Exception as e:
            extracted_text += f"OCR error: {str(e)}\n"
    
    return extracted_text.strip()

def detect_if_handwritten(image_path: str) -> bool:
    """
    Advanced detection of handwritten vs. printed text using
    texture analysis and stroke characteristics.
    
    This is a more sophisticated approach that analyzes the 
    stroke properties, texture, and layout of the text.
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            return False
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Calculate histogram of oriented gradients (simplified)
        # This helps detect stroke characteristics
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)
        
        # Bin the angles
        bins = 16
        angle_bins = np.zeros(bins)
        for i in range(bins):
            lower = i * (180 / bins)
            upper = (i + 1) * (180 / bins)
            mask = (ang >= lower) & (ang < upper)
            angle_bins[i] = np.sum(mag[mask])
        
        # Normalize
        angle_bins = angle_bins / np.sum(angle_bins) if np.sum(angle_bins) > 0 else angle_bins
        
        # Calculate standard deviation of angles
        # Handwritten text typically has more varied stroke directions
        angle_std = np.std(angle_bins)
        
        # Calculate stroke width variation
        # We use a simplified approach by analyzing the width of connected components
        # in the binary image
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate stroke width for each contour
        stroke_widths = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20:  # Skip very small contours
                continue
            
            # Approximate perimeter
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            # Width approximation
            stroke_width = area / perimeter
            stroke_widths.append(stroke_width)
        
        # Calculate coefficient of variation of stroke widths
        if stroke_widths:
            stroke_width_cv = np.std(stroke_widths) / np.mean(stroke_widths) if np.mean(stroke_widths) > 0 else 0
        else:
            stroke_width_cv = 0
        
        # Analyze texture using Local Binary Patterns or Gray-Level Co-occurrence Matrix
        # Here we use a simplified approach with image variance
        texture_variance = np.var(gray)
        
        # Analyze layout regularity
        # Printed text typically has more regular spacing
        layout_score = 0
        if len(contours) > 10:
            # Get bounding boxes
            bboxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 20]
            if bboxes:
                # Calculate vertical positions
                y_positions = [y for _, y, _, _ in bboxes]
                
                # Try to cluster into lines
                if len(y_positions) > 1:
                    kmeans = KMeans(n_clusters=min(len(y_positions), 5), random_state=42).fit(np.array(y_positions).reshape(-1, 1))
                    y_clusters = kmeans.labels_
                    
                    # Calculate standard deviation within clusters
                    y_std_within_clusters = []
                    for i in range(kmeans.n_clusters):
                        cluster_y = [y_positions[j] for j in range(len(y_positions)) if y_clusters[j] == i]
                        if cluster_y:
                            y_std_within_clusters.append(np.std(cluster_y))
                    
                    if y_std_within_clusters:
                        layout_score = np.mean(y_std_within_clusters)
        
        # Combine features into a handwriting score
        handwriting_score = (
            angle_std * 5 +  # More varied angles suggest handwriting
            stroke_width_cv * 3 +  # More varied stroke widths suggest handwriting
            texture_variance / 10000 +  # Higher texture variance suggests handwriting
            layout_score / 10  # Higher layout irregularity suggests handwriting
        )
        
        # Determine if it's handwritten
        # These thresholds may need adjustment based on your specific use case
        return handwriting_score > 0.5
    except Exception as e:
        print(f"Error in handwriting detection: {str(e)}")
        return False  # Default to printed text on error

def extract_images_from_pdf(pdf_path: str) -> List[str]:
    """Extract images from PDF for separate processing."""
    try:
        document = fitz.open(pdf_path)
        image_list = []
        
        # Create a temporary directory for extracted images
        temp_dir = tempfile.mkdtemp()
        
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            image_dict = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_dict):
                xref = img_info[0]
                try:
                    base_image = document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Save the image to a temporary file
                    image_path = os.path.join(temp_dir, f"image_p{page_num+1}_{img_index+1}.png")
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    image_list.append(image_path)
                except Exception as e:
                    print(f"Error extracting image: {str(e)}")
        
        document.close()
        return image_list
    except Exception as e:
        print(f"Error extracting images from PDF: {str(e)}")
        return []

def process_file(file_path: str) -> Dict:
    """
    Process a single file (PDF or image) and return JSON-formatted result.
    """
    file_id = str(uuid.uuid4())[:8]  # Generate a unique ID
    filename = os.path.basename(file_path)
    result = {"file_id": file_id, "filename": filename}
    
    try:
        # Determine file type
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            # First try to extract native text
            pdf_text, needs_ocr = extract_text_from_pdf(file_path)
            
            # If needed, perform OCR
            if needs_ocr:
                ocr_text = perform_ocr_on_pdf(file_path)
                result["extracted_text"] = ocr_text
                result["extraction_method"] = "ocr"
            else:
                result["extracted_text"] = pdf_text
                result["extraction_method"] = "native"
            
            # Extract and process images in the PDF
            images = extract_images_from_pdf(file_path)
            if images:
                image_texts = []
                for img_path in images:
                    try:
                        if detect_if_handwritten(img_path):
                            # Use Whisper or specialized processing for handwriting
                            img_text = process_image_with_whisper(img_path)
                        else:
                            # Use regular OCR
                            img_text = process_image_with_ocr(img_path)
                        
                        if img_text and len(img_text.strip()) > 0:
                            image_texts.append(img_text)
                    except Exception as e:
                        print(f"Error processing embedded image: {str(e)}")
                    finally:
                        # Clean up temp image file
                        try:
                            if os.path.exists(img_path):
                                os.unlink(img_path)
                        except:
                            pass
                
                if image_texts:
                    result["image_extracted_text"] = image_texts
        
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
            # Process image file
            if detect_if_handwritten(file_path):
                # Use Whisper or specialized processing for handwriting
                img_text = process_image_with_whisper(file_path)
                result["extraction_method"] = "whisper/azure"
            else:
                # Use regular OCR
                img_text = process_image_with_ocr(file_path)
                result["extraction_method"] = "ocr"
            
            result["extracted_text"] = img_text
        
        else:
            result["extracted_text"] = "Unsupported file format"
            result["extraction_method"] = "none"
    
    except Exception as e:
        result["error"] = str(e)
        result["extracted_text"] = f"Error processing file: {str(e)}"
        result["extraction_method"] = "error"
    
    return result

def main():
    # Setup
    ensure_directory(OUTPUT_DIR)
    
    # API key setup prompt
    print("Do you want to set up API keys for improved handwriting recognition? (y/n)")
    setup_apis = input().lower().strip() == 'y'
    
    if setup_apis:
        print("Enter your OpenAI API key (leave blank to skip):")
        openai_key = input().strip()
        if openai_key:
            global OPENAI_API_KEY
            OPENAI_API_KEY = openai_key
            
        print("Enter your Azure Vision API key (leave blank to skip):")
        azure_key = input().strip()
        if azure_key:
            global AZURE_VISION_KEY
            AZURE_VISION_KEY = azure_key
            
        print("Enter your Azure Vision endpoint (leave blank to skip):")
        azure_endpoint = input().strip()
        if azure_endpoint:
            global AZURE_VISION_ENDPOINT
            AZURE_VISION_ENDPOINT = azure_endpoint
    
    # Directory containing files to process
    input_dir = "attachments"  # Change to your input directory
    print(f"Processing files from directory: {input_dir}")
    
    # Check if directory exists
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} does not exist. Creating it...")
        os.makedirs(input_dir)
        print(f"Please place files to process in the {input_dir} directory and run the script again.")
        return
    
    # Get list of files
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    if not files:
        print(f"No files found in {input_dir}. Please add files and run again.")
        return
    
    print(f"Found {len(files)} files to process.")
    
    # Process all files in directory
    processed_files = []
    
    for i, filename in enumerate(files):
        file_path = os.path.join(input_dir, filename)
        
        # Process the file
        print(f"Processing {i+1}/{len(files)}: {filename}...")
        try:
            result = process_file(file_path)
            processed_files.append(result)
            
            # Save individual result
            output_path = os.path.join(OUTPUT_DIR, f"{result['file_id']}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
            print(f"  Saved result to {output_path}")
        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "all_processed_files.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(processed_files, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessed {len(processed_files)} files. Summary saved to {summary_path}")
    print(f"Results are available in the {OUTPUT_DIR} directory.")

if __name__ == "__main__":
    main()
