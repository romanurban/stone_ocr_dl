import os
import json
import argparse
from pathlib import Path
import pytesseract
from PIL import Image, UnidentifiedImageError

def perform_ocr(image_path):
    """
    Run OCR on an image using Tesseract.
    
    Args:
        image_path: Path to the image file
    Returns:
        Extracted text as string
    """
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang='eng')
    return text.strip()

def log_result(log_path, entry):
    """
    Append OCR result to the log file.
    
    Args:
        log_path: Path to the log file
        entry: Dictionary containing image name and OCR text
    """
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def process_images(input_folder, output_log, allowed_exts=None):
    """
    Process all images in a folder through OCR.
    
    Args:
        input_folder: Input directory containing images
        output_log: Path to output log file
        allowed_exts: List of allowed file extensions
    """
    input_folder = Path(input_folder)
    allowed_exts = allowed_exts or ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

    for img_path in input_folder.iterdir():
        if img_path.suffix.lower() not in allowed_exts:
            continue

        try:
            text = perform_ocr(img_path)
            entry = {
                'image': str(img_path.name),
                'ocr_text': text
            }
            log_result(output_log, entry)
        except (UnidentifiedImageError, TypeError, OSError) as e:
            print(f"Error processing {img_path.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR on images and log results.")
    parser.add_argument('--input_folder', type=str, required=True, help='Path to input folder with images')
    parser.add_argument('--output_log', type=str, required=True, help='Path to output log file')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_log), exist_ok=True)
    process_images(args.input_folder, args.output_log)
