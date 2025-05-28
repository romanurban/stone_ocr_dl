import cv2
import numpy as np
import argparse
from pathlib import Path

def process_inscription(image_path, output_prefix=None, generate_both=False):
    """
    Process an inscription image to enhance readability.
    
    Args:
        image_path: Path to input image
        output_prefix: Prefix for output filenames
        generate_both: Generate both binary and non-binary outputs
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Set output paths
    if output_prefix is None:
        input_path = Path(image_path)
        output_prefix = input_path.stem
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initial noise reduction
    denoised = cv2.bilateralFilter(gray, 11, 30, 30)
    
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Edge preservation
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    refined = cv2.filter2D(enhanced, -1, kernel)
    
    # Final enhancement
    final_enhanced = cv2.filter2D(refined, -1, kernel)
    
    # Save non-binary version
    non_binary_output = f"{output_prefix}_enhanced.jpg"
    cv2.imwrite(non_binary_output, final_enhanced)
    print(f"Enhanced image saved to {non_binary_output}")
    
    # Create binary version if requested
    if generate_both:
        # Binarization
        _, otsu = cv2.threshold(final_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(final_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 21, 4)
        combined = cv2.bitwise_and(otsu, adaptive)
        
        # Clean up binary image
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
        final_binary = cv2.medianBlur(cleaned, 3)
        
        # Save binary version
        binary_output = f"{output_prefix}_binary.jpg" 
        cv2.imwrite(binary_output, final_binary)
        print(f"Binary image saved to {binary_output}")
    
    return final_enhanced

def main():
    parser = argparse.ArgumentParser(description='Process inscription images to enhance readability.')
    parser.add_argument('input', help='Path to the input image')
    parser.add_argument('--output', '-o', help='Output filename prefix')
    parser.add_argument('--both', '-b', action='store_true', help='Generate both binary and non-binary outputs')
    
    args = parser.parse_args()
    
    process_inscription(args.input, args.output, args.both)
    
if __name__ == "__main__":
    main()