import cv2
import numpy as np
import argparse
from PIL import Image

def process_low_contrast_image(image_path, output_path=None, show=False):
    """
    Enhance low-contrast images using contrast stretching, high-pass filtering, raking light simulation, and lightening.
    Args:
        image_path (str): Path to input grayscale image
        output_path (str): Path to save the processed image (optional)
        show (bool): Whether to display the result using PIL
    Returns:
        np.ndarray: Processed image (RGB)
    """
    # Read image as grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Step 1: Contrast Stretching
    contrast_stretched = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Step 2: High-Pass Filtering using 3x3 kernel
    kernel = np.array([[ -1, -1, -1],
                       [ -1,  9, -1],
                       [ -1, -1, -1]])
    high_pass = cv2.filter2D(contrast_stretched, -1, kernel)

    # Step 3: Combine contrast-stretched image with high-pass filtered image (simulate raking light)
    combined = cv2.addWeighted(contrast_stretched, 0.6, high_pass, 0.4, 0)

    # Step 4: Final lightening for visibility
    lightened = cv2.convertScaleAbs(combined, alpha=1.2, beta=30)  # Increase brightness and slight contrast

    # Convert to RGB for saving/display
    lightened_rgb = cv2.cvtColor(lightened, cv2.COLOR_GRAY2RGB)
    if output_path is not None:
        Image.fromarray(lightened_rgb).save(output_path)
        print(f"Processed image saved to {output_path}")
    if show:
        Image.fromarray(lightened_rgb).show()
    return lightened_rgb

def main():
    parser = argparse.ArgumentParser(description='Enhance low-contrast images (contrast stretching + high-pass + raking light + lightening).')
    parser.add_argument('input', help='Path to the input grayscale image')
    parser.add_argument('--output', '-o', help='Output filename for processed image')
    parser.add_argument('--show', action='store_true', help='Show the result image')
    args = parser.parse_args()
    process_low_contrast_image(args.input, args.output, args.show)

if __name__ == "__main__":
    main()
