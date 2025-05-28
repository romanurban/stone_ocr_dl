import cv2
import numpy as np
import argparse

def process_reflection_image(image_path, mask_path, output_path=None, show=False):
    """
    Correct reflections in images using a provided mask.
    Args:
        image_path (str): Path to input image
        mask_path (str): Path to binary mask (reflection region=255, background=0)
        output_path (str): Path to save the processed image (optional)
        show (bool): Whether to display the result using OpenCV
    Returns:
        np.ndarray: Processed image
    """
    # Read image and mask
    img = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        print(f"Error: Could not read image or mask at {image_path}, {mask_path}")
        return

    # Ensure mask is binary
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask_bin)

    # --- Background (non-reflection) processing ---
    # CLAHE on L channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    # Gaussian blur
    blurred = cv2.GaussianBlur(img_clahe, (5, 5), 0)
    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)

    # --- Reflection region processing ---
    reflection = cv2.bitwise_and(img, img, mask=mask_bin)
    reflection_inv = cv2.bitwise_not(reflection)

    # --- Combine processed background and reflection region ---
    background = cv2.bitwise_and(sharpened, sharpened, mask=mask_inv)
    combined = cv2.add(background, reflection_inv)

    if output_path is not None:
        cv2.imwrite(str(output_path), combined)
        print(f"Processed image saved to {output_path}")
    if show:
        cv2.imshow("Reflection Corrected", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return combined

def main():
    parser = argparse.ArgumentParser(description='Correct reflections in images using a mask (CLAHE + blur + sharpen + selective inversion).')
    parser.add_argument('input', help='Path to the input image')
    parser.add_argument('mask', help='Path to the binary mask image (reflection region=255)')
    parser.add_argument('--output', '-o', help='Output filename for processed image')
    parser.add_argument('--show', action='store_true', help='Show the result image')
    args = parser.parse_args()
    process_reflection_image(args.input, args.mask, args.output, args.show)

if __name__ == "__main__":
    main()
