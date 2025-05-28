import cv2
import argparse

def process_eroded_image(image_path, output_path=None):
    """
    Preprocess eroded inscription images using CLAHE, bilateral filtering, and unsharp masking.
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save the processed image (optional)
    Returns:
        np.ndarray: Processed image
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Convert to LAB and apply CLAHE to L channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Bilateral filtering (preserve edges, reduce noise)
    bilateral = cv2.bilateralFilter(img_clahe, d=9, sigmaColor=75, sigmaSpace=75)

    # Unsharp masking (sharpening)
    gaussian = cv2.GaussianBlur(bilateral, (0, 0), sigmaX=2)
    unsharp = cv2.addWeighted(bilateral, 1.5, gaussian, -0.5, 0)

    # Save if requested
    if output_path is not None:
        cv2.imwrite(str(output_path), unsharp)
        print(f"Processed image saved to {output_path}")
    return unsharp

def main():
    parser = argparse.ArgumentParser(description='Preprocess eroded inscription images (CLAHE + Bilateral + Unsharp Masking).')
    parser.add_argument('input', help='Path to the input image')
    parser.add_argument('--output', '-o', help='Output filename for processed image')
    args = parser.parse_args()
    process_eroded_image(args.input, args.output)

if __name__ == "__main__":
    main()
