from pathlib import Path
from collections import Counter

def main():
    data_dir = Path("data")  # change if needed
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return

    class_counts = {}
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            images = [f for f in class_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}]
            class_counts[class_dir.name] = len(images)

    print("Class distribution:")
    for label, count in class_counts.items():
        print(f"  {label}: {count}")

if __name__ == "__main__":
    main()
