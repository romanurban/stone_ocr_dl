import argparse
from pathlib import Path
import sys

# Import processing functions from each module
def try_import(module, func):
    try:
        mod = __import__(module, fromlist=[func])
        return getattr(mod, func)
    except Exception as e:
        print(f"Error importing {func} from {module}: {e}")
        sys.exit(1)

process_low_contrast_image = try_import('stone_ocr.preprocessing.low_contrast', 'process_low_contrast_image')
process_reflection_image = try_import('stone_ocr.preprocessing.reflection_correction', 'process_reflection_image')
process_eroded_image = try_import('stone_ocr.preprocessing.eroded_correction', 'process_eroded_image')
process_shadow_image = try_import('stone_ocr.preprocessing.shadow_correction', 'process_inscription')
process_low_contrast_fix = try_import('stone_ocr.preprocessing.low_contrast_fix', 'enhance_contrast')


def run_all_corrections(input_folder, output_folder, mask_folder=None, both_shadow=False):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    allowed_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

    # Detect defect type from folder name
    folder_name = input_folder.name.lower()
    apply_lc = 'low-contrast' in folder_name or 'low_contrast' in folder_name
    apply_lcf = 'low-contrast' in folder_name or 'low_contrast' in folder_name
    apply_eroded = 'eroded' in folder_name
    apply_shadow = 'shadow' in folder_name
    apply_reflection = 'reflection' in folder_name

    for img_path in input_folder.iterdir():
        if img_path.suffix.lower() not in allowed_exts:
            continue
        name = img_path.stem
        # Only apply relevant corrections
        if apply_lc:
            out_lc = output_folder / f"{name}_low_contrast.png"
            process_low_contrast_image(str(img_path), str(out_lc))
        if apply_lcf:
            out_lcf = output_folder / f"{name}_low_contrast_fix.png"
            process_low_contrast_fix(str(img_path), str(out_lcf))
        if apply_eroded:
            out_eroded = output_folder / f"{name}_eroded.png"
            process_eroded_image(str(img_path), str(out_eroded))
        if apply_shadow:
            out_shadow = output_folder / f"{name}_shadow_enhanced.jpg"
            process_shadow_image(str(img_path), str(output_folder / name), both_shadow)
        if apply_reflection and mask_folder:
            mask_path = Path(mask_folder) / f"{img_path.name}"
            if mask_path.exists():
                out_reflection = output_folder / f"{name}_reflection.png"
                process_reflection_image(str(img_path), str(mask_path), str(out_reflection))
            else:
                print(f"No mask found for {img_path.name}, skipping reflection correction.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all preprocessing corrections on a folder of images.")
    parser.add_argument('input_folder', help='Path to folder with input images')
    parser.add_argument('output_folder', help='Path to folder for saving processed images')
    parser.add_argument('--mask_folder', help='Folder with binary masks for reflection correction (optional)')
    parser.add_argument('--both_shadow', action='store_true', help='Generate both binary and non-binary outputs for shadow correction')
    args = parser.parse_args()
    run_all_corrections(args.input_folder, args.output_folder, args.mask_folder, args.both_shadow)
