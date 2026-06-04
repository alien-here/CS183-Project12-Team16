import argparse
from pathlib import Path

from ocr_engine import configure_tesseract, recognize_text
from preprocess import preprocess_document, save_image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

#Author:Wu Xuran
#Function:parse_args
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch OCR for document images.")
    parser.add_argument("--input-dir", required=True, help="Folder containing images.")
    parser.add_argument("--output-dir", required=True, help="Folder for text outputs.")
    parser.add_argument("--processed-dir", help="Optional folder for processed images.")
    parser.add_argument(
        "--tesseract",
        help="Optional path to tesseract.exe if it is not available in PATH.",
    )
    parser.add_argument("--psm", default="6", help="Tesseract page segmentation mode.")
    return parser.parse_args()

#Author:Wu Xuran
#Function:iter_image
def iter_images(input_dir: Path):
    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path

#Author:Wu Xuran
#Function:run batch OCR process
def main() -> None:
    args = parse_args()
    configure_tesseract(args.tesseract)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    processed_dir = Path(args.processed_dir) if args.processed_dir else None

    output_dir.mkdir(parents=True, exist_ok=True)
    if processed_dir:
        processed_dir.mkdir(parents=True, exist_ok=True)

    for image_path in iter_images(input_dir):
        processed = preprocess_document(str(image_path))
        text = recognize_text(processed, psm=args.psm)

        output_path = output_dir / f"{image_path.stem}.txt"
        output_path.write_text(text, encoding="utf-8")

        if processed_dir:
            processed_path = processed_dir / f"{image_path.stem}_processed.png"
            save_image(processed, str(processed_path))

        print(f"Processed: {image_path.name} -> {output_path.name}")


if __name__ == "__main__":
    main()

