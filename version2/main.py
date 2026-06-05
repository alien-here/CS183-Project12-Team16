import argparse
from pathlib import Path

from src.ocr_engine import configure_tesseract, recognize_text
from src.preprocess import preprocess_document, save_image

#Function:parse command-line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize printed English text from a document image."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--output", help="Path to save recognized text.")
    parser.add_argument("--processed", help="Path to save the preprocessed image.")
    parser.add_argument(
        "--tesseract",
        help="Optional path to tesseract.exe if it is not available in PATH.",
    )
    parser.add_argument(
        "--psm",
        default="6",
        help="Tesseract page segmentation mode. Use 6 for a text block, 3 for full page.",
    )
    return parser.parse_args()

#Function:run single image OCR workflow
def main() -> None:
    args = parse_args()
    configure_tesseract(args.tesseract)

    processed = preprocess_document(args.image)
    text = recognize_text(processed, psm=args.psm)

    if args.processed:
        save_image(processed, args.processed)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")

    print(text.strip())


if __name__ == "__main__":
    main()
