# Text Recognition Technique

Text Recognition Technique is a course project for recognizing printed English
text from document images. The system uses Python, OpenCV, Tesseract OCR,
EasyOCR, and Streamlit to provide an automated recognition workflow for uploading
an image, preprocessing it, running OCR, selecting the best result, and
downloading the recognized text.

## Project Information

- Project Type: Reserve
- Project Area: Computer Vision, Image Processing, Recognition System
- Tech Stack: Python, OpenCV, Tesseract OCR, EasyOCR, Streamlit
- Expected Outcome: A machine-learning-assisted automation tool that converts
  printed English text in document images into editable digital text.

## Features

- Upload document images containing printed English text.
- Display the original uploaded image.
- Preprocess the image with OpenCV:
  - OCR-oriented resizing
  - Grayscale conversion
  - Denoising
  - Contrast enhancement
  - Sharpening for mild blur
  - Skew correction for tilted text
  - Rotated candidate search for difficult skewed photos
  - Otsu binary thresholding
  - Adaptive thresholding
  - Morphological processing
- Display the final preprocessed image and optional intermediate steps.
- Recognize English text with Tesseract OCR through `pytesseract`.
- Recognize English text with an optional EasyOCR machine learning model.
- Select a recognition area such as full image, left page, right page, or center
  page area for photographed books and two-page images.
- Select the OCR crop area directly with the mouse to remove unrelated page
  edges or nearby text before recognition starts.
- Enable automatic skew correction or control the recognition angle manually
  with the rotation slider.
- Use Fast, Balanced, or Thorough recognition modes. Fast mode is the default
  for interactive use and avoids slow machine-learning OCR unless selected.
- Rebuild OCR output from detected text-box coordinates so the result keeps
  line breaks and is easier to compare with the original page.
- Try multiple OCR input variants and keep the best result using OCR confidence
  and text quality scoring.
- Show an automation report with selected engine, confidence, word count, and
  processing summary.
- Fine-tune rotation manually when automatic skew detection is not enough.
- Display the OCR result in the browser with one-click copy buttons for
  formatted layout text and plain text.
- Download the recognized text as a `.txt` file when a backup export is needed.
- Show project overview, technology stack, and expected outcomes for presentation.

## Project Structure

```text
.
+-- app.py
+-- main.py
+-- requirements.txt
+-- README.md
+-- data/
|   +-- input/
|   +-- output/
|   +-- processed/
+-- src/
    +-- batch_ocr.py
    +-- ocr_engine.py
    +-- preprocess.py
```

`app.py` is the Streamlit web application for course demonstration. The existing
`main.py` and `src/` files can still be used as a command-line OCR version.

## Environment Installation

Python 3.9 or newer is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Installing Tesseract OCR

This project requires the Tesseract OCR engine in addition to the Python package
`pytesseract`.

### Windows

1. Download and install Tesseract from:
   https://github.com/UB-Mannheim/tesseract/wiki
2. The common installation path is:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

3. Add the Tesseract installation folder to your system PATH, or enter the full
   path in the Streamlit sidebar when running the app.

### macOS

Install with Homebrew:

```bash
brew install tesseract
```

### Linux

On Ubuntu or Debian:

```bash
sudo apt update
sudo apt install tesseract-ocr
```

## How to Run

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

Upload an image that contains printed English text. The system will show the
original image, display the preprocessed image, run OCR, and provide a button to
download the recognized text.

## Function Description

1. Image Upload: Users upload a document image in PNG, JPG, JPEG, BMP, TIF, or
   TIFF format.
2. Original Image Display: The uploaded image is shown in the web page for
   visual comparison.
3. OpenCV Preprocessing: The image is resized for OCR, converted to grayscale,
   denoised, contrast-enhanced, sharpened, deskewed, searched across multiple
   rotation angles, thresholded, and cleaned with morphology.
4. OCR Recognition: Tesseract OCR and the optional EasyOCR machine learning
   model recognize English printed text from candidate images.
5. Automation Selection: The system scores candidate outputs and selects the
   best engine/source automatically.
6. Layout Reconstruction: OCR words are grouped by their detected positions to
   produce a formatted layout view with line breaks, plus a plain text view for
   copying.
7. Result Display: The recognized text and automation report are displayed.
8. Text Copy and Export: Users can copy the recognized text directly from the
   app, or download the OCR output and report as a TXT file.
9. Project Presentation: The app includes the project overview, technical stack,
   and expected outcome for course submission and defense.

## Future Improvements

- Add batch OCR for multiple uploaded images.
- Support PDF input and convert PDF pages to images.
- Add image enhancement controls such as threshold block size and blur strength.
- Add deep-learning based super-resolution for very low-quality images.
- Add document boundary detection and perspective correction for camera photos.
- Fine-tune or train a custom OCR model for the target document style.
- Save OCR history and processed images for experiment comparison.
- Evaluate OCR accuracy with ground-truth text samples.
