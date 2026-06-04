import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from pytesseract import TesseractError, TesseractNotFoundError
from streamlit_cropper import st_cropper


PROJECT_NAME = "Text Recognition Technique"
PROJECT_TYPE = "Reserve"
PROJECT_AREA = "Computer Vision, Image Processing, Recognition System"
TECH_STACK = "Python, OpenCV, Tesseract OCR, EasyOCR, Streamlit"
DEFAULT_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


@dataclass
class PreprocessResult:
    steps: dict[str, np.ndarray]
    ocr_candidates: dict[str, np.ndarray]
    display_image: np.ndarray
    scale: float
    recognition_area: str
    region_box: tuple[int, int, int, int]
    crop_box: tuple[int, int, int, int]
    skew_angle: float
    applied_rotation: float
    skew_method: str
    rotation_search_enabled: bool


@dataclass
class OCRResult:
    text: str
    engine: str
    source: str
    score: float
    confidence: float
    word_count: int


@dataclass
class WordBox:
    text: str
    confidence: float
    left: float
    top: float
    right: float
    bottom: float


COMMON_ENGLISH_WORDS = {
    "a",
    "and",
    "are",
    "ask",
    "but",
    "came",
    "center",
    "chinese",
    "class",
    "course",
    "courses",
    "dear",
    "difficulty",
    "email",
    "for",
    "found",
    "friday",
    "from",
    "have",
    "help",
    "here",
    "hua",
    "i",
    "idea",
    "in",
    "interesting",
    "know",
    "last",
    "learning",
    "let",
    "li",
    "library",
    "me",
    "month",
    "morning",
    "mornings",
    "my",
    "no",
    "note",
    "number",
    "of",
    "ok",
    "on",
    "or",
    "phone",
    "please",
    "provides",
    "reply",
    "sir",
    "some",
    "student",
    "students",
    "summer",
    "taking",
    "the",
    "to",
    "told",
    "tuesdays",
    "university",
    "use",
    "was",
    "which",
    "with",
    "writing",
    "you",
    "your",
    "yours",
}

VALID_SHORT_TOKENS = {
    "a",
    "i",
    "am",
    "an",
    "as",
    "at",
    "be",
    "by",
    "do",
    "go",
    "he",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "no",
    "of",
    "ok",
    "on",
    "or",
    "so",
    "to",
    "up",
    "we",
}


def configure_page() -> None:
    st.set_page_config(
        page_title=PROJECT_NAME,
        layout="wide",
    )
    st.markdown(
        """
        <style>
        :root {
            --ocr-ink: #243142;
            --ocr-muted: #6b7280;
            --ocr-panel: #ffffff;
            --ocr-soft: #eef4ff;
            --ocr-line: #d9e4ef;
            --ocr-blue: #2458ff;
            --ocr-sky: #0ea5e9;
            --ocr-amber: #ffb020;
        }

        .stApp {
            background: #f3f7ff;
            color: var(--ocr-ink);
        }

        .block-container {
            max-width: 1580px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: var(--ocr-ink);
            letter-spacing: 0;
        }

        h1 {
            border-left: 0;
            border-top: 8px solid var(--ocr-blue);
            padding-top: 0.95rem;
            margin-top: 0.2rem;
            margin-bottom: 0.4rem;
            display: inline-block;
        }

        [data-testid="stSidebar"] {
            background: #edf3f8;
            border-right: 1px solid var(--ocr-line);
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #1d3557;
        }

        [data-testid="stMetric"] {
            background: var(--ocr-panel);
            border: 1px solid var(--ocr-line);
            border-left: 4px solid var(--ocr-amber);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            box-shadow: 0 8px 22px rgba(36, 88, 255, 0.08);
        }

        [data-testid="stFileUploader"] section {
            background: #eaf1ff;
            border-color: #9cb8ff;
            border-radius: 8px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            border-bottom: 1px solid var(--ocr-line);
        }

        .stTabs [data-baseweb="tab"] {
            background: #edf3f8;
            border-radius: 8px 8px 0 0;
            color: var(--ocr-ink);
        }

        .stTabs [aria-selected="true"] {
            background: #ffffff;
            color: var(--ocr-blue);
            border-top: 3px solid var(--ocr-amber);
        }

        [data-testid="stExpander"] {
            background: #ffffff;
            border: 1px solid var(--ocr-line);
            border-radius: 8px;
        }

        textarea {
            font-family: Consolas, "Courier New", monospace !important;
            line-height: 1.45 !important;
            background: #fbfdff !important;
            border: 1px solid #9cb8ff !important;
            border-radius: 8px !important;
            color: var(--ocr-ink) !important;
        }

        div.stDownloadButton > button {
            background: #ffffff;
            border: 1px solid #9cb8ff;
            border-radius: 8px;
            color: var(--ocr-blue);
            font-weight: 600;
        }

        div.stButton > button {
            background: var(--ocr-blue);
            border: 1px solid var(--ocr-blue);
            border-radius: 8px;
            color: #ffffff;
            font-weight: 700;
        }

        div.stButton > button:hover {
            background: #173fd2;
            border-color: #173fd2;
            color: #ffffff;
        }

        div.stDownloadButton > button:hover {
            background: #eaf1ff;
            border-color: var(--ocr-blue);
            color: var(--ocr-blue);
        }

        .ocr-section-label {
            margin: 1.5rem 0 0.5rem;
            color: var(--ocr-blue);
            font-size: 0.92rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0;
        }

        .ocr-note {
            color: var(--ocr-muted);
            font-size: 0.94rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def configure_tesseract(tesseract_path: Optional[str]) -> None:
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    elif Path(DEFAULT_TESSERACT_PATH).exists():
        pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT_PATH


@st.cache_resource(show_spinner=False)
def load_easyocr_reader():
    import easyocr

    return easyocr.Reader(["en"], gpu=False, verbose=False)


def decode_uploaded_image(uploaded_file) -> tuple[Optional[np.ndarray], Optional[str]]:
    raw_bytes = uploaded_file.getvalue()
    if not raw_bytes:
        return None, "The uploaded file is empty. Please upload a valid image file."

    file_bytes = np.frombuffer(raw_bytes, np.uint8)
    try:
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except cv2.error:
        return None, "OpenCV could not decode this file. Please upload a PNG, JPG, BMP, or TIFF image."

    if image is None:
        return None, "The uploaded file is not a readable image. Please try another image."
    return image, None


def select_recognition_region(
    image: np.ndarray,
    recognition_area: str,
    mouse_crop_box: Optional[tuple[int, int, int, int]],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    if recognition_area == "Mouse crop selection" and mouse_crop_box is not None:
        x1, y1, x2, y2 = mouse_crop_box
        if x2 - x1 < max(20, int(width * 0.05)):
            x1, x2 = 0, width
        if y2 - y1 < max(20, int(height * 0.05)):
            y1, y2 = 0, height
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        return image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

    regions = {
        "Full image": (0, 0, width, height),
        "Left page / left half": (0, 0, int(width * 0.62), height),
        "Right page / main page": (int(width * 0.25), 0, width, height),
        "Center page area": (
            int(width * 0.08),
            int(height * 0.04),
            int(width * 0.92),
            int(height * 0.96),
        ),
    }
    x1, y1, x2, y2 = regions.get(recognition_area, regions["Full image"])
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def crop_text_region(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    connected_text = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        connected_text,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    height, width = image.shape[:2]
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > width * height * 0.002 and w > 30 and h > 8:
            boxes.append((x, y, x + w, y + h))

    if not boxes:
        return image.copy(), (0, 0, width, height)

    x_padding = max(30, int(width * 0.04))
    y_padding = max(30, int(height * 0.04))
    x1 = max(min(box[0] for box in boxes) - x_padding, 0)
    y1 = max(min(box[1] for box in boxes) - y_padding, 0)
    x2 = min(max(box[2] for box in boxes) + x_padding, width)
    y2 = min(max(box[3] for box in boxes) + y_padding, height)

    crop_width = x2 - x1
    crop_height = y2 - y1
    if crop_width < width * 0.2 or crop_height < height * 0.05:
        return image.copy(), (0, 0, width, height)

    return image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def resize_for_ocr(
    image: np.ndarray,
    min_width: int = 1200,
    max_width: int = 1700,
) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = 1.0

    if width < min_width:
        scale = min(min_width / width, 3.0)
    elif width > max_width:
        scale = max_width / width

    if scale == 1.0:
        return image.copy(), scale

    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=interpolation,
    )
    return resized, scale


def sharpen_image(gray_image: np.ndarray, strength: str) -> np.ndarray:
    amount = 1.8 if strength == "Strong" else 1.5
    blur_weight = amount - 1.0
    blurred = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(gray_image, amount, blurred, -blur_weight, 0)


def normalize_background(gray_image: np.ndarray) -> np.ndarray:
    kernel_size = max(31, min(71, (gray_image.shape[1] // 35) | 1))
    background = cv2.medianBlur(gray_image, kernel_size)
    normalized = cv2.divide(gray_image, background, scale=255)
    return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)


def clean_text_components(binary_image: np.ndarray) -> np.ndarray:
    height, width = binary_image.shape[:2]
    foreground = cv2.bitwise_not(binary_image)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8,
    )

    clean_foreground = np.zeros_like(foreground)
    image_area = height * width
    max_component_height = max(18, int(height * 0.08))
    max_component_width = max(80, int(width * 0.35))
    max_component_area = max(200, int(image_area * 0.008))

    for label in range(1, component_count):
        x, y, w, h, area = stats[label]
        if area < 4:
            continue
        if h > max_component_height or w > max_component_width:
            continue
        if area > max_component_area:
            continue

        density = area / max(w * h, 1)
        if density > 0.9:
            continue

        clean_foreground[labels == label] = 255

    return cv2.bitwise_not(clean_foreground)


def normalize_angle(angle: float) -> float:
    while angle <= -45:
        angle += 90
    while angle > 45:
        angle -= 90
    return angle


def estimate_skew_by_projection(binary_inverse: np.ndarray) -> float:
    best_angle = 0.0
    best_score = -1.0

    for angle in np.arange(-15.0, 15.01, 0.5):
        rotated = rotate_image(binary_inverse, float(angle))
        row_projection = np.sum(rotated > 0, axis=1).astype(np.float32)
        score = float(np.sum(np.diff(row_projection) ** 2))
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    return best_angle


def estimate_skew_by_hough(binary_inverse: np.ndarray) -> Optional[float]:
    edges = cv2.Canny(binary_inverse, 50, 150, apertureSize=3)
    min_line_length = max(binary_inverse.shape[1] // 6, 80)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_line_length,
        maxLineGap=20,
    )

    if lines is None:
        return None

    weighted_angles: list[float] = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue

        angle = np.degrees(np.arctan2(dy, dx))
        angle = normalize_angle(float(angle))
        length = float(np.hypot(dx, dy))
        if abs(angle) <= 20 and length >= min_line_length:
            weighted_angles.extend([angle] * max(1, int(length // 50)))

    if len(weighted_angles) < 3:
        return None
    return -float(np.median(weighted_angles))


def estimate_skew_by_min_area(binary_inverse: np.ndarray) -> Optional[float]:
    coordinates = np.column_stack(np.where(binary_inverse > 0))

    if len(coordinates) < 100:
        return None

    angle = cv2.minAreaRect(coordinates)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) > 30:
        return None
    return float(angle)


def estimate_skew_angle(gray_image: np.ndarray) -> tuple[float, str]:
    _, binary = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3))
    text_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, horizontal_kernel)

    hough_angle = estimate_skew_by_hough(text_lines)
    if hough_angle is not None and abs(hough_angle) >= 0.3:
        return hough_angle, "Hough text-line detection"

    projection_angle = estimate_skew_by_projection(text_lines)
    if abs(projection_angle) >= 0.3:
        return projection_angle, "Projection profile search"

    min_area_angle = estimate_skew_by_min_area(binary)
    if min_area_angle is not None:
        return min_area_angle, "Minimum area rectangle"

    return 0.0, "No reliable skew detected"


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.3:
        return image.copy()

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def build_threshold_variants(gray_image: np.ndarray) -> dict[str, np.ndarray]:
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    normalized = normalize_background(gray_image)

    _, binary = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    _, normalized_binary = cv2.threshold(
        normalized,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    adaptive = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        13,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    text_mask = cv2.bitwise_not(adaptive)
    closed_text = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel)
    morph = cv2.bitwise_not(closed_text)
    component_cleaned = clean_text_components(normalized_binary)

    return {
        "Gaussian Blur": blurred,
        "Background Normalized": normalized,
        "Otsu Binary": binary,
        "Normalized Otsu Binary": normalized_binary,
        "Adaptive Threshold": adaptive,
        "Morphological Processing": morph,
        "Component Cleaned Text": component_cleaned,
    }


def is_valid_short_token(token: str) -> bool:
    stripped = token.strip(".,;:!?()'\"/-").lower()
    if not stripped:
        return False
    if stripped in VALID_SHORT_TOKENS:
        return True
    if any(character.isdigit() for character in stripped):
        return True
    if len(stripped) == 1 and stripped.upper() in {"A", "B", "C", "D", "E", "F", "T"}:
        return True
    return False


def format_word_boxes_as_layout(
    word_boxes: list[WordBox],
    image_width: int,
) -> str:
    boxes = [box for box in word_boxes if box.text.strip()]
    if not boxes:
        return ""

    heights = [max(1.0, box.bottom - box.top) for box in boxes]
    median_height = float(np.median(heights)) if heights else 12.0
    line_tolerance = max(8.0, median_height * 0.7)

    sorted_boxes = sorted(
        boxes,
        key=lambda box: ((box.top + box.bottom) / 2.0, box.left),
    )
    line_groups: list[dict[str, object]] = []

    for box in sorted_boxes:
        center_y = (box.top + box.bottom) / 2.0
        box_height = max(1.0, box.bottom - box.top)
        best_group = None
        best_distance = float("inf")

        for group in line_groups:
            group_center = float(group["center"])
            group_top = float(group["top"])
            group_bottom = float(group["bottom"])
            group_height = max(1.0, group_bottom - group_top)
            overlap = max(0.0, min(box.bottom, group_bottom) - max(box.top, group_top))
            overlap_ratio = overlap / max(min(box_height, group_height), 1.0)
            distance = abs(center_y - group_center)

            if overlap_ratio >= 0.35 or distance <= max(line_tolerance, group_height * 0.55):
                if distance < best_distance:
                    best_group = group
                    best_distance = distance

        if best_group is None:
            line_groups.append(
                {
                    "items": [box],
                    "center": center_y,
                    "top": box.top,
                    "bottom": box.bottom,
                }
            )
            continue

        items = best_group["items"]
        assert isinstance(items, list)
        items.append(box)
        best_group["top"] = min(float(best_group["top"]), box.top)
        best_group["bottom"] = max(float(best_group["bottom"]), box.bottom)
        best_group["center"] = float(
            np.mean([(item.top + item.bottom) / 2.0 for item in items])
        )

    char_widths = [
        max(3.0, (box.right - box.left) / max(len(box.text), 1))
        for box in boxes
        if box.right > box.left
    ]
    median_char_width = (
        float(np.median(char_widths))
        if char_widths
        else max(6.0, median_height * 0.45)
    )
    column_gap = max(median_height * 3.0, image_width * 0.04)

    output_lines: list[str] = []
    previous_bottom: Optional[float] = None
    for group in sorted(line_groups, key=lambda item: (float(item["top"]), float(item["center"]))):
        items = group["items"]
        assert isinstance(items, list)
        items = sorted(items, key=lambda item: item.left)
        line_parts: list[str] = []
        last_right: Optional[float] = None

        for item in items:
            text = re.sub(r"\s+", " ", item.text).strip()
            if not text:
                continue

            if last_right is None:
                line_parts.append(text)
            else:
                gap = item.left - last_right
                spaces = 1
                if gap > median_char_width * 1.5:
                    spaces = min(
                        10,
                        max(2, int(round(gap / max(median_char_width * 2.0, 1.0)))),
                    )
                if gap > column_gap:
                    spaces = max(spaces, 5)
                line_parts.append(" " * spaces + text)
            last_right = max(last_right if last_right is not None else item.right, item.right)

        line_text = "".join(line_parts).strip()
        if not line_text:
            continue

        group_top = float(group["top"])
        group_bottom = float(group["bottom"])
        if (
            previous_bottom is not None
            and group_top - previous_bottom > median_height * 1.7
        ):
            output_lines.append("")

        output_lines.append(line_text)
        previous_bottom = max(previous_bottom or group_bottom, group_bottom)

    return "\n".join(output_lines)


def text_to_plain_paragraphs(text: str) -> str:
    paragraphs: list[str] = []
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line:
            current_lines.append(line)
            continue
        if current_lines:
            paragraphs.append(" ".join(current_lines))
            current_lines = []

    if current_lines:
        paragraphs.append(" ".join(current_lines))

    return "\n\n".join(paragraphs)


def preprocess_image(
    image: np.ndarray,
    enhancement_level: str,
    manual_rotation: float,
    rotation_search_enabled: bool,
    recognition_area: str,
    mouse_crop_box: Optional[tuple[int, int, int, int]],
    use_auto_skew: bool,
) -> PreprocessResult:
    """Improve document quality before OCR and keep each step for display."""
    region_image, region_box = select_recognition_region(
        image,
        recognition_area,
        mouse_crop_box,
    )
    cropped, crop_box = crop_text_region(region_image)
    resized, scale = resize_for_ocr(cropped)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    denoise_strength = 14 if enhancement_level == "Strong" else 10
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    if enhancement_level == "Strong":
        denoised = cv2.fastNlMeansDenoising(
            denoised,
            None,
            h=denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    sharpened = sharpen_image(contrast, enhancement_level)

    if use_auto_skew:
        skew_angle, skew_method = estimate_skew_angle(sharpened)
    else:
        skew_angle, skew_method = 0.0, "Manual rotation only"
    applied_rotation = skew_angle + manual_rotation
    deskewed = rotate_image(sharpened, applied_rotation)
    base_variants = build_threshold_variants(deskewed)
    blurred = base_variants["Gaussian Blur"]
    normalized = base_variants["Background Normalized"]
    binary = base_variants["Otsu Binary"]
    normalized_binary = base_variants["Normalized Otsu Binary"]
    adaptive = base_variants["Adaptive Threshold"]
    morph = base_variants["Morphological Processing"]
    component_cleaned = base_variants["Component Cleaned Text"]

    steps = {
        "Selected Recognition Area": cv2.cvtColor(region_image, cv2.COLOR_BGR2RGB),
        "Cropped Text Region": cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB),
        "Resized Image": cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
        "Gray Image": gray,
        "Denoised Image": denoised,
        "Contrast Enhanced": contrast,
        "Sharpened Image": sharpened,
        "Deskewed Image": deskewed,
        "Gaussian Blur": blurred,
        "Background Normalized": normalized,
        "Otsu Binary": binary,
        "Normalized Otsu Binary": normalized_binary,
        "Adaptive Threshold": adaptive,
        "Morphological Processing": morph,
        "Component Cleaned Text": component_cleaned,
    }
    candidates = {
        "Applied Rotation Original Color": cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
        "Applied Rotation Enhanced Grayscale": deskewed,
        "Applied Rotation Background Normalized": normalized,
        "Applied Rotation Component Cleaned Text": component_cleaned,
        "Applied Rotation Normalized Otsu Binary": normalized_binary,
        "Applied Rotation Otsu Binary": binary,
        "Applied Rotation Adaptive Threshold": adaptive,
        "Applied Rotation Morphological Processing": morph,
    }

    if rotation_search_enabled:
        for offset in (-6, 6):
            candidate_angle = applied_rotation + offset
            rotated = rotate_image(sharpened, candidate_angle)
            variants = build_threshold_variants(rotated)
            candidates[f"Rotation {candidate_angle:+.1f} Enhanced Grayscale"] = rotated
            candidates[f"Rotation {candidate_angle:+.1f} Background Normalized"] = variants[
                "Background Normalized"
            ]
            candidates[f"Rotation {candidate_angle:+.1f} Component Cleaned Text"] = variants[
                "Component Cleaned Text"
            ]
            candidates[f"Rotation {candidate_angle:+.1f} Otsu Binary"] = variants[
                "Otsu Binary"
            ]
            candidates[f"Rotation {candidate_angle:+.1f} Adaptive Threshold"] = variants[
                "Adaptive Threshold"
            ]

    return PreprocessResult(
        steps=steps,
        ocr_candidates=candidates,
        display_image=normalized,
        scale=scale,
        recognition_area=recognition_area,
        region_box=region_box,
        crop_box=crop_box,
        skew_angle=skew_angle,
        applied_rotation=applied_rotation,
        skew_method=skew_method,
        rotation_search_enabled=rotation_search_enabled,
    )


def tesseract_config(psm: str) -> str:
    return (
        f"--oem 3 --psm {psm} "
        "-c preserve_interword_spaces=1 "
        "-c user_defined_dpi=300"
    )


def recognize_text(processed_image: np.ndarray, psm: str) -> str:
    return pytesseract.image_to_string(
        processed_image,
        lang="eng",
        config=tesseract_config(psm),
    )


def recognize_text_with_confidence(processed_image: np.ndarray, psm: str) -> tuple[str, float]:
    if processed_image.ndim == 3:
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)

    data = pytesseract.image_to_data(
        processed_image,
        lang="eng",
        config=tesseract_config(psm),
        output_type=pytesseract.Output.DICT,
    )
    word_boxes: list[WordBox] = []
    confidences: list[float] = []

    for index, (word, confidence) in enumerate(zip(data["text"], data["conf"])):
        word = word.strip()
        try:
            confidence_value = float(confidence)
        except ValueError:
            continue

        if word:
            word_boxes.append(
                WordBox(
                    text=word,
                    confidence=confidence_value,
                    left=float(data["left"][index]),
                    top=float(data["top"][index]),
                    right=float(data["left"][index] + data["width"][index]),
                    bottom=float(data["top"][index] + data["height"][index]),
                )
            )
        if word and confidence_value >= 0:
            confidences.append(confidence_value)

    text = format_word_boxes_as_layout(word_boxes, processed_image.shape[1])
    mean_confidence = float(np.mean(confidences)) if confidences else 0.0
    return text, mean_confidence


def recognize_with_easyocr(image: np.ndarray) -> tuple[str, float]:
    reader = load_easyocr_reader()
    if image.ndim == 2:
        easyocr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        easyocr_image = image

    detections = reader.readtext(
        easyocr_image,
        detail=1,
        paragraph=False,
        decoder="greedy",
        text_threshold=0.4,
        low_text=0.25,
        link_threshold=0.4,
        width_ths=1.2,
        add_margin=0.05,
    )
    if not detections:
        return "", 0.0

    word_boxes: list[WordBox] = []
    confidences: list[float] = []
    for box, text, confidence in detections:
        text = text.strip()
        if not text:
            continue

        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        confidence_value = float(confidence) * 100 if confidence is not None else 0.0
        word_boxes.append(
            WordBox(
                text=text,
                confidence=confidence_value,
                left=min(xs),
                top=min(ys),
                right=max(xs),
                bottom=max(ys),
            )
        )
        if confidence is not None:
            confidences.append(confidence_value)

    confidence = float(np.mean(confidences)) if confidences else 0.0
    return format_word_boxes_as_layout(word_boxes, easyocr_image.shape[1]), confidence


def score_ocr_text(text: str) -> float:
    cleaned = text.strip()
    if not cleaned:
        return 0.0

    words = re.findall(r"[A-Za-z]{2,}", cleaned)
    tokens = re.findall(r"[A-Za-z0-9@.']+", cleaned)
    letters = sum(character.isalpha() for character in cleaned)
    digits = sum(character.isdigit() for character in cleaned)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    useful_punctuation = sum(character in ".,;:!?()[]{}'\"-/&%" for character in cleaned)
    punctuation = sum(character in ".,;:!?()[]{}'\"-/&%`~|\\_^*" for character in cleaned)
    unusual = sum(
        not (
            character.isalnum()
            or character.isspace()
            or character in ".,;:!?()[]{}'\"-/&%"
        )
        for character in cleaned
    )
    total_chars = max(len(cleaned), 1)
    unusual_ratio = unusual / total_chars
    punctuation_ratio = punctuation / total_chars
    short_or_broken_tokens = sum(
        1
        for token in tokens
        if len(token) <= 2 and not is_valid_short_token(token)
    )
    known_word_hits = sum(1 for word in words if word.lower() in COMMON_ENGLISH_WORDS)
    line_bonus = sum(12 for line in lines if len(line.split()) >= 4)
    broken_line_penalty = sum(8 for line in lines if 0 < len(line.split()) <= 2)
    natural_text_bonus = known_word_hits * 18
    noise_penalty = (
        unusual * 18
        + punctuation * 2
        + short_or_broken_tokens * 14
        + unusual_ratio * 900
        + max(0.0, punctuation_ratio - 0.08) * 800
    )
    return (
        letters * 2
        + digits
        + useful_punctuation * 0.5
        + len(words) * 8
        + line_bonus
        + natural_text_bonus
        - broken_line_penalty
        - noise_penalty
    )


def build_ocr_result(
    text: str,
    engine: str,
    source: str,
    confidence: float,
) -> OCRResult:
    text = clean_ocr_text(text)
    word_count = len(re.findall(r"[A-Za-z0-9]{2,}", text))
    score = confidence * 6 + score_ocr_text(text)
    return OCRResult(
        text=text,
        engine=engine,
        source=source,
        score=score,
        confidence=confidence,
        word_count=word_count,
    )


def clean_ocr_text(text: str) -> str:
    cleaned_lines = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        line = re.sub(r"[^A-Za-z0-9@.,;:!?()'\"/\-\s]", " ", raw_line)
        line = re.sub(r"\s+", " ", line).strip()
        tokens = []
        for token in line.split():
            alpha_count = sum(character.isalpha() for character in token)
            digit_count = sum(character.isdigit() for character in token)
            if len(token) == 1 and not is_valid_short_token(token):
                continue
            if alpha_count == 0 and digit_count == 0:
                continue
            if len(token) >= 4 and alpha_count == 1 and digit_count == 0:
                continue
            tokens.append(token)

        if tokens:
            cleaned_lines.append(" ".join(tokens))

    return "\n".join(cleaned_lines).strip()


def recognize_best_text(
    candidates: dict[str, np.ndarray],
    psm: str,
    try_multiple_variants: bool,
    deep_search: bool,
) -> OCRResult:
    default_candidate_name = "Applied Rotation Background Normalized"
    if not try_multiple_variants:
        text, confidence = recognize_text_with_confidence(
            candidates[default_candidate_name],
            psm=psm,
        )
        return build_ocr_result(
            text,
            engine=f"Tesseract PSM {psm}",
            source=default_candidate_name,
            confidence=confidence,
        )

    best_result = build_ocr_result(
        "",
        engine=f"Tesseract PSM {psm}",
        source=default_candidate_name,
        confidence=0.0,
    )
    best_candidate_image = None
    best_candidate_psm = psm
    if deep_search:
        psm_values = list(dict.fromkeys([psm, "6", "3", "4"]))
        candidate_items = list(candidates.items())
    else:
        psm_values = [psm]
        preferred_names = [
            "Applied Rotation Original Color",
            "Applied Rotation Enhanced Grayscale",
            "Applied Rotation Background Normalized",
            "Applied Rotation Component Cleaned Text",
            "Applied Rotation Normalized Otsu Binary",
        ]
        candidate_items = [
            (name, candidates[name]) for name in preferred_names if name in candidates
        ]

    for candidate_name, candidate_image in candidate_items:
        for current_psm in psm_values:
            text, confidence = recognize_text_with_confidence(candidate_image, psm=current_psm)
            result = build_ocr_result(
                text,
                engine=f"Tesseract PSM {current_psm}",
                source=candidate_name,
                confidence=confidence,
            )
            if result.score > best_result.score:
                best_result = result
                best_candidate_image = candidate_image
                best_candidate_psm = current_psm

    if best_candidate_image is not None:
        raw_tesseract_text = clean_ocr_text(
            recognize_text(best_candidate_image, psm=best_candidate_psm)
        )
        if (
            raw_tesseract_text.strip()
            and score_ocr_text(raw_tesseract_text) > score_ocr_text(best_result.text) + 120
        ):
            best_result.text = raw_tesseract_text
            best_result.word_count = len(re.findall(r"[A-Za-z0-9]{2,}", raw_tesseract_text))
            best_result.score = best_result.confidence * 6 + score_ocr_text(raw_tesseract_text)

    return best_result


def run_automation_ocr(
    preprocessing: PreprocessResult,
    psm: str,
    use_machine_learning: bool,
    try_multiple_variants: bool,
    deep_search: bool,
) -> OCRResult:
    results = [
        recognize_best_text(
            preprocessing.ocr_candidates,
            psm,
            try_multiple_variants,
            deep_search,
        )
    ]

    ml_error = None
    if use_machine_learning:
        ml_sources = {
            "ML Source: Applied Rotation Original Color": preprocessing.ocr_candidates[
                "Applied Rotation Original Color"
            ]
        }
        if deep_search:
            ml_sources[
                "ML Source: Applied Rotation Enhanced Grayscale"
            ] = preprocessing.ocr_candidates["Applied Rotation Enhanced Grayscale"]
            ml_sources[
                "ML Source: Applied Rotation Background Normalized"
            ] = preprocessing.ocr_candidates["Applied Rotation Background Normalized"]

        if deep_search and preprocessing.rotation_search_enabled:
            for source_name, candidate in preprocessing.ocr_candidates.items():
                if "Enhanced Grayscale" in source_name and "Rotation" in source_name:
                    ml_sources[f"ML Source: {source_name}"] = candidate

        try:
            for source_name, candidate in ml_sources.items():
                text, confidence = recognize_with_easyocr(candidate)
                results.append(
                    build_ocr_result(
                        text,
                        engine="EasyOCR ML model",
                        source=source_name,
                        confidence=confidence,
                    )
                )
        except Exception as error:
            ml_error = str(error)

    best_result = max(results, key=lambda result: result.score)
    if ml_error:
        best_result.source = f"{best_result.source} | ML fallback: {ml_error[:120]}"
    return best_result


def render_project_overview() -> None:
    st.title(PROJECT_NAME)
    st.caption("English printed text recognition system for course project demonstration.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Project Type", PROJECT_TYPE)
    col2.metric("Project Area", "Computer Vision")
    col3.metric("Output", "Editable Text")

    with st.expander("Project overview"):
        st.markdown(
            f"""
            **Project Field:** {PROJECT_AREA}

            **Technology Stack:** {TECH_STACK}

            This project recognizes printed English text from document images. It uses
            OpenCV to resize, enhance, denoise, sharpen, deskew, and binarize images
            before applying Tesseract OCR and an optional EasyOCR machine learning model.

            **Expected Outcomes:** A machine learning and automation-style tool that
            helps transform paper document images into editable digital text.
            """
        )


def render_sidebar() -> tuple[
    str,
    str,
    str,
    str,
    bool,
    float,
    bool,
    bool,
    bool,
    bool,
]:
    st.sidebar.header("OCR Settings")
    tesseract_path = st.sidebar.text_input(
        "Tesseract executable path",
        value="",
        placeholder=DEFAULT_TESSERACT_PATH,
        help="Leave empty if Tesseract is already available in your system PATH.",
    )
    enhancement_level = st.sidebar.selectbox(
        "Image enhancement level",
        options=["Auto System", "Balanced", "Strong"],
        index=0,
        help="Use Strong for blurrier, darker, or noisier photos.",
    )
    recognition_mode = st.sidebar.selectbox(
        "Recognition speed",
        options=["Fast", "Balanced", "Thorough"],
        index=0,
        help="Fast is recommended for interactive use. Thorough enables slower OCR searches.",
    )
    recognition_area = st.sidebar.selectbox(
        "Recognition area",
        options=[
            "Full image",
            "Right page / main page",
            "Left page / left half",
            "Center page area",
        ],
        index=0,
        help="For book photos, choose the page that actually contains the target text.",
    )
    psm = st.sidebar.selectbox(
        "Page segmentation mode",
        options=["6", "3", "4", "11", "12"],
        index=0,
        help="PSM 6 works well for a uniform block of printed text.",
    )
    use_auto_skew = st.sidebar.checkbox(
        "Use automatic skew correction",
        value=True,
        help="Turn this off when you want the manual rotation slider to control the angle directly.",
    )
    if "manual_rotation_angle" not in st.session_state:
        st.session_state["manual_rotation_angle"] = 0.0

    st.sidebar.caption("Rotation joystick")
    nudge_cols = st.sidebar.columns(5)
    if nudge_cols[0].button("-5"):
        st.session_state["manual_rotation_angle"] = max(
            -30.0,
            st.session_state["manual_rotation_angle"] - 5.0,
        )
    if nudge_cols[1].button("-1"):
        st.session_state["manual_rotation_angle"] = max(
            -30.0,
            st.session_state["manual_rotation_angle"] - 1.0,
        )
    if nudge_cols[2].button("0"):
        st.session_state["manual_rotation_angle"] = 0.0
    if nudge_cols[3].button("+1"):
        st.session_state["manual_rotation_angle"] = min(
            30.0,
            st.session_state["manual_rotation_angle"] + 1.0,
        )
    if nudge_cols[4].button("+5"):
        st.session_state["manual_rotation_angle"] = min(
            30.0,
            st.session_state["manual_rotation_angle"] + 5.0,
        )

    manual_rotation = st.sidebar.slider(
        "Manual rotation angle",
        min_value=-30.0,
        max_value=30.0,
        step=0.5,
        key="manual_rotation_angle",
        help="Positive values rotate counter-clockwise; negative values rotate clockwise.",
    )
    if recognition_mode == "Fast":
        rotation_search_enabled = False
        try_multiple_variants = False
        use_machine_learning = False
        deep_search = False
    elif recognition_mode == "Balanced":
        rotation_search_enabled = False
        try_multiple_variants = True
        use_machine_learning = False
        deep_search = False
    else:
        rotation_search_enabled = True
        try_multiple_variants = True
        use_machine_learning = True
        deep_search = False

    with st.sidebar.expander("Advanced OCR switches"):
        st.caption("These options can improve difficult images but may slow recognition.")
        rotation_search_enabled = st.checkbox(
            "Search rotated candidates",
            value=rotation_search_enabled,
            help="Tests several rotation angles and lets OCR confidence choose the best one.",
        )
        try_multiple_variants = st.checkbox(
            "Try multiple OCR variants",
            value=try_multiple_variants,
            help="Runs OCR on several preprocessed versions and keeps the best-looking result.",
        )
        use_machine_learning = st.checkbox(
            "Use machine learning OCR",
            value=use_machine_learning,
            help="EasyOCR can help some photos but is much slower on CPU.",
        )
        deep_search = st.checkbox(
            "Deep search mode",
            value=deep_search,
            help="Slowest mode that tests more OCR variants and page segmentation settings.",
        )

    st.sidebar.markdown(
        """
        **Preprocessing Pipeline**

        1. Recognition area selection
        2. Mouse crop selection
        3. Manual or automatic rotation
        4. Resize for OCR
        5. Grayscale conversion
        6. Denoising and contrast enhancement
        7. Sharpening for mild blur
        8. Tesseract OCR candidate scoring
        9. ML OCR candidate scoring
        10. Layout-preserving text output
        """
    )
    return (
        psm,
        tesseract_path,
        enhancement_level,
        recognition_area,
        use_auto_skew,
        manual_rotation,
        rotation_search_enabled,
        try_multiple_variants,
        use_machine_learning,
        deep_search,
    )


def draw_region_overlay(
    image: np.ndarray,
    region_box: tuple[int, int, int, int],
) -> np.ndarray:
    preview = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = preview.shape[:2]
    x1, y1, x2, y2 = region_box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    overlay = preview.copy()
    overlay[:, :] = (36, 88, 255)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    outside = mask == 0
    preview[outside] = cv2.addWeighted(
        preview,
        0.42,
        overlay,
        0.58,
        0,
    )[outside]

    thickness = max(3, width // 300)
    cv2.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), (255, 176, 32), thickness)
    return preview


def render_uploaded_image(
    image: np.ndarray,
    region_box: Optional[tuple[int, int, int, int]] = None,
) -> None:
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if region_box is None:
        st.image(original_rgb, caption="Original Uploaded Image", width="stretch")
        return

    preview = draw_region_overlay(image, region_box)
    st.image(preview, caption="Original Image with OCR Region", width="stretch")


def crop_region_for_preview(
    image: np.ndarray,
    region_box: tuple[int, int, int, int],
) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = region_box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return image[y1:y2, x1:x2].copy()


def render_rotation_preview(
    image: np.ndarray,
    region_box: tuple[int, int, int, int],
    manual_rotation: float,
) -> None:
    selected_region = crop_region_for_preview(image, region_box)
    rotated_region = rotate_image(selected_region, manual_rotation)
    rotated_rgb = cv2.cvtColor(rotated_region, cv2.COLOR_BGR2RGB)
    st.image(
        rotated_rgb,
        caption=f"OCR Area Rotation Preview ({manual_rotation:+.1f} degrees)",
        width="stretch",
    )


def render_mouse_crop_selector(
    image: np.ndarray,
    enabled: bool,
    upload_key: str,
) -> Optional[tuple[int, int, int, int]]:
    if not enabled:
        return None

    st.subheader("Mouse Select OCR Area")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    width, height = pil_image.size
    default_coords = (
        int(width * 0.08),
        int(width * 0.92),
        int(height * 0.08),
        int(height * 0.92),
    )
    _, crop_box = st_cropper(
        pil_image,
        realtime_update=True,
        default_coords=default_coords,
        box_color="#2458ff",
        return_type="both",
        key=f"ocr_mouse_crop_{upload_key}",
        stroke_width=3,
    )

    left = int(crop_box["left"])
    top = int(crop_box["top"])
    right = left + int(crop_box["width"])
    bottom = top + int(crop_box["height"])
    st.caption(
        f"Selected OCR box: left {left}, top {top}, right {right}, bottom {bottom}"
    )
    return (left, top, right, bottom)


def render_preprocessing_results(result: PreprocessResult) -> None:
    st.image(
        result.display_image,
        caption="Final Preprocessed Image",
        width="stretch",
        clamp=True,
    )
    st.caption(
        f"Recognition area: {result.recognition_area} | Region box: "
        f"{result.region_box} | Crop box: {result.crop_box} | "
        f"Scale: {result.scale:.2f}x | "
        f"Estimated skew correction: "
        f"{result.skew_angle:.2f} degrees | Applied rotation: "
        f"{result.applied_rotation:.2f} degrees | Method: {result.skew_method}"
    )

    with st.expander("View preprocessing steps"):
        for step_name, step_image in result.steps.items():
            st.image(step_image, caption=step_name, width="stretch", clamp=True)


def render_automation_report(result: OCRResult, preprocessing: PreprocessResult) -> None:
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Selected Engine", result.engine)
    metric_col2.metric("Confidence", f"{result.confidence:.1f}")
    metric_col3.metric("Words", str(result.word_count))
    metric_col4.metric("Automation Score", f"{result.score:.1f}")

    st.markdown(
        f"""
        **Automation Decision:** The system selected `{result.engine}` using
        `{result.source}` after preprocessing, rotation correction, candidate
        generation, and OCR scoring.

        **Processing Summary:** recognition area `{preprocessing.recognition_area}`,
        region `{preprocessing.region_box}`, crop `{preprocessing.crop_box}`,
        scale `{preprocessing.scale:.2f}x`, applied rotation
        `{preprocessing.applied_rotation:.2f}` degrees.
        """
    )


def build_report_text(result: OCRResult) -> str:
    return (
        f"Project: {PROJECT_NAME}\n"
        f"Selected engine: {result.engine}\n"
        f"Selected source: {result.source}\n"
        f"Confidence: {result.confidence:.1f}\n"
        f"Automation score: {result.score:.1f}\n"
        f"Word count: {result.word_count}\n\n"
        f"Formatted layout output:\n{result.text}\n\n"
        f"Plain text output:\n{text_to_plain_paragraphs(result.text)}"
    )


def render_copy_button(text: str, label: str, element_id: str) -> None:
    status_id = f"{element_id}_status"
    text_json = json.dumps(text)
    label_json = json.dumps(label)
    element_json = json.dumps(element_id)
    status_json = json.dumps(status_id)
    components.html(
        f"""
        <div style="display:flex;align-items:center;gap:10px;margin:0 0 10px 0;">
            <button id={element_json}
                style="
                    background:#2458ff;
                    border:1px solid #2458ff;
                    border-radius:8px;
                    color:#ffffff;
                    cursor:pointer;
                    font:600 14px Arial, sans-serif;
                    min-height:38px;
                    padding:0 16px;
                ">
            </button>
            <span id={status_json}
                style="color:#ffb020;font:600 13px Arial, sans-serif;"></span>
        </div>
        <script>
        const copyText = {text_json};
        const button = document.getElementById({element_json});
        const status = document.getElementById({status_json});
        button.textContent = {label_json};

        async function copyToClipboard() {{
            try {{
                if (navigator.clipboard && window.isSecureContext) {{
                    await navigator.clipboard.writeText(copyText);
                }} else {{
                    const textarea = document.createElement("textarea");
                    textarea.value = copyText;
                    textarea.style.position = "fixed";
                    textarea.style.opacity = "0";
                    document.body.appendChild(textarea);
                    textarea.focus();
                    textarea.select();
                    document.execCommand("copy");
                    textarea.remove();
                }}
                status.textContent = "Copied";
                window.setTimeout(() => status.textContent = "", 1800);
            }} catch (error) {{
                status.textContent = "Copy failed, select text manually";
            }}
        }}

        button.addEventListener("click", copyToClipboard);
        </script>
        """,
        height=54,
    )


def render_result_panel(
    ocr_result: OCRResult,
    preprocessing: PreprocessResult,
    uploaded_file_name: str,
) -> None:
    render_automation_report(ocr_result, preprocessing)

    cleaned_text = ocr_result.text.strip()
    if cleaned_text:
        plain_text = text_to_plain_paragraphs(cleaned_text)
        layout_tab, plain_tab = st.tabs(["Formatted Layout", "Plain Text"])
        with layout_tab:
            render_copy_button(cleaned_text, "Copy Formatted Text", "copy_formatted_text")
            st.text_area("OCR Output", value=cleaned_text, height=520)
        with plain_tab:
            render_copy_button(plain_text, "Copy Plain Text", "copy_plain_text")
            st.text_area("OCR Output", value=plain_text, height=420)
    else:
        st.warning("No text was recognized. Try a clearer image or another PSM value.")
        st.text_area("OCR Output", value="", height=260)

    output_name = f"{Path(uploaded_file_name).stem}_ocr_result.txt"
    with st.expander("Download backup"):
        st.download_button(
            label="Download Result as TXT",
            data=build_report_text(ocr_result),
            file_name=output_name,
            mime="text/plain",
        )


def main() -> None:
    configure_page()
    render_project_overview()

    (
        psm,
        tesseract_path,
        enhancement_level,
        recognition_area,
        use_auto_skew,
        manual_rotation,
        rotation_search_enabled,
        try_multiple_variants,
        use_machine_learning,
        deep_search,
    ) = render_sidebar()
    configure_tesseract(tesseract_path.strip() or None)

    uploaded_file = st.file_uploader(
        "Upload an image containing printed English text",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    )

    if uploaded_file is None:
        st.info("Upload a document image to start recognition.")
        return

    if getattr(uploaded_file, "size", 0) == 0:
        st.error("The uploaded file is empty. Please upload a valid image file.")
        return

    image, decode_error = decode_uploaded_image(uploaded_file)
    if decode_error:
        st.error(decode_error)
        return

    use_mouse_crop = st.toggle(
        "Select OCR area with mouse",
        value=False,
        help="Turn this on, drag a box on the image, then click Run OCR.",
    )
    upload_key = f"{Path(uploaded_file.name).stem}_{getattr(uploaded_file, 'size', 0)}"
    mouse_crop_box = render_mouse_crop_selector(image, use_mouse_crop, upload_key)
    effective_recognition_area = (
        "Mouse crop selection" if use_mouse_crop and mouse_crop_box else recognition_area
    )

    preview_box = mouse_crop_box
    if preview_box is None:
        _, preview_box = select_recognition_region(image, recognition_area, None)

    preview_col, rotation_preview_col = st.columns(2, gap="large")
    with preview_col:
        st.subheader("Original Image Preview")
        render_uploaded_image(image, preview_box)
    with rotation_preview_col:
        st.subheader("Rotation Preview")
        render_rotation_preview(image, preview_box, manual_rotation)

    if not st.button("Run OCR", type="primary", use_container_width=True):
        st.info("Adjust the crop area and rotation angle, then click Run OCR.")
        return

    preprocessing = preprocess_image(
        image,
        enhancement_level,
        manual_rotation,
        rotation_search_enabled=rotation_search_enabled and try_multiple_variants,
        recognition_area=effective_recognition_area,
        mouse_crop_box=mouse_crop_box,
        use_auto_skew=use_auto_skew,
    )

    try:
        with st.spinner("Running automated OCR system and ML candidate selection..."):
            ocr_result = run_automation_ocr(
                preprocessing,
                psm=psm,
                use_machine_learning=use_machine_learning,
                try_multiple_variants=try_multiple_variants,
                deep_search=deep_search,
            )
    except TesseractNotFoundError:
        st.error(
            "Tesseract OCR was not found. Install Tesseract OCR or enter the full "
            "path to tesseract.exe in the sidebar."
        )
        return
    except TesseractError as error:
        st.error(f"Tesseract OCR failed: {error}")
        return

    st.markdown(
        '<div class="ocr-section-label">Document and result comparison</div>',
        unsafe_allow_html=True,
    )
    original_col, result_col = st.columns([0.95, 1.05], gap="large")
    with original_col:
        st.subheader("Original Image")
        render_uploaded_image(image, preprocessing.region_box)
    with result_col:
        st.subheader("Recognized Text Result")
        render_result_panel(ocr_result, preprocessing, uploaded_file.name)

    st.markdown(
        '<div class="ocr-section-label">Image processing preview</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Preprocessed Image")
    render_preprocessing_results(preprocessing)


if __name__ == "__main__":
    main()
