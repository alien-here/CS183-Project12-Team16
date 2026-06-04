from pathlib import Path

import cv2
import numpy as np

##Author:Zhu Ruichen
#Function:load_image
def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image

##Author:Zhu Ruichen
#Function:load_image
def resize_if_large(image: np.ndarray, max_width: int = 1800) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= max_width:
        return image

    scale = max_width / width
    new_size = (max_width, int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

##Author:Zhu Ruichen
#Function:estimate_skew_angle
def estimate_skew_angle(binary_image: np.ndarray) -> float:
    inverted = cv2.bitwise_not(binary_image)
    coordinates = np.column_stack(np.where(inverted > 0))

    if len(coordinates) < 20:
        return 0.0

    angle = cv2.minAreaRect(coordinates)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return angle

##Author:Zhu Ruichen
#Function:rotate_image
def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.5:
        return image

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

##Author:Zhu Ruichen
#Function:preprocess_document
def preprocess_document(image_path: str) -> np.ndarray:
    image = load_image(image_path)
    image = resize_if_large(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    rough_binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    angle = estimate_skew_angle(rough_binary)
    deskewed = rotate_image(gray, angle)

    processed = cv2.adaptiveThreshold(
        deskewed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )

    kernel = np.ones((1, 1), np.uint8)
    return cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)

##Author:Zhu Ruichen
#Function:save_image
def save_image(image: np.ndarray, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)

