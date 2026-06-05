from typing import Optional

import numpy as np
import pytesseract

#Author:Wu Xuran
#Function:configure_
def configure_tesseract(tesseract_path: Optional[str] = None) -> None:
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

#Author:Wu Xuran
#Function:configure_
def recognize_text(image: np.ndarray, psm: str = "6") -> str:
    config = f"--oem 3 --psm {psm} -l eng"
    return pytesseract.image_to_string(image, config=config)
