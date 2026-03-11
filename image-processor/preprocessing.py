import io
import logging
import asyncio
from functools import partial

import cv2
import numpy as np
from pdf2image import convert_from_bytes

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/tiff"}
SUPPORTED_PDF_TYPE = "application/pdf"


def _pdf_to_image_bytes(file_bytes: bytes) -> bytes:
    pages = convert_from_bytes(file_bytes, dpi=300, first_page=1, last_page=1)
    if not pages:
        raise ValueError("pdf2image returned no pages from the uploaded PDF.")

    buffer = io.BytesIO()
    pages[0].save(buffer, format="PNG")
    return buffer.getvalue()


def _apply_opencv_preprocessing(image_bytes: bytes) -> bytes:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("OpenCV could not decode the image buffer.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.adaptiveThreshold(
        denoised,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    success, buffer = cv2.imencode(".jpg", cleaned, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise RuntimeError("OpenCV could not encode the processed image.")

    return buffer.tobytes()


async def preprocess_upload(file_bytes: bytes, content_type: str) -> bytes:
    loop = asyncio.get_running_loop()

    if content_type == SUPPORTED_PDF_TYPE:
        logger.info("PDF detected — converting first page to image.")
        image_bytes = await loop.run_in_executor(
            None, partial(_pdf_to_image_bytes, file_bytes)
        )
    elif content_type in SUPPORTED_IMAGE_TYPES:
        image_bytes = file_bytes
    else:
        raise ValueError(
            f"Unsupported content type '{content_type}'. "
            f"Accepted: PDF, JPEG, PNG, TIFF."
        )

    logger.info("Applying OpenCV preprocessing (grayscale + adaptive threshold).")
    cleaned_bytes = await loop.run_in_executor(
        None, partial(_apply_opencv_preprocessing, image_bytes)
    )

    logger.info("Preprocessing complete. Image size: %d bytes.", len(cleaned_bytes))
    return cleaned_bytes
