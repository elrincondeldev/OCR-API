"""
Step 1 — Ingestion & Preprocessing.

Converts an uploaded PDF or image file into a cleaned, binarised JPEG
suitable for high-accuracy OCR.  All heavy image operations run in a
thread-pool executor so they do not block the event loop.
"""

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
    """Convert the first page of a PDF to PNG bytes (runs in thread pool)."""
    pages = convert_from_bytes(file_bytes, dpi=300, first_page=1, last_page=1)
    if not pages:
        raise ValueError("pdf2image returned no pages from the uploaded PDF.")

    buffer = io.BytesIO()
    pages[0].save(buffer, format="PNG")
    return buffer.getvalue()


def _apply_opencv_preprocessing(image_bytes: bytes) -> bytes:
    """
    Apply grayscale conversion and adaptive thresholding to remove grease
    stains, shadows, and uneven lighting.  Returns cleaned JPEG bytes.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("OpenCV could not decode the image buffer.")

    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Denoise with a fast Gaussian blur before thresholding
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. Adaptive binarisation — handles uneven illumination well
    binary = cv2.adaptiveThreshold(
        denoised,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,   # neighbourhood size — must be odd
        C=15,           # constant subtracted from the mean
    )

    # 4. Morphological opening to remove small noise specks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 5. Encode back to JPEG (Document AI accepts JPEG/PNG/TIFF)
    success, buffer = cv2.imencode(".jpg", cleaned, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise RuntimeError("OpenCV could not encode the processed image.")

    return buffer.tobytes()


async def preprocess_upload(file_bytes: bytes, content_type: str) -> bytes:
    """
    Public async entry-point consumed by the route handler.

    1. If PDF → render first page to PNG bytes.
    2. Apply OpenCV grayscale + adaptive binarisation.
    3. Return cleaned JPEG bytes ready for Document AI.
    """
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
