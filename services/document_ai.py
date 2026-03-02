"""
Step 2 — Google Document AI OCR.

Sends the preprocessed image to a Document OCR processor, then parses
the response to extract per-token confidence scores.  Any line that
contains at least one token below CONFIDENCE_THRESHOLD is flagged for
human review.
"""

import logging
import os
from google.cloud import documentai_v1 as documentai
from google.api_core.exceptions import GoogleAPICallError

from config import settings
from models import BoundingBox, LineResult, TokenResult

logger = logging.getLogger(__name__)


def _build_client() -> documentai.DocumentProcessorServiceClient:
    """Instantiate the Document AI client.

    The SDK picks up credentials automatically from the
    GOOGLE_APPLICATION_CREDENTIALS environment variable, which is set by
    python-dotenv via config.py at startup.
    """
    os.environ.setdefault(
        "GOOGLE_APPLICATION_CREDENTIALS", settings.GOOGLE_APPLICATION_CREDENTIALS
    )
    client_options = {"api_endpoint": f"{settings.DOCUMENT_AI_LOCATION}-documentai.googleapis.com"}
    return documentai.DocumentProcessorServiceClient(client_options=client_options)


def _bounding_box_from_vertices(
    vertices: list,
) -> BoundingBox | None:
    """Convert normalised bounding polygon vertices to a BoundingBox model."""
    if not vertices:
        return None
    xs = [v.x for v in vertices]
    ys = [v.y for v in vertices]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return BoundingBox(
        x=round(x_min, 6),
        y=round(y_min, 6),
        width=round(x_max - x_min, 6),
        height=round(y_max - y_min, 6),
    )


def _extract_text(document: documentai.Document, element) -> str:
    """Extract the substring of document.text that a layout segment refers to."""
    text = ""
    for segment in element.layout.text_anchor.text_segments:
        start = int(segment.start_index)
        end = int(segment.end_index)
        text += document.text[start:end]
    return text.strip()


def _parse_document(document: documentai.Document) -> list[LineResult]:
    """
    Walk the Document AI page structure to build a list of LineResult objects.

    Document AI groups tokens into paragraphs, but does NOT expose a native
    "line" entity for Document OCR processors.  We reconstruct lines by
    treating each paragraph's tokens and grouping them by approximate Y
    coordinate of their bounding box (within a 1 % page-height tolerance).
    """
    lines: list[LineResult] = []

    for page in document.pages:
        page_height = page.dimension.height or 1.0

        # Collect all tokens on this page with their vertical midpoint.
        token_data: list[tuple[float, TokenResult]] = []
        for token in page.tokens:
            token_text = _extract_text(document, token)
            if not token_text:
                continue

            confidence = token.layout.confidence
            vertices = (
                token.layout.bounding_poly.normalized_vertices
                if token.layout.bounding_poly.normalized_vertices
                else []
            )
            bbox = _bounding_box_from_vertices(vertices)
            y_mid = bbox.y + bbox.height / 2 if bbox else 0.0

            token_data.append(
                (y_mid, TokenResult(text=token_text, confidence=confidence, bounding_box=bbox))
            )

        if not token_data:
            continue

        # Sort tokens top-to-bottom, then left-to-right within the same row.
        token_data.sort(key=lambda t: (t[0], t[1].bounding_box.x if t[1].bounding_box else 0))

        # Group into lines: tokens within LINE_TOLERANCE of each other share a line.
        LINE_TOLERANCE = 0.015  # 1.5 % of normalised page height
        grouped: list[list[TokenResult]] = []
        current_group: list[TokenResult] = []
        current_y: float | None = None

        for y_mid, token_result in token_data:
            if current_y is None or abs(y_mid - current_y) > LINE_TOLERANCE:
                if current_group:
                    grouped.append(current_group)
                current_group = [token_result]
                current_y = y_mid
            else:
                current_group.append(token_result)

        if current_group:
            grouped.append(current_group)

        # Build LineResult for each group.
        for group in grouped:
            line_text = " ".join(t.text for t in group)
            low_conf = [t.text for t in group if t.confidence < settings.CONFIDENCE_THRESHOLD]
            requires_review = len(low_conf) > 0

            # Merge bounding boxes of all tokens to get the line's bbox.
            bboxes = [t.bounding_box for t in group if t.bounding_box]
            if bboxes:
                x_min = min(b.x for b in bboxes)
                y_min = min(b.y for b in bboxes)
                x_max = max(b.x + b.width for b in bboxes)
                y_max = max(b.y + b.height for b in bboxes)
                line_bbox = BoundingBox(
                    x=round(x_min, 6),
                    y=round(y_min, 6),
                    width=round(x_max - x_min, 6),
                    height=round(y_max - y_min, 6),
                )
            else:
                line_bbox = None

            lines.append(
                LineResult(
                    line_text=line_text,
                    tokens=group,
                    low_confidence_tokens=low_conf,
                    requires_review=requires_review,
                    bounding_box=line_bbox,
                )
            )

    return lines


async def run_ocr(image_bytes: bytes) -> tuple[str, list[LineResult]]:
    """
    Send *image_bytes* to Document AI and return:
      - raw_text: the full document text string
      - lines:    list of LineResult (with requires_review flags set)

    This call is synchronous under the hood (the Document AI client does not
    provide an async interface), so it runs in the default thread-pool.
    """
    import asyncio
    from functools import partial

    loop = asyncio.get_running_loop()

    def _call_document_ai() -> documentai.Document:
        client = _build_client()
        processor_name = client.processor_path(
            settings.GOOGLE_CLOUD_PROJECT,
            settings.DOCUMENT_AI_LOCATION,
            settings.DOCUMENT_AI_PROCESSOR_ID,
        )
        raw_document = documentai.RawDocument(
            content=image_bytes,
            mime_type="image/jpeg",
        )
        request = documentai.ProcessRequest(
            name=processor_name,
            raw_document=raw_document,
        )
        try:
            result = client.process_document(request=request)
            return result.document
        except GoogleAPICallError as exc:
            logger.error("Document AI API error: %s", exc)
            raise RuntimeError(f"Document AI call failed: {exc}") from exc

    logger.info("Sending image to Document AI OCR processor.")
    document = await loop.run_in_executor(None, _call_document_ai)

    raw_text = document.text or ""
    lines = _parse_document(document)

    flagged_count = sum(1 for ln in lines if ln.requires_review)
    logger.info(
        "OCR complete. %d lines extracted, %d flagged for review.",
        len(lines),
        flagged_count,
    )
    return raw_text, lines
