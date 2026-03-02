import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, status

import database
from config import settings
from models import (
    AlgoliaMatch,
    DeliveryNoteDocument,
    DeliveryNoteIngredient,
    DeliveryNoteStatus,
    LineResult,
    ProcessingResponse,
)
from services.preprocessing import preprocess_upload
from services.document_ai import run_ocr
from services.algolia_service import search_flagged_lines
from services.replicate_llm import extract_delivery_note

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/tiff",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attach_review_context(
    ingredients: list[DeliveryNoteIngredient],
    flagged_lines: list[tuple[str, list[str], AlgoliaMatch | None]],
) -> None:
    if not flagged_lines:
        return

    for ingredient in ingredients:
        if not ingredient.requires_review or not ingredient.name:
            continue

        name_words = set[str](ingredient.name.lower().split())
        best_score = 0
        best_tokens: list[str] = []
        best_match: AlgoliaMatch | None = None

        for line_text, low_conf_tokens, algolia_match in flagged_lines:
            line_words = set(line_text.lower().split())
            score = len(name_words & line_words)
            if score > best_score:
                best_score = score
                best_tokens = low_conf_tokens
                best_match = algolia_match

        if best_score > 0:
            ingredient.low_confidence_tokens = best_tokens
            ingredient.algolia_match = best_match


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect_db()
    yield
    await database.close_db()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Restaurant Delivery Note OCR Pipeline",
    description=(
        "Human-in-the-Loop OCR pipeline: Document AI → Algolia search → "
        "Replicate LLM invoice extraction → MongoDB storage."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
async def health_check():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Main pipeline endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/process-delivery-note",
    response_model=ProcessingResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["pipeline"],
    summary="Process a delivery note through the full OCR + AI pipeline.",
)
async def process_delivery_note(
    file: UploadFile = File(..., description="PDF, JPEG, or PNG delivery note."),
    restaurant_id: str | None = None,
):
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{content_type}'. "
                f"Accepted: PDF, JPEG, PNG, TIFF."
            ),
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    filename = file.filename or "upload"
    logger.info("Received '%s' (%d bytes, %s)", filename, len(file_bytes), content_type)

    try:
        cleaned_image = await preprocess_upload(file_bytes, content_type)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.exception("Preprocessing failed.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    # ── Step 2: Document AI OCR ───────────────────────────────────────────
    try:
        raw_text, lines = await run_ocr(cleaned_image)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    except Exception as exc:
        logger.exception("Document AI OCR failed.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    flagged_lines: list[LineResult] = [ln for ln in lines if ln.requires_review]
    flagged_texts = [ln.line_text for ln in flagged_lines]
    logger.info("%d / %d lines flagged for review.", len(flagged_lines), len(lines))

    # ── Step 3: Algolia search (concurrent with LLM below) ───────────────
    # We gather Algolia results first so they feed into the LLM prompt.
    algolia_results: list[AlgoliaMatch | None] = []
    if flagged_texts:
        try:
            algolia_results = await search_flagged_lines(flagged_texts)
        except Exception as exc:
            logger.warning("Algolia search failed (non-fatal): %s", exc)
            algolia_results = [None] * len(flagged_texts)

    # Build the combined flagged-line context for the LLM.
    flagged_context: list[tuple[str, list[str], AlgoliaMatch | None]] = [
        (ln.line_text, ln.low_confidence_tokens, algolia_results[i] if i < len(algolia_results) else None)
        for i, ln in enumerate(flagged_lines)
    ]

    # ── Step 4: LLM — full invoice extraction ─────────────────────────────
    pipeline_status = DeliveryNoteStatus.PROCESSED
    try:
        header_data, ingredients = await extract_delivery_note(raw_text, flagged_context)
    except Exception as exc:
        logger.exception("LLM invoice extraction failed.")
        header_data = {}
        ingredients = []
        pipeline_status = DeliveryNoteStatus.ERROR

    # Attach algolia_match + low_confidence_tokens to flagged ingredients.
    _attach_review_context(ingredients, flagged_context)

    has_issues = any(i.requires_review for i in ingredients)
    if has_issues and pipeline_status != DeliveryNoteStatus.ERROR:
        pipeline_status = DeliveryNoteStatus.PENDING

    # ── Step 5: MongoDB persistence ───────────────────────────────────────
    document = DeliveryNoteDocument(
        filename=filename,
        raw_text=raw_text,
        status=pipeline_status,
        restaurant_id=restaurant_id,
        has_issues=has_issues,
        delivery_note_ingredients=ingredients,
        **header_data,
    )

    doc_dict = document.model_dump(mode="json", exclude_none=True)

    try:
        collection = database.get_collection()
        result = await collection.insert_one(doc_dict)
        document_id = str(result.inserted_id)
        logger.info("Document saved to MongoDB _id=%s (status=%s)", document_id, pipeline_status)
    except Exception as exc:
        logger.exception("MongoDB insert failed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database write failed: {exc}",
        )

    flagged_count = sum(1 for i in ingredients if i.requires_review)

    return ProcessingResponse(
        document_id=document_id,
        message="Delivery note processed successfully." if pipeline_status != DeliveryNoteStatus.ERROR
                else "Processing completed with errors — manual review required.",
        status=pipeline_status,
        has_issues=has_issues,
        flagged_ingredient_count=flagged_count,
        payload=document,
    )
