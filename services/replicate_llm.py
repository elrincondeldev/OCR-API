"""
Step 4 — Replicate LLM: Full Invoice Extraction.

A single LLM call receives:
  - The complete OCR raw text of the delivery note.
  - Contextual hints for every line that had low OCR confidence
    (the low-confidence words + the top Algolia inventory match).

The model returns ONE JSON object with the complete structured invoice:
  header fields (supplier, date, serial number, totals) +
  all ingredient line items (name, quantity, unit, price, VAT, discount),
  with flagged ingredients marked requires_review=true.
"""

import json
import logging
import asyncio
import os
from functools import partial

import replicate

from config import settings
from models import AlgoliaMatch, DeliveryNoteIngredient, IngredientUnit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a JSON-only data extraction assistant for a restaurant purchasing system.
You receive the raw OCR text of a supplier delivery note / invoice, and optionally
a list of lines that had low OCR confidence with their best inventory match from
a search engine.

Extract all information and output ONE valid JSON object with this exact structure:

{
  "supplier_name": string | null,
  "date": string | null,
  "serial_number": string | null,
  "total_vat": number | null,
  "total_discount": number | null,
  "total_expense": number | null,
  "delivery_note_ingredients": [
    {
      "name": string | null,
      "price_per_format": number,
      "quantity": number | null,
      "vat_percentage": number | null,
      "discount": number | null,
      "format_quantity": number,
      "unit": string | null,
      "requires_review": boolean,
      "human_warning_message": string | null
    }
  ]
}

STRICT RULES:
- Output ONLY the JSON object. No markdown fences, no explanation, no extra text.
- Use null for any field you cannot find or confidently determine.
- date must be ISO 8601 format (YYYY-MM-DD) or null.
- unit must be one of [kg, g, l, ml, unit, box, crate, pack] or null.
- price_per_format and format_quantity must always be numbers (default 0.0 and 1.0).
- All other numeric fields must be numbers, never strings.
- For any ingredient whose line appears in FLAGGED LINES: set requires_review to
  true and write a one-sentence human_warning_message explaining the issue.
  Use the inventory_hint to correct the name when it makes sense.
- For all other ingredients: requires_review is false, human_warning_message is null.
"""


def _build_user_message(
    raw_text: str,
    flagged_lines: list[tuple[str, list[str], AlgoliaMatch | None]],
) -> str:
    parts = [f"FULL OCR TEXT:\n{raw_text}"]

    if flagged_lines:
        parts.append("\nFLAGGED LINES (low OCR confidence — needs correction):")
        for line_text, low_conf_words, match in flagged_lines:
            hint = match.matched_item_name if match else "no match found"
            parts.append(
                f'  • Line       : "{line_text}"\n'
                f'    Low-conf   : {", ".join(low_conf_words) or "none"}\n'
                f'    Inventory  : {hint}'
            )
    else:
        parts.append("\nNo flagged lines — all OCR confidence was acceptable.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Replicate call helpers
# ---------------------------------------------------------------------------

def _collect_stream(stream) -> str:
    parts: list[str] = []
    for chunk in stream:
        if isinstance(chunk, str):
            parts.append(chunk)
    return "".join(parts)


def _call_replicate(system_prompt: str, user_message: str) -> str:
    """Synchronous Replicate call — must run in a thread-pool executor."""
    try:
        output = replicate.run(
            settings.REPLICATE_MODEL,
            input={
                "prompt": user_message,
                "system_prompt": system_prompt,
                "max_new_tokens": 2048,   # invoices can be long
                "temperature": 0.1,       # low = deterministic JSON
                "top_p": 0.9,
            },
        )
        if hasattr(output, "__iter__") and not isinstance(output, str):
            return _collect_stream(output)
        return str(output)
    except Exception as exc:
        logger.error("Replicate API error: %s", exc)
        raise RuntimeError(f"Replicate call failed: {exc}") from exc


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _strip_fences(raw: str) -> str:
    """Remove accidental ```json ... ``` markdown fences."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            ln for ln in lines if not ln.strip().startswith("```")
        )
    return cleaned.strip()


def _parse_ingredient(data: dict) -> DeliveryNoteIngredient:
    unit_raw = data.get("unit")
    unit: IngredientUnit | None = None
    if unit_raw:
        try:
            unit = IngredientUnit(str(unit_raw).lower())
        except ValueError:
            logger.warning("Unknown unit value from LLM: '%s'", unit_raw)

    return DeliveryNoteIngredient(
        name=data.get("name"),
        price_per_format=float(data.get("price_per_format") or 0.0),
        quantity=_to_float(data.get("quantity")),
        vat_percentage=_to_float(data.get("vat_percentage")),
        discount=_to_float(data.get("discount")),
        format_quantity=float(data.get("format_quantity") or 1.0),
        unit=unit,
        requires_review=bool(data.get("requires_review", False)),
        human_warning_message=data.get("human_warning_message"),
    )


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_llm_output(raw_output: str) -> dict:
    """
    Parse the LLM's raw string into a plain dict.
    Returns an empty-ish fallback dict on parse failure.
    """
    try:
        return json.loads(_strip_fences(raw_output))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error(
            "Failed to parse LLM invoice JSON: %s\nRaw output (first 500): %s",
            exc,
            raw_output[:500],
        )
        return {"delivery_note_ingredients": []}


# ---------------------------------------------------------------------------
# Public async entry-point
# ---------------------------------------------------------------------------

async def extract_delivery_note(
    raw_text: str,
    flagged_lines: list[tuple[str, list[str], AlgoliaMatch | None]],
) -> tuple[dict, list[DeliveryNoteIngredient]]:
    """
    Call the LLM once with the full OCR text and all flagged-line context.

    Returns:
      - header_data: dict with supplier_name, date, serial_number, totals
      - ingredients: list[DeliveryNoteIngredient] (already parsed + typed)
    """
    os.environ.setdefault("REPLICATE_API_TOKEN", settings.REPLICATE_API_TOKEN)

    user_message = _build_user_message(raw_text, flagged_lines)

    loop = asyncio.get_running_loop()
    raw_output = await loop.run_in_executor(
        None, partial(_call_replicate, _SYSTEM_PROMPT, user_message)
    )

    data = _parse_llm_output(raw_output)

    header_data = {
        "supplier_name": data.get("supplier_name"),
        "date": data.get("date"),
        "serial_number": data.get("serial_number"),
        "total_vat": _to_float(data.get("total_vat")),
        "total_discount": _to_float(data.get("total_discount")),
        "total_expense": _to_float(data.get("total_expense")),
    }

    ingredients: list[DeliveryNoteIngredient] = []
    for item in data.get("delivery_note_ingredients", []):
        if isinstance(item, dict):
            ingredients.append(_parse_ingredient(item))

    logger.info(
        "LLM extracted: supplier='%s', date='%s', %d ingredient(s), %d flagged.",
        header_data.get("supplier_name"),
        header_data.get("date"),
        len(ingredients),
        sum(1 for i in ingredients if i.requires_review),
    )

    return header_data, ingredients
