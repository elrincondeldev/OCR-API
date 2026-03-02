from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Any


# ---------------------------------------------------------------------------
# Domain enums
# ---------------------------------------------------------------------------

class IngredientUnit(str, Enum):
    KG = "kg"
    G = "g"
    L = "l"
    ML = "ml"
    UNIT = "unit"
    BOX = "box"
    CRATE = "crate"
    PACK = "pack"


class DeliveryNoteStatus(str, Enum):
    PENDING = "pending"
    PROCESSED = "processed"
    VALIDATED = "validated"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Document AI intermediate models (internal pipeline use only)
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class TokenResult(BaseModel):
    text: str
    confidence: float
    bounding_box: BoundingBox | None = None


class LineResult(BaseModel):
    line_text: str
    tokens: list[TokenResult]
    low_confidence_tokens: list[str]
    requires_review: bool
    bounding_box: BoundingBox | None = None


# ---------------------------------------------------------------------------
# Algolia search model
# ---------------------------------------------------------------------------

class AlgoliaMatch(BaseModel):
    object_id: str
    matched_item_name: str
    standard_quantity: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Restaurant domain — ingredient line item
# ---------------------------------------------------------------------------

class DeliveryNoteIngredient(BaseModel):
    name: str | None = None
    price_per_format: float = 0.0       # Price for one format unit (e.g. €10 per 5 L bottle)
    quantity: float | None = None        # Number of format units delivered
    vat_percentage: float | None = None  # e.g. 10.0 for 10 %
    discount: float | None = None        # Discount applied to this line
    format_quantity: float = 1.0         # Size of one unit (e.g. 5.0 for a 5 L bottle)
    unit: IngredientUnit | None = None

    # Review metadata — populated for low-confidence lines
    requires_review: bool = False
    low_confidence_tokens: list[str] = Field(default_factory=list)
    algolia_match: AlgoliaMatch | None = None
    human_warning_message: str | None = None


# ---------------------------------------------------------------------------
# Top-level MongoDB document
# ---------------------------------------------------------------------------

class DeliveryNoteDocument(BaseModel):
    filename: str
    raw_text: str
    status: DeliveryNoteStatus = DeliveryNoteStatus.PENDING
    restaurant_id: str | None = None
    supplier_id: str | None = None
    has_issues: bool = False

    # Header fields extracted by the LLM
    supplier_name: str | None = None
    date: str | None = None          # ISO 8601 date string (YYYY-MM-DD)
    serial_number: str | None = None
    total_vat: float | None = None
    total_discount: float | None = None
    total_expense: float | None = None

    # Structured line items
    delivery_note_ingredients: list[DeliveryNoteIngredient] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# API response
# ---------------------------------------------------------------------------

class ProcessingResponse(BaseModel):
    document_id: str
    message: str
    status: DeliveryNoteStatus
    has_issues: bool
    flagged_ingredient_count: int
    payload: DeliveryNoteDocument
