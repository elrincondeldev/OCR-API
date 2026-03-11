export enum IngredientUnit {
  KG = "kg",
  G = "g",
  L = "l",
  ML = "ml",
  UNIT = "unit",
  BOX = "box",
  CRATE = "crate",
  PACK = "pack",
}

export enum DeliveryNoteStatus {
  PENDING = "pending",
  PROCESSED = "processed",
  VALIDATED = "validated",
  ERROR = "error",
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface TokenResult {
  text: string;
  confidence: number;
  bounding_box?: BoundingBox;
}

export interface LineResult {
  line_text: string;
  tokens: TokenResult[];
  low_confidence_tokens: string[];
  requires_review: boolean;
  bounding_box?: BoundingBox;
}

export interface AlgoliaMatch {
  object_id: string;
  matched_item_name: string;
  standard_quantity?: string;
  metadata?: Record<string, unknown>;
}

export interface DeliveryNoteIngredient {
  name?: string;
  price_per_format: number;
  quantity?: number;
  vat_percentage?: number;
  discount?: number;
  format_quantity: number;
  unit?: IngredientUnit;
  requires_review: boolean;
  low_confidence_tokens?: string[];
  algolia_match?: AlgoliaMatch;
  human_warning_message?: string;
}

export interface DeliveryNoteDocument {
  filename: string;
  raw_text: string;
  status: DeliveryNoteStatus;
  restaurant_id?: string;
  has_issues: boolean;
  supplier_name?: string;
  date?: string;
  serial_number?: string;
  total_vat?: number;
  total_discount?: number;
  total_expense?: number;
  delivery_note_ingredients: DeliveryNoteIngredient[];
  created_at: Date;
  image_url?: string;
}

export interface ProcessingResponse {
  document_id: string;
  message: string;
  status: DeliveryNoteStatus;
  has_issues: boolean;
  flagged_ingredient_count: number;
  payload: DeliveryNoteDocument;
}
