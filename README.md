# OCR-API — Restaurant Delivery Note Pipeline

A **Human-in-the-Loop OCR pipeline** that turns restaurant supplier delivery notes (PDFs or images) into structured, validated JSON data.

Upload a photo or scan of a delivery note → get back a fully parsed invoice with ingredient names, quantities, prices, VAT, and discounts — with low-confidence items flagged for human review before being validated.

---

### Pipeline steps

| #   | Step                 | Service             | What it does                                                                                                                                                                |
| --- | -------------------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Preprocessing**    | `image-processor`   | Converts PDF → image (300 DPI), then applies grayscale → Gaussian blur → adaptive threshold → morphological open to clean the scan                                          |
| 2   | **OCR**              | Google Document AI  | Extracts text with per-token confidence scores. Tokens are grouped into lines by Y-midpoint (1.5% tolerance). Lines with any token below `CONFIDENCE_THRESHOLD` are flagged |
| 3   | **Inventory search** | Algolia             | Concurrent typo-tolerant search for every flagged line. Returns the best matching ingredient from your inventory index as a hint for the LLM                                |
| 4   | **LLM extraction**   | Replicate (Llama 3) | One call with the full OCR text + flagged-line context. Returns a complete structured invoice: header fields + all ingredient line items                                    |
| 5   | **Persistence**      | MongoDB Atlas       | Saves the document with `status=pending` (has issues) or `status=processed` (clean). Preprocessed image saved to `/uploads/`                                                |

---

## Architecture

The backend is split into two independent microservices:

### `image-processor` — Python FastAPI

Handles only image preprocessing. Stateless, lightweight. Accepts any image or PDF and returns a clean JPEG ready for OCR.

### `node-orchestrator` — Node.js / Express

Orchestrates the full pipeline. Calls `image-processor`, then runs Document AI → Algolia → Replicate → MongoDB in sequence. Exposes the REST API consumed by the frontend.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) + [Docker Compose](https://docs.docker.com/compose/)
- A **Google Cloud** project with [Document AI](https://cloud.google.com/document-ai) enabled and a processor created
- A **Replicate** account and API token — [replicate.com](https://replicate.com)
- An **Algolia** account with an index — [algolia.com](https://www.algolia.com)
- A **MongoDB Atlas** cluster — [mongodb.com/atlas](https://www.mongodb.com/atlas)

---

## Quick start

**1. Clone the repo**

```bash
git clone <your-repo-url>
cd OCR-API
```

**2. Create your environment file**

```bash
cp .env.example .env
```

Open `.env` and fill in every value (see [Environment variables](#environment-variables) below).

**3. Build and start all services**

```bash
docker-compose up --build
```

The first build downloads dependencies and may take a few minutes. Subsequent starts are fast.

**4. Verify everything is running**

---

## Environment variables

Create a `.env` file at the repo root. All variables are required unless marked optional.

```env
# Path to your GCP service account JSON on the HOST machine
GCP_KEY_PATH=/absolute/path/to/your/gcp-service-account.json

# Google Document AI
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
DOCUMENT_AI_PROCESSOR_ID=your-processor-id
DOCUMENT_AI_LOCATION=us               # us or eu

# Replicate
REPLICATE_API_TOKEN=r8_xxxxxxxxxxxx
REPLICATE_MODEL=meta/meta-llama-3-70b-instruct

# Algolia
ALGOLIA_APP_ID=XXXXXXXXXX
ALGOLIA_API_KEY=xxxxxxxxxxxxxxxxxxxx
ALGOLIA_INDEX_NAME=restaurant_inventory

# MongoDB Atlas
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB_NAME=ocr_pipeline
MONGODB_COLLECTION=delivery_notes

# Pipeline
CONFIDENCE_THRESHOLD=0.80              # optional, default 0.80
CORS_ORIGIN=http://localhost:5173      # optional, default http://localhost:5173
```

### Getting your GCP service account key

1. Go to [IAM & Admin → Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts)
2. Create a service account with the **Document AI API User** role
3. Create a JSON key and download it
4. Set `GCP_KEY_PATH` to its absolute path on your machine

---

## API reference

Base URL: `http://localhost:3000`

### `POST /api/v1/process-delivery-note`

Run a delivery note through the full pipeline.

**Request** — `multipart/form-data`

| Field           | Type        | Required | Description                           |
| --------------- | ----------- | -------- | ------------------------------------- |
| `file`          | File        | Yes      | PDF, JPEG, PNG, or TIFF delivery note |
| `restaurant_id` | Query param | No       | Associate the note with a restaurant  |

**Response** `201` — `ProcessingResponse`

```json
{
  "document_id": "64f3a1b2c9e1234567890abc",
  "message": "Delivery note processed successfully.",
  "status": "pending",
  "has_issues": true,
  "flagged_ingredient_count": 2,
  "payload": {
    "filename": "albaran-001.pdf",
    "raw_text": "...",
    "status": "pending",
    "has_issues": true,
    "supplier_name": "Mercados del Sur S.L.",
    "date": "2024-03-10",
    "serial_number": "ALB-2024-001",
    "total_vat": 84.25,
    "total_discount": 15.0,
    "total_expense": 425.0,
    "image_url": "/uploads/64f3a1b2c9e1234567890abc.jpg",
    "delivery_note_ingredients": [
      {
        "name": "Solomillo de Ternera",
        "price_per_format": 25.0,
        "quantity": 20,
        "vat_percentage": 21.0,
        "discount": 15.0,
        "format_quantity": 1.0,
        "unit": "kg",
        "requires_review": false,
        "human_warning_message": null
      },
      {
        "name": "Aceite Oliva Virgen",
        "price_per_format": 8.5,
        "quantity": 10,
        "vat_percentage": 10.0,
        "discount": null,
        "format_quantity": 1.0,
        "unit": "l",
        "requires_review": true,
        "low_confidence_tokens": ["Aceite", "Oliva"],
        "algolia_match": {
          "object_id": "abc123",
          "matched_item_name": "Aceite de Oliva Virgen Extra"
        },
        "human_warning_message": "Low OCR confidence on ingredient name — please verify against inventory match."
      }
    ],
    "created_at": "2024-03-10T14:32:00.000Z"
  }
}
```

**Status values**

| Status      | Meaning                                                 |
| ----------- | ------------------------------------------------------- |
| `processed` | All items extracted cleanly, no review needed           |
| `pending`   | One or more ingredients flagged — awaiting human review |
| `validated` | A human has reviewed and confirmed the document         |
| `error`     | LLM extraction failed — manual input required           |

---

### `GET /api/v1/delivery-notes`

List delivery notes with optional filtering and pagination.

**Query params**

| Param    | Default | Options                               |
| -------- | ------- | ------------------------------------- |
| `filter` | `all`   | `all` · `validated` · `non_validated` |
| `skip`   | `0`     | integer                               |
| `limit`  | `50`    | integer, max 500                      |

**Response** `200`

```json
{
  "total": 142,
  "items": [ { "id": "...", "status": "pending", ... } ]
}
```

---

### `PATCH /api/v1/delivery-notes/:id/validate`

Validate a delivery note after human review. Optionally correct any fields. Ingredient names are automatically upserted to the Algolia inventory index.

**Request body** — JSON (all fields optional)

```json
{
  "supplier_name": "Mercados del Sur S.L.",
  "delivery_note_ingredients": [
    {
      "name": "Aceite de Oliva Virgen Extra",
      "quantity": 10,
      "price_per_format": 8.5,
      "unit": "l"
    }
  ]
}
```

**Response** `200` — the updated document with `status: "validated"`.

---

### `GET /uploads/:id.jpg`

Retrieve the preprocessed image for a delivery note.

```
GET http://localhost:3000/uploads/64f3a1b2c9e1234567890abc.jpg
```

---

---

## Project structure

```
OCR-API/
├── docker-compose.yml
├── .env.example
│
├── image-processor/              # Microservice 1 — Python FastAPI
│   ├── main.py                   # POST /api/v1/process-image
│   ├── preprocessing.py          # pdf2image + OpenCV pipeline
│   ├── requirements.txt
│   └── Dockerfile
│
└── node-orchestrator/            # Microservice 2 — Node.js Express
    ├── index.ts                  # App entry point + all route handlers
    ├── models.ts                 # TypeScript interfaces
    ├── services/
    │   ├── algoliaService.ts     # Inventory search + ingredient upsert
    │   ├── documentAiService.ts  # OCR + line grouping logic
    │   ├── replicateLlmService.ts # Invoice extraction prompt + parsing
    │   └── database.ts           # MongoDB connection
    ├── uploads/                  # Preprocessed images (auto-created)
    ├── package.json
    └── Dockerfile
```

---

## Development (without Docker)

Run each service locally in separate terminals.

**image-processor**

```bash
cd image-processor
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

**node-orchestrator**

```bash
cd node-orchestrator
cp .env.example .env   # fill in values
npm install
IMAGE_PROCESSOR_URL=http://localhost:8001 npm run dev
```

---

## Postman quick test

1. Open Postman → `POST http://localhost:3000/api/v1/process-delivery-note`
2. Body → **form-data** → key: `file` (type: File) → select your delivery note
3. Hit Send → you'll receive a full `ProcessingResponse` with extracted ingredients

To review and validate a flagged document:

```
PATCH http://localhost:3000/api/v1/delivery-notes/<document_id>/validate
Content-Type: application/json

{ "delivery_note_ingredients": [ { "name": "Corrected Name", ... } ] }
```
