# OCR-API

Human-in-the-Loop OCR pipeline that turns restaurant supplier delivery notes into structured, validated JSON.

OCR-API is a microservices pipeline that takes a photo, scan, or PDF of a delivery note and returns a fully parsed invoice — ingredient names, quantities, prices, VAT, and discounts. Low-confidence items are automatically flagged for human review before validation, so the data you keep is always correct.

## 📌 Why This Project?

OCR alone is noisy — a single misread digit on an invoice becomes a wrong price. This project pairs machine extraction with a human review step, giving you the speed of automation without sacrificing accuracy.

It's also a practical showcase of:

- A real microservices split (Python preprocessing + Node orchestration)
- Combining OCR, search, and LLMs into one coherent pipeline
- Confidence-based flagging and human-in-the-loop validation
- Clean REST API design

## 🔬 Pipeline

| # | Step | Service | What it does |
|---|------|---------|--------------|
| 1 | **Preprocessing** | `image-processor` | PDF → image (300 DPI), then grayscale → blur → adaptive threshold → morphological open |
| 2 | **OCR** | Google Document AI | Extracts text with per-token confidence; tokens grouped into lines, low-confidence lines flagged |
| 3 | **Inventory search** | Algolia | Typo-tolerant search for each flagged line, returning the best inventory match as an LLM hint |
| 4 | **LLM extraction** | Replicate (Llama 3) | One call with full OCR text + flagged context → complete structured invoice |
| 5 | **Persistence** | MongoDB Atlas | Saves the document as `pending` (needs review) or `processed` (clean) |

## 🧱 Architecture

```
OCR-API/
├── image-processor/    # Python FastAPI — image preprocessing only
├── node-orchestrator/  # Node.js Express — OCR + search + LLM + DB
└── docker-compose.yml
```

Two independent services: `image-processor` cleans the scan and returns a JPEG, `node-orchestrator` runs the full pipeline and exposes the REST API.

## ⚙️ Quick Start

```bash
git clone <your-repo-url>
cd OCR-API
cp .env.example .env      # fill in your credentials
docker-compose up --build
```

The API is then available at `http://localhost:3000`.

You'll need accounts/credentials for **Google Document AI**, **Replicate**, **Algolia**, and **MongoDB Atlas** — set them in `.env`.

## 🧪 Example Usage

```bash
curl -X POST http://localhost:3000/api/v1/process-delivery-note \
  -F "file=@albaran-001.pdf"
```

Returns a `ProcessingResponse` with the extracted invoice and any flagged ingredients. Review and validate a flagged document with:

```bash
curl -X PATCH http://localhost:3000/api/v1/delivery-notes/<id>/validate \
  -H "Content-Type: application/json" \
  -d '{ "delivery_note_ingredients": [ { "name": "Corrected Name" } ] }'
```

## 📡 API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/process-delivery-note` | Run a delivery note through the full pipeline |
| `GET`  | `/api/v1/delivery-notes` | List delivery notes (filter, skip, limit) |
| `PATCH`| `/api/v1/delivery-notes/:id/validate` | Validate a note after human review |
| `GET`  | `/uploads/:id.jpg` | Retrieve the preprocessed image |

## 📄 License

MIT License. Free to use and modify.

## 💡 Contributing

Open to collaboration — fork it, suggest improvements, or help extend the pipeline.
