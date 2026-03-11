import "dotenv/config";
import express, { Request, Response, NextFunction } from "express";
import cors from "cors";
import fs from "fs";
import path from "path";
import multer from "multer";
import axios from "axios";
import FormData from "form-data";
import { ObjectId } from "mongodb";

import * as db from "./services/database.js";
import {
  searchFlaggedLines,
  saveIngredients,
} from "./services/algoliaService.js";
import { runOcr } from "./services/documentAiService.js";
import { extractDeliveryNote } from "./services/replicateLlmService.js";
import {
  AlgoliaMatch,
  DeliveryNoteDocument,
  DeliveryNoteIngredient,
  DeliveryNoteStatus,
  ProcessingResponse,
} from "./models.js";

const app = express();
const PORT = process.env.PORT || 3000;
const IMAGE_PROCESSOR_URL =
  process.env.IMAGE_PROCESSOR_URL || "http://image-processor:8000";
const UPLOADS_DIR = path.resolve("uploads");
fs.mkdirSync(UPLOADS_DIR, { recursive: true });

app.use(
  cors({
    origin: process.env.CORS_ORIGIN || "http://localhost:5173",
    methods: ["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

app.use("/uploads", express.static(UPLOADS_DIR));

const upload = multer({ storage: multer.memoryStorage() });

const ALLOWED_MIME_TYPES = new Set([
  "application/pdf",
  "image/jpeg",
  "image/jpg",
  "image/png",
  "image/tiff",
]);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type FlaggedLine = [string, string[], AlgoliaMatch | null];

function attachReviewContext(
  ingredients: DeliveryNoteIngredient[],
  flaggedLines: FlaggedLine[],
): void {
  if (!flaggedLines.length) return;

  for (const ingredient of ingredients) {
    if (!ingredient.requires_review || !ingredient.name) continue;

    const nameWords = new Set(ingredient.name.toLowerCase().split(/\s+/));
    let bestScore = 0;
    let bestTokens: string[] = [];
    let bestMatch: AlgoliaMatch | undefined;

    for (const [lineText, lowConfTokens, algoliaMatch] of flaggedLines) {
      const lineWords = new Set(lineText.toLowerCase().split(/\s+/));
      const score = [...nameWords].filter((w) => lineWords.has(w)).length;
      if (score > bestScore) {
        bestScore = score;
        bestTokens = lowConfTokens;
        bestMatch = algoliaMatch ?? undefined;
      }
    }

    if (bestScore > 0) {
      ingredient.low_confidence_tokens = bestTokens;
      ingredient.algolia_match = bestMatch;
    }
  }
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

app.get("/health", (_req, res) => {
  res.json({ status: "ok" });
});

// List delivery notes
app.get("/api/v1/delivery-notes", async (req: Request, res: Response) => {
  const filter = (req.query.filter as string) ?? "all";
  const skip = parseInt((req.query.skip as string) ?? "0", 10);
  const limit = Math.min(
    parseInt((req.query.limit as string) ?? "50", 10),
    500,
  );

  let mongoFilter: Record<string, unknown> = {};
  if (filter === "validated") {
    mongoFilter = { status: DeliveryNoteStatus.VALIDATED };
  } else if (filter === "non_validated") {
    mongoFilter = { status: { $ne: DeliveryNoteStatus.VALIDATED } };
  }

  const collection = db.getCollection();
  const total = await collection.countDocuments(mongoFilter);
  const items = await collection
    .find(mongoFilter)
    .skip(skip)
    .limit(limit)
    .toArray();

  const mapped = items.map((doc) => {
    const { _id, ...rest } = doc;
    return { id: _id.toString(), ...rest };
  });

  res.json({ total, items: mapped });
});

// Validate a delivery note
app.patch(
  "/api/v1/delivery-notes/:id/validate",
  express.json(),
  async (req: Request, res: Response) => {
    let oid: ObjectId;
    try {
      oid = new ObjectId(req.params.id as string);
    } catch {
      res.status(422).json({ detail: "Invalid document_id format." });
      return;
    }

    const collection = db.getCollection();
    const existing = await collection.findOne({ _id: oid });
    if (!existing) {
      res.status(404).json({ detail: "Delivery note not found." });
      return;
    }

    const updateData: Record<string, unknown> = { ...req.body };

    if (Array.isArray(updateData.delivery_note_ingredients)) {
      for (const ing of updateData.delivery_note_ingredients as Record<
        string,
        unknown
      >[]) {
        ing.requires_review = false;
      }
    }

    updateData.validated = true;
    updateData.status = DeliveryNoteStatus.VALIDATED;
    updateData.has_issues = false;

    await collection.updateOne({ _id: oid }, { $set: updateData });

    const updated = await collection.findOne({ _id: oid });
    if (!updated) {
      res.status(500).json({ detail: "Update failed." });
      return;
    }

    const { _id, ...rest } = updated;
    const result = { id: _id.toString(), ...rest };

    // Push ingredient names to Algolia
    const names = (result.delivery_note_ingredients ?? [])
      .map((i) => i.name)
      .filter((n): n is string => !!n?.trim());

    if (names.length) {
      try {
        await saveIngredients(names);
      } catch (err) {
        console.warn("Algolia ingredient upsert failed (non-fatal):", err);
      }
    }

    res.json(result);
  },
);

// Main pipeline endpoint
app.post(
  "/api/v1/process-delivery-note",
  upload.single("file"),
  async (req: Request, res: Response) => {
    const file = req.file;
    if (!file) {
      res.status(400).json({ detail: "No file uploaded." });
      return;
    }

    if (!ALLOWED_MIME_TYPES.has(file.mimetype)) {
      res.status(415).json({
        detail: `Unsupported file type '${file.mimetype}'. Accepted: PDF, JPEG, PNG, TIFF.`,
      });
      return;
    }

    const filename = file.originalname || "upload";
    const restaurantId = req.query.restaurant_id as string | undefined;
    console.info(
      `Received '${filename}' (${file.size} bytes, ${file.mimetype})`,
    );

    // ── Step 1: Preprocessing (image-processor microservice) ──────────────
    let cleanedImage: Buffer;
    try {
      const form = new FormData();
      form.append("file", file.buffer, {
        filename: file.originalname,
        contentType: file.mimetype,
      });

      const response = await axios.post(
        `${IMAGE_PROCESSOR_URL}/api/v1/process-image`,
        form,
        {
          headers: form.getHeaders(),
          responseType: "arraybuffer",
          maxBodyLength: Infinity,
        },
      );
      cleanedImage = Buffer.from(response.data);
      console.info(`Image preprocessed: ${cleanedImage.length} bytes.`);
    } catch (err: any) {
      console.error("Preprocessing failed:", err.message);
      const status = err.response?.status ?? 500;
      res.status(status >= 400 && status < 600 ? status : 502).json({
        detail: `Preprocessing failed: ${err.message}`,
      });
      return;
    }

    // ── Step 2: Document AI OCR ───────────────────────────────────────────
    let rawText: string;
    let lines: Awaited<ReturnType<typeof runOcr>>[1];
    try {
      [rawText, lines] = await runOcr(cleanedImage);
    } catch (err) {
      console.error("Document AI OCR failed:", err);
      res.status(502).json({ detail: `OCR failed: ${err}` });
      return;
    }

    const flaggedLines = lines.filter((l) => l.requires_review);
    console.info(
      `${flaggedLines.length} / ${lines.length} lines flagged for review.`,
    );

    // ── Step 3: Algolia search ────────────────────────────────────────────
    let algoliaResults: (AlgoliaMatch | null)[] = [];
    if (flaggedLines.length) {
      try {
        algoliaResults = await searchFlaggedLines(
          flaggedLines.map((l) => l.line_text),
        );
      } catch (err) {
        console.warn("Algolia search failed (non-fatal):", err);
        algoliaResults = new Array(flaggedLines.length).fill(null);
      }
    }

    const flaggedContext: FlaggedLine[] = flaggedLines.map((ln, i) => [
      ln.line_text,
      ln.low_confidence_tokens,
      algoliaResults[i] ?? null,
    ]);

    // ── Step 4: LLM — full invoice extraction ─────────────────────────────
    let headerData: Record<string, unknown> = {};
    let ingredients: DeliveryNoteIngredient[] = [];
    let pipelineStatus = DeliveryNoteStatus.PROCESSED;

    try {
      [headerData, ingredients] = await extractDeliveryNote(
        rawText,
        flaggedContext,
      );
    } catch (err) {
      console.error("LLM invoice extraction failed:", err);
      pipelineStatus = DeliveryNoteStatus.ERROR;
    }

    attachReviewContext(ingredients, flaggedContext);

    const hasIssues = ingredients.some((i) => i.requires_review);
    if (hasIssues && pipelineStatus !== DeliveryNoteStatus.ERROR) {
      pipelineStatus = DeliveryNoteStatus.PENDING;
    }

    // ── Step 5: MongoDB persistence ───────────────────────────────────────
    const document: DeliveryNoteDocument = {
      filename,
      raw_text: rawText,
      status: pipelineStatus,
      ...(restaurantId && { restaurant_id: restaurantId }),
      has_issues: hasIssues,
      delivery_note_ingredients: ingredients,
      created_at: new Date(),
      ...(headerData.supplier_name && {
        supplier_name: headerData.supplier_name as string,
      }),
      ...(headerData.date && { date: headerData.date as string }),
      ...(headerData.serial_number && {
        serial_number: headerData.serial_number as string,
      }),
      ...(headerData.total_vat != null && {
        total_vat: headerData.total_vat as number,
      }),
      ...(headerData.total_discount != null && {
        total_discount: headerData.total_discount as number,
      }),
      ...(headerData.total_expense != null && {
        total_expense: headerData.total_expense as number,
      }),
    };

    let documentId: string;
    try {
      const collection = db.getCollection();
      const result = await collection.insertOne(document as any);
      documentId = result.insertedId.toString();

      const imageUrl = `/uploads/${documentId}.jpg`;
      fs.writeFileSync(path.join(UPLOADS_DIR, `${documentId}.jpg`), cleanedImage);
      await collection.updateOne(
        { _id: result.insertedId },
        { $set: { image_url: imageUrl } },
      );
      document.image_url = imageUrl;

      console.info(
        `Document saved to MongoDB _id=${documentId} (status=${pipelineStatus})`,
      );
    } catch (err) {
      console.error("MongoDB insert failed:", err);
      res.status(503).json({ detail: `Database write failed: ${err}` });
      return;
    }

    const flaggedCount = ingredients.filter((i) => i.requires_review).length;

    const response: ProcessingResponse = {
      document_id: documentId,
      message:
        pipelineStatus !== DeliveryNoteStatus.ERROR
          ? "Delivery note processed successfully."
          : "Processing completed with errors — manual review required.",
      status: pipelineStatus,
      has_issues: hasIssues,
      flagged_ingredient_count: flaggedCount,
      payload: document,
    };

    res.status(201).json(response);
  },
);

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

async function start() {
  await db.connectDb();

  app.listen(PORT, () => {
    console.info(`Node orchestrator running on http://localhost:${PORT}`);
  });
}

start().catch((err) => {
  console.error("Failed to start:", err);
  process.exit(1);
});
