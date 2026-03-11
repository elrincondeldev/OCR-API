import Replicate from "replicate";
import { AlgoliaMatch, DeliveryNoteIngredient, IngredientUnit } from "../models.js";

const SYSTEM_PROMPT = `\
You are a JSON-only data extraction assistant for a restaurant purchasing system.
You receive the raw OCR text of a supplier delivery note / invoice (often in Spanish
or French), and optionally a list of lines that had low OCR confidence with their
best inventory match from a search engine.

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

COLUMN HEADER MAPPINGS — delivery notes use many equivalent labels:
  quantity      : UDS., UNIDADES, QTY, CANTIDAD, UNITS, CANT.
  price_per_format : BASE UD., PRECIO UNIT., PVP, PRIX UNIT., UNIT PRICE, P.UNIT.
  discount      : % DTO, DTO%, DESCUENTO, DESC., REMISE, DISC., % DESC.
  vat_percentage: % IVA, % I.V.A., TVA%, IVA, VAT%
  total line    : BASE TOTAL, TOTAL LÍNEA, IMPORTE, MONTANT — use only for cross-checking

DISCOUNT RULE — THIS IS CRITICAL:
  - "discount" is ALWAYS a percentage between 0 and 100, never an absolute amount.
  - Read it directly from the % DTO / DESCUENTO column on the same row as the ingredient.
  - If the column shows "15%" store 15.0. If "0%" store 0.0. If absent store null.
  - Never confuse the discount % with the VAT %. They are different columns.
  - Cross-check: price_per_format × quantity × (1 - discount/100) should equal BASE TOTAL.

WORKED EXAMPLE — given this OCR table row:
  "Solomillo de Ternera  20  25.00 €  15%  425.00 €  21%  89.25"
  Columns:  name=Solomillo de Ternera, quantity=20, price_per_format=25.00,
            discount=15.0, base_total=425.00, vat_percentage=21.0
  Verify: 20 × 25.00 × (1 - 0.15) = 425.00 ✓

SERIAL NUMBER: look for labels like "Albarán", "Nº Albarán", "Factura", "Nº:", "Número de albarán".

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
- For all other ingredients: requires_review is false, human_warning_message is null.`;

type FlaggedLine = [string, string[], AlgoliaMatch | null];

function buildUserMessage(rawText: string, flaggedLines: FlaggedLine[]): string {
  const parts = [`FULL OCR TEXT:\n${rawText}`];

  if (flaggedLines.length) {
    parts.push("\nFLAGGED LINES (low OCR confidence — needs correction):");
    for (const [lineText, lowConfWords, match] of flaggedLines) {
      const hint = match?.matched_item_name ?? "no match found";
      parts.push(
        `  • Line       : "${lineText}"\n` +
          `    Low-conf   : ${lowConfWords.join(", ") || "none"}\n` +
          `    Inventory  : ${hint}`
      );
    }
  } else {
    parts.push("\nNo flagged lines — all OCR confidence was acceptable.");
  }

  return parts.join("\n");
}

async function collectStream(output: unknown): Promise<string> {
  if (output == null) return "";
  if (typeof output === "string") return output;
  if (Array.isArray(output)) return output.join("");
  if (Symbol.asyncIterator in Object(output)) {
    let result = "";
    for await (const chunk of output as AsyncIterable<string>) {
      result += chunk;
    }
    return result;
  }
  return String(output);
}

function stripFences(raw: string): string {
  const cleaned = raw.trim();
  if (!cleaned.startsWith("```")) return cleaned;
  return cleaned
    .split("\n")
    .filter((l) => !l.trim().startsWith("```"))
    .join("\n")
    .trim();
}

function toFloat(value: unknown): number | undefined {
  if (value == null) return undefined;
  const n = parseFloat(String(value));
  return isNaN(n) ? undefined : n;
}

function parseIngredient(data: Record<string, unknown>): DeliveryNoteIngredient {
  let unit: IngredientUnit | undefined;
  const unitRaw = data.unit;
  if (unitRaw) {
    const unitStr = String(unitRaw).toLowerCase();
    if (Object.values(IngredientUnit).includes(unitStr as IngredientUnit)) {
      unit = unitStr as IngredientUnit;
    } else {
      console.warn(`Unknown unit value from LLM: '${unitRaw}'`);
    }
  }

  return {
    name: (data.name as string) ?? undefined,
    price_per_format: parseFloat(String(data.price_per_format ?? 0)) || 0,
    quantity: toFloat(data.quantity),
    vat_percentage: toFloat(data.vat_percentage),
    discount: toFloat(data.discount),
    format_quantity: parseFloat(String(data.format_quantity ?? 1)) || 1,
    unit,
    requires_review: Boolean(data.requires_review),
    human_warning_message: (data.human_warning_message as string) ?? undefined,
  };
}

export async function extractDeliveryNote(
  rawText: string,
  flaggedLines: FlaggedLine[]
): Promise<[Record<string, unknown>, DeliveryNoteIngredient[]]> {
  const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN! });
  const userMessage = buildUserMessage(rawText, flaggedLines);

  console.info("Calling Replicate LLM for full invoice extraction.");

  let rawOutput: string;
  try {
    const output = await replicate.run(
      process.env.REPLICATE_MODEL! as `${string}/${string}`,
      {
        input: {
          prompt: userMessage,
          system_prompt: SYSTEM_PROMPT,
          max_new_tokens: 2048,
          temperature: 0.1,
          top_p: 0.9,
        },
      }
    );
    rawOutput = await collectStream(output);
  } catch (err) {
    throw new Error(`Replicate call failed: ${err}`);
  }

  let data: Record<string, unknown>;
  try {
    data = JSON.parse(stripFences(rawOutput));
  } catch (err) {
    console.error(
      `Failed to parse LLM invoice JSON: ${err}\nRaw output (first 500): ${rawOutput.slice(0, 500)}`
    );
    data = { delivery_note_ingredients: [] };
  }

  const headerData: Record<string, unknown> = {
    supplier_name: data.supplier_name ?? undefined,
    date: data.date ?? undefined,
    serial_number: data.serial_number ?? undefined,
    total_vat: toFloat(data.total_vat),
    total_discount: toFloat(data.total_discount),
    total_expense: toFloat(data.total_expense),
  };

  const ingredients: DeliveryNoteIngredient[] = (
    (data.delivery_note_ingredients as unknown[]) ?? []
  )
    .filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null)
    .map(parseIngredient);

  const flaggedCount = ingredients.filter((i) => i.requires_review).length;
  console.info(
    `LLM extracted: supplier='${headerData.supplier_name}', date='${headerData.date}', ` +
      `${ingredients.length} ingredient(s), ${flaggedCount} flagged.`
  );

  return [headerData, ingredients];
}
