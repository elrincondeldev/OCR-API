import { DocumentProcessorServiceClient } from "@google-cloud/documentai";
import { BoundingBox, LineResult, TokenResult } from "../models.js";

const CONFIDENCE_THRESHOLD = parseFloat(
  process.env.CONFIDENCE_THRESHOLD ?? "0.80"
);
const LINE_TOLERANCE = 0.015; // 1.5% of normalised page height

function buildClient(): DocumentProcessorServiceClient {
  return new DocumentProcessorServiceClient({
    apiEndpoint: `${process.env.DOCUMENT_AI_LOCATION!}-documentai.googleapis.com`,
    keyFilename: process.env.GOOGLE_APPLICATION_CREDENTIALS,
  });
}

function extractText(docText: string, element: any): string {
  let result = "";
  for (const seg of element?.layout?.textAnchor?.textSegments ?? []) {
    const start = Number(seg.startIndex ?? 0);
    const end = Number(seg.endIndex ?? 0);
    result += docText.slice(start, end);
  }
  return result.trim();
}

function boundingBoxFromVertices(vertices: any[]): BoundingBox | undefined {
  if (!vertices?.length) return undefined;
  const xs = vertices.map((v) => v.x ?? 0);
  const ys = vertices.map((v) => v.y ?? 0);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  return {
    x: Math.round(xMin * 1e6) / 1e6,
    y: Math.round(yMin * 1e6) / 1e6,
    width: Math.round((xMax - xMin) * 1e6) / 1e6,
    height: Math.round((yMax - yMin) * 1e6) / 1e6,
  };
}

function parseDocument(document: any): LineResult[] {
  const lines: LineResult[] = [];
  const docText: string = document?.text ?? "";

  for (const page of document?.pages ?? []) {
    const tokenData: [number, TokenResult][] = [];

    for (const token of page?.tokens ?? []) {
      const text = extractText(docText, token);
      if (!text) continue;

      const confidence: number = token.layout?.confidence ?? 0;
      const vertices = token.layout?.boundingPoly?.normalizedVertices ?? [];
      const bbox = boundingBoxFromVertices(vertices);
      const yMid = bbox ? bbox.y + bbox.height / 2 : 0;

      tokenData.push([yMid, { text, confidence, bounding_box: bbox }]);
    }

    if (!tokenData.length) continue;

    // Sort top-to-bottom, left-to-right within the same row
    tokenData.sort(([aY, aT], [bY, bT]) => {
      if (Math.abs(aY - bY) > LINE_TOLERANCE) return aY - bY;
      return (aT.bounding_box?.x ?? 0) - (bT.bounding_box?.x ?? 0);
    });

    // Group tokens into lines by Y proximity
    const grouped: TokenResult[][] = [];
    let currentGroup: TokenResult[] = [];
    let currentY: number | null = null;

    for (const [yMid, token] of tokenData) {
      if (currentY === null || Math.abs(yMid - currentY) > LINE_TOLERANCE) {
        if (currentGroup.length) grouped.push(currentGroup);
        currentGroup = [token];
        currentY = yMid;
      } else {
        currentGroup.push(token);
      }
    }
    if (currentGroup.length) grouped.push(currentGroup);

    for (const group of grouped) {
      const lineText = group.map((t) => t.text).join(" ");
      const lowConf = group
        .filter((t) => t.confidence < CONFIDENCE_THRESHOLD)
        .map((t) => t.text);

      const bboxes = group
        .map((t) => t.bounding_box)
        .filter(Boolean) as BoundingBox[];
      let lineBbox: BoundingBox | undefined;
      if (bboxes.length) {
        const xMin = Math.min(...bboxes.map((b) => b.x));
        const yMin = Math.min(...bboxes.map((b) => b.y));
        const xMax = Math.max(...bboxes.map((b) => b.x + b.width));
        const yMax = Math.max(...bboxes.map((b) => b.y + b.height));
        lineBbox = {
          x: xMin,
          y: yMin,
          width: xMax - xMin,
          height: yMax - yMin,
        };
      }

      lines.push({
        line_text: lineText,
        tokens: group,
        low_confidence_tokens: lowConf,
        requires_review: lowConf.length > 0,
        bounding_box: lineBbox,
      });
    }
  }

  return lines;
}

export async function runOcr(
  imageBytes: Buffer
): Promise<[string, LineResult[]]> {
  const client = buildClient();
  const name = client.processorPath(
    process.env.GOOGLE_CLOUD_PROJECT!,
    process.env.DOCUMENT_AI_LOCATION!,
    process.env.DOCUMENT_AI_PROCESSOR_ID!
  );

  console.info("Sending image to Document AI OCR processor.");

  const [response] = await client.processDocument({
    name,
    rawDocument: { content: imageBytes, mimeType: "image/jpeg" },
  });

  const doc = response.document;
  const rawText: string = (doc as any)?.text ?? "";
  const lines = parseDocument(doc);

  const flaggedCount = lines.filter((l) => l.requires_review).length;
  console.info(
    `OCR complete. ${lines.length} lines extracted, ${flaggedCount} flagged.`
  );

  return [rawText, lines];
}
