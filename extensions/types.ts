/**
 * pi-ocr — shared types for OCR backends
 */

export const TASKS = ["text", "formula", "table", "figure", "auto"] as const;
export type Task = (typeof TASKS)[number];

/** All supported OCR backends */
export const BACKENDS = ["mineru", "mineru-pro", "ollama", "tesseract", "pix2text"] as const;
export type Backend = (typeof BACKENDS)[number];

export interface OcrConfig {
  backend: Backend;
  ollamaHost: string;
  model: string;
  /** MinerU: auto-split PDFs with >20 pages into free-tier chunks */
  mineruSplitPdf: boolean;
  /** MinerU Pro: API token for precision API */
  mineruToken?: string;
  /** Ollama: context window size sent via options.num_ctx. When undefined, Ollama's default is used. */
  numCtx?: number;
}

export interface OcrResult {
  text: string;
  details: Record<string, unknown>;
}

export interface OcrProgressCallback {
  (msg: string): void;
}
