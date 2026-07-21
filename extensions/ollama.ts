/**
 * pi-ocr — Ollama backend
 *
 * Uses any locally-running Ollama vision model (default: glm-ocr) to OCR
 * images and PDFs. Converts PDF pages to PNG before sending to Ollama.
 */

import { readFileSync, existsSync, mkdtempSync, readdirSync, unlinkSync, rmdirSync } from "node:fs";
import { extname, join } from "node:path";
import { tmpdir } from "node:os";
import { spawn } from "node:child_process";
import type { Task, OcrResult, OcrProgressCallback } from "./types";

// ── Helpers ──────────────────────────────────────────────────────────────────

export function isImage(filePath: string): boolean {
  const ext = extname(filePath).toLowerCase();
  return [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"].includes(ext);
}

export function isPdf(filePath: string): boolean {
  const ext = extname(filePath).toLowerCase();
  if (ext === ".pdf") return true;
  try {
    const buf = readFileSync(filePath).subarray(0, 4);
    return buf.toString() === "%PDF";
  } catch {
    return false;
  }
}

function execCmdCapture(cmd: string, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { stdio: ["ignore", "pipe", "pipe"] });
    const outChunks: Buffer[] = [];
    const errChunks: Buffer[] = [];
    child.stdout.on("data", (d) => outChunks.push(d));
    child.stderr.on("data", (d) => errChunks.push(d));
    child.on("error", (e) => {
      reject(new Error(`${cmd}: ${(e as any).code === "ENOENT" ? "command not found" : e.message}`));
    });
    child.on("close", (code) => {
      const stderr = Buffer.concat(errChunks).toString("utf8").trim();
      if (code === 0) {
        resolve(Buffer.concat(outChunks).toString("utf8"));
      } else {
        reject(new Error(`${cmd} exited with code ${code}${stderr ? ": " + stderr : ""}`));
      }
    });
  });
}

function cleanupDir(dir: string) {
  try {
    for (const f of readdirSync(dir)) unlinkSync(join(dir, f));
    rmdirSync(dir);
  } catch { /* best effort */ }
}

function buildPrompt(task: Task): string {
  switch (task) {
    case "text": return "Text Recognition";
    case "formula": return "Formula Recognition";
    case "table": return "Table Recognition";
    case "figure": return "Figure Recognition";
    case "auto":
      return "Recognize all text, formulas, tables, and figures in this document. Output formulas in LaTeX format, tables in Markdown format.";
  }
}

// ── PDF helpers ──────────────────────────────────────────────────────────────

export async function getPdfPageCount(pdfPath: string): Promise<number> {
  if (process.platform === "darwin") {
    try {
      const out = await execCmdCapture("mdls", ["-name", "kMDItemNumberOfPages", "-raw", pdfPath]);
      const n = parseInt(out.trim(), 10);
      if (!isNaN(n) && n > 0) return n;
    } catch { /* fall through */ }
  } else if (process.platform === "linux") {
    try {
      const out = await execCmdCapture("pdfinfo", [pdfPath]);
      const m = out.match(/Pages:\s+(\d+)/);
      if (m) return parseInt(m[1], 10) || 1;
    } catch { /* fall through */ }
  }
  return 1;
}

async function convertPdfPageMac(pdfPath: string, pageIndex: number, outPath: string): Promise<void> {
  if (pageIndex === 0) {
    try {
      await execCmdCapture("sips", ["-s", "format", "png", pdfPath, "--out", outPath]);
      return;
    } catch (e: any) {
      throw new Error(`sips PDF conversion failed: ${e.message}`);
    }
  }
  try {
    await execCmdCapture("pdftoppm", [
      "-png", "-r", "150", "-f", String(pageIndex + 1), "-l", String(pageIndex + 1),
      "-singlefile", pdfPath, outPath.replace(/\.png$/, ""),
    ]);
    if (!existsSync(outPath) || readFileSync(outPath).length === 0) {
      throw new Error(`pdftoppm produced no output for page ${pageIndex + 1}`);
    }
    return;
  } catch (e: any) {
    const msg = e.message || String(e);
    if (msg.includes("command not found") || msg.includes("ENOENT")) {
      throw new Error(`pdftoppm not found. Install with: brew install poppler. Only page 1 was processed with sips.`);
    }
    throw new Error(`PDF page ${pageIndex + 1} conversion failed: ${msg}`);
  }
}

async function convertPdfPage(pdfPath: string, pageIndex: number, outPath: string): Promise<void> {
  if (process.platform === "darwin") {
    await convertPdfPageMac(pdfPath, pageIndex, outPath);
  } else if (process.platform === "linux") {
    await execCmdCapture("pdftoppm", [
      "-png", "-r", "150", "-f", String(pageIndex + 1), "-l", String(pageIndex + 1),
      "-singlefile", pdfPath, outPath.replace(/\.png$/, ""),
    ]);
  }
}

// ── Ollama API call ──────────────────────────────────────────────────────────

async function callOllama(
  host: string, imagePath: string, signal: AbortSignal | undefined, model: string,
  numCtx?: number,
): Promise<string> {
  const imageBase64 = readFileSync(imagePath).toString("base64");
  const prompt = buildPrompt("auto");

  // Only set options.num_ctx when explicitly configured; otherwise let Ollama use its default.
  const options = numCtx ? { num_ctx: numCtx } : undefined;
  const body = JSON.stringify({
    model, prompt, images: [imageBase64], stream: false,
    ...(options ? { options } : {}),
  });

  const response = await fetch(`${host}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    signal,
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`Ollama API error ${response.status}: ${text.slice(0, 200)}. Is Ollama running and is the ${model} model pulled?`);
  }

  const data = (await response.json()) as { response?: string; error?: string };
  if (data.error) throw new Error(`OCR error: ${data.error}`);
  return data.response?.trim() || "";
}

// ── Public API ───────────────────────────────────────────────────────────────

export async function ollamaOcr(
  filePath: string, ollamaHost: string, model: string,
  signal: AbortSignal | undefined, onProgress: OcrProgressCallback,
  numCtx?: number,
): Promise<OcrResult> {
  let resultText = "";
  let tmpDir: string | null = null;

  try {
    if (isPdf(filePath)) {
      onProgress("📄 Converting PDF pages to images…");
      tmpDir = mkdtempSync(join(tmpdir(), "pi-ocr-"));
      const pageCount = await getPdfPageCount(filePath);

      // Proactive check for multi-page PDF on macOS without pdftoppm
      if (pageCount > 1 && process.platform === "darwin") {
        try { await execCmdCapture("pdftoppm", ["-v"]); } catch {
          onProgress(`⚠️ Multi-page PDF (${pageCount} pages) but pdftoppm is not installed. Only page 1 will be processed.\nInstall: brew install poppler`);
        }
      }

      const pageResults: string[] = [];
      for (let i = 0; i < pageCount; i++) {
        if (signal?.aborted) throw new Error("Aborted");
        const pageOut = join(tmpDir, `page_${i + 1}.png`);

        try {
          await convertPdfPage(filePath, i, pageOut);
        } catch (e: any) {
          pageResults.push(`## Page ${i + 1}\n\n> ⚠️ Skipped: ${e.message}`);
          continue;
        }

        onProgress(`🔍 OCR page ${i + 1}/${pageCount}…`);
        const pageText = await callOllama(ollamaHost, pageOut, signal, model, numCtx);
        if (!pageText.trim()) {
          pageResults.push(`## Page ${i + 1}\n\n> ⚠️ OCR returned empty result for this page.`);
        } else {
          pageResults.push(`## Page ${i + 1}\n\n${pageText}`);
        }
      }
      resultText = pageResults.join("\n\n");
    } else {
      resultText = await callOllama(ollamaHost, filePath, signal, model, numCtx);
    }

    return { text: resultText, details: { backend: "ollama", model, ...(numCtx ? { numCtx } : {}) } };
  } finally {
    if (tmpDir) cleanupDir(tmpDir);
  }
}

/** Check if a model exists locally via Ollama API */
export async function ollamaCheckModel(host: string, model: string): Promise<boolean> {
  try {
    const resp = await fetch(`${host}/api/show`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: model }),
    });
    return resp.ok;
  } catch {
    return false;
  }
}

/** Pull a model via ollama pull */
export function ollamaPullModel(model: string): Promise<void> {
  return execCmdCapture("ollama", ["pull", model]).then(() => {});
}
