# pi-ocr

> ### ⚡ Zero setup. Works out of the box.
>
> Default backend is **MinerU** — a free cloud API.
> No GPU, no API key, no pip install. Just `pi install` and `/ocr`.

OCR for [Pi Coding Agent](https://pi.dev). Bridges the multimodal gap for non-vision LLMs like DeepSeek: when your model can't see images, `pi_ocr` reads them for you.

---

## Quickstart

```bash
pi install npm:pi-ocr
/ocr ./screenshot.png
/ocr ./paper.pdf
```

**That's all.** MinerU (free cloud API) is the default — zero config.

The `pi_ocr` tool takes only a file `path`. Backend, model, and task are configured by the user via `/ocr` settings — the AI doesn't need to manage them.

---

## Backends

Switch anytime with `/ocr` (no args).

| | Backend | Best for | Setup |
|---|---|---|---|
| ☁️ | **MinerU** (default) | PDFs, general docs | None |
| ☁️ | **MinerU Pro** | Large PDFs, vlm accuracy | API token |
| 🦙 | Ollama | Math formulas → LaTeX | GPU + 2.2GB model |
| 🔤 | Tesseract | Plain text (~30MB) | `brew install tesseract` |
| 📐 | Pix2Text | Math + text, GPU/CPU | `pip install pix2text` |

> 💡 Unsure which backend to pick? See the [benchmark](pi-ocr-benchmark.md) with real test results and the ground truth for comparison.

---

## MinerU (default)

Free cloud API. Images are wrapped as PDF so language-aware OCR applies.

**Limits:** ≤10MB, ≤20 pages/request. PDFs >20 pages auto-split via pypdfium2.

---

## MinerU Pro (vlm model)

Higher accuracy via token-based precision API. **≤200MB, ≤200 pages** — no splitting needed.

Get a free token at [mineru.net/apiManage](https://mineru.net/apiManage), then set it in `/ocr` settings. 1000 pages/day high-priority.

---

## Ollama

Local GPU OCR via [glm-ocr](https://ollama.com) — state-of-the-art formula recognition (94.6 OmniDocBench). Outputs LaTeX.

```bash
# macOS
brew install ollama && ollama pull glm-ocr
brew install poppler   # multi-page PDFs

# Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama pull glm-ocr
sudo apt install poppler-utils
```

### Context window (`num_ctx`)

By default, Ollama uses its model's default context size. For dense pages or
multi-page PDFs, increase it via the `/ocr` **Ollama num_ctx** setting, or in
`~/.pi/agent/settings.json`:

```json
{
  "piOcr": {
    "backend": "ollama",
    "model": "glm-ocr",
    "numCtx": 50000
  }
}
```

An `OCR_NUM_CTX` environment variable overrides both. Leave unset to use
Ollama's default.

---

## Tesseract

Classic OCR engine. Ultra-lightweight (~30MB). **No formula support** — use Ollama or Pix2Text for math.

```bash
brew install tesseract              # macOS
sudo apt install tesseract-ocr      # Linux
```

Supports Chinese: `brew install tesseract-lang` (auto-installed on macOS).

---

## Pix2Text

Mathpix alternative — handles text + formulas on GPU (CUDA/MPS) or CPU. Auto-detects best device.

```bash
pip install pix2text
```

First run downloads ONNX models (~50MB).

---

## Settings

Open with `/ocr` (no args).

| Setting | Description |
|---|---|
| OCR Backend | Switch between MinerU, Ollama, Pix2Text, Tesseract |
| MinerU: Split PDF >20 pages | Auto-split large PDFs into free-tier chunks |
| MinerU Pro Token | API token from mineru.net/apiManage |
| Ollama Model | Vision model (glm-ocr, minicpm-v, etc.) |
| Ollama num_ctx | Context window size for OCR requests (blank = Ollama default) |
| Clear OCR temp files | Remove cached OCR output from /tmp |

---

## Output Behavior

Results ≤2000 chars are returned inline in the tool response. Longer results are written to a temp file (`/tmp/pi-ocr-*.md`); the tool response includes the file path for the AI to read.

---

## Commands

| Command | |
|---|---|
| `/ocr` | Open settings (backend, model, split toggle, clear cache) |
| `/ocr <file>` | OCR a file |
| `/ocr <file> formula` | Math LaTeX output (Ollama backend) |

## Troubleshooting

**MinerU 429** → Wait a minute or switch backend.

**MinerU Pro 401** → Regenerate token at mineru.net/apiManage.

**"Is Ollama running?"** → `ollama serve`

**"pdftoppm not found"** → `brew install poppler` / `sudo apt install poppler-utils`

**"python3 not found" (Pix2Text)** → `pip install pix2text`

**"tesseract not found"** → `brew install tesseract` / `sudo apt install tesseract-ocr`

---

## License

MIT
