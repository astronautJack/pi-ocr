/**
 * pi-ocr — Multi-backend OCR for Pi Coding Agent
 *
 * Registers a `pi_ocr` tool that the LLM can call to read images and PDFs
 * using one of three backends:
 *   - Ollama (local vision models like glm-ocr)
 *   - MinerU API (free Agent API, ≤10MB, ≤20 pages)
 *   - Pix2Text (local Python library)
 *
 * Single command:
 *   /ocr                    → open settings UI (backend, model, split toggle)
 *   /ocr <file> [task]      → OCR file with current settings
 *
 * Settings persisted to ~/.pi/agent/settings.json.
 *
 * Prerequisites:
 *   Ollama:     brew install ollama && ollama pull glm-ocr
 *   MinerU:     no setup (free API, IP rate-limited)
 *   Pix2Text:  pip install pix2text
 *   PDF tools:  brew install poppler (macOS multi-page PDF for Ollama)
 *
 * Install: pi install npm:pi-ocr
 */

import { Type } from "@earendil-works/pi-ai";
import {
	defineTool,
	getSettingsListTheme,
	type ExtensionAPI,
	type ExtensionContext,
} from "@earendil-works/pi-coding-agent";
import {
	Container,
	Input,
	Key,
	matchesKey,
	Text,
	type SettingItem,
	SettingsList,
	type SelectItem,
	SelectList,
} from "@earendil-works/pi-tui";
import {
	existsSync,
	readFileSync,
	writeFileSync,
	mkdirSync,
	readdirSync,
	unlinkSync,
} from "node:fs";
import { basename, extname, dirname, join } from "node:path";
import { homedir, tmpdir } from "node:os";

import type { Backend, OcrConfig } from "./types";
import { BACKENDS } from "./types";
import {
	isImage,
	isPdf,
	ollamaOcr,
	ollamaCheckModel,
	ollamaPullModel,
} from "./ollama";
import { mineruOcr } from "./mineru";
import { tesseractOcr } from "./tesseract";
import { pix2textOcr } from "./pix2text";
import { mineruProOcr } from "./mineru-pro";

// ── Config persistence ───────────────────────────────────────────────────────

const SETTINGS_PATH = join(homedir(), ".pi", "agent", "settings.json");

function loadOcrConfig(): Partial<OcrConfig> {
	try {
		if (!existsSync(SETTINGS_PATH)) return {};
		const settings = JSON.parse(readFileSync(SETTINGS_PATH, "utf8"));
		// Migrate from old key (pi-minimodel-ocr) to new key (pi-ocr)
		const old = (settings as any).minimodelOcr;
		const current = (settings as any).piOcr;
		if (old && !current) {
			(settings as any).piOcr = old;
			delete (settings as any).minimodelOcr;
			writeFileSync(
				SETTINGS_PATH,
				JSON.stringify(settings, null, 2) + "\n",
				"utf8",
			);
		}
		return (settings as any).piOcr || {};
	} catch {
		return {};
	}
}

function saveOcrConfig(updates: Partial<OcrConfig>) {
	try {
		const dir = dirname(SETTINGS_PATH);
		if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
		const settings = existsSync(SETTINGS_PATH)
			? JSON.parse(readFileSync(SETTINGS_PATH, "utf8"))
			: {};
		settings.piOcr = { ...(settings.piOcr || {}), ...updates };
		writeFileSync(
			SETTINGS_PATH,
			JSON.stringify(settings, null, 2) + "\n",
			"utf8",
		);
	} catch {
		/* best effort */
	}
}

function getConfig(): OcrConfig {
	const s = loadOcrConfig();
	const envNumCtx = process.env.OCR_NUM_CTX ? Number(process.env.OCR_NUM_CTX) : undefined;
	const numCtx = envNumCtx && !isNaN(envNumCtx)
		? envNumCtx
		: typeof s.numCtx === "number" ? s.numCtx : undefined;
	return {
		backend: (BACKENDS.includes(s.backend as Backend)
			? s.backend
			: "mineru") as Backend,
		ollamaHost:
			process.env.OLLAMA_HOST || s.ollamaHost || "http://localhost:11434",
		model: process.env.OCR_MODEL || s.model || "glm-ocr",
		mineruSplitPdf: s.mineruSplitPdf !== false,
		mineruToken: s.mineruToken,
		numCtx,
	};
}

// ── Recommended models ───────────────────────────────────────────────────────

const RECOMMENDED_MODELS = [
	{ name: "glm-ocr:q8_0", desc: "balanced — smallest (1.6GB), fast" },
	{ name: "glm-ocr", desc: "best formula OCR (2.2GB, 94.6 OmniDocBench)" },
	{ name: "minicpm-v", desc: "strong all-around vision + OCR (8B, 5.5GB)" },
	{ name: "llama3.2-vision", desc: "Meta's vision model (11B)" },
];

// ── Tool Definition ──────────────────────────────────────────────────────────

const ocrSchema = Type.Object({
	path: Type.String({
		description:
			"Absolute or relative path to the image or PDF file to OCR. Supported formats: PNG, JPG, GIF, WEBP, BMP, TIFF, PDF.",
	}),
});

const ocrTool = defineTool({
	name: "pi_ocr",
	label: "Minimodel OCR",
	description:
		"Extract text, math formulas (LaTeX), and tables from images or PDFs. " +
		"Multi-backend: MinerU (free cloud, ≤10MB, ≤20pp), MinerU Pro (vlm, token), Ollama (local GPU), Pix2Text (local Python), Tesseract (classic). " +
		"Use this when the user asks about the content of an image or PDF. " +
		"Works with non-vision LLMs like DeepSeek that cannot process images directly. " +
		"Backend and model are configured by the user via /ocr.",
	promptSnippet: "Extract text/formulas/tables from images and PDFs",
	promptGuidelines: [
		"When the user asks about the content of an image or PDF, use pi_ocr to extract the text first.",
		"Only path is needed — the backend and model are managed by the user via /ocr settings.",
		"The OCR result includes a **Backend:** field telling you which backend processed the request.",
	],
	parameters: ocrSchema,
	async execute(_toolCallId, params, signal, onUpdate, _ctx) {
		const { path: filePath } = params as { path: string };
		const config = getConfig();
		const resolvedModel = config.model;

		if (!existsSync(filePath)) throw new Error(`File not found: ${filePath}`);
		if (!isImage(filePath) && !isPdf(filePath)) {
			throw new Error(
				`Unsupported file type "${extname(filePath)}". Supported: PNG, JPG, GIF, WEBP, BMP, TIFF, PDF.`,
			);
		}

		const backendLabel = {
			mineru: "☁️ MinerU",
			"mineru-pro": "☁️ MinerU Pro",
			ollama: "🦙 Ollama",
			tesseract: "🔤 Tesseract",
			pix2text: "📐 Pix2Text",
		}[config.backend];
		onUpdate?.({
			content: [
				{
					type: "text",
					text: `🔍 OCR ${basename(filePath)} via ${backendLabel}…`,
				},
			],
			details: {},
		});

		const onProgress = (msg: string) =>
			onUpdate?.({ content: [{ type: "text", text: msg }], details: {} });

		try {
			let result: { text: string; details: Record<string, unknown> };

			switch (config.backend) {
				case "ollama":
					result = await ollamaOcr(
						filePath,
						config.ollamaHost,
						resolvedModel,
						signal,
						onProgress,
						config.numCtx,
					);
					break;
				case "mineru": {
					const { stat } = await import("node:fs/promises");
					const stats = await stat(filePath);
					if (stats.size > 10 * 1024 * 1024) {
						onProgress(
							`⚠️ File is ${(stats.size / 1024 / 1024).toFixed(1)}MB. MinerU free tier limit is 10MB.\n💡 Compress at https://ilovepdf.com/compress_pdf or switch backend with /ocr.`,
						);
					}
					result = await mineruOcr(
						filePath,
						config.mineruSplitPdf,
						signal,
						onProgress,
					);
					break;
				}
				case "mineru-pro": {
					const token = config.mineruToken || process.env.MINERU_TOKEN;
					if (!token)
						throw new Error(
							"MinerU Pro requires a token. Get one at https://mineru.net/apiManage, then set it with /ocr settings.",
						);
					result = await mineruProOcr(filePath, token, signal, onProgress);
					break;
				}
				case "tesseract":
					result = await tesseractOcr(filePath, signal, onProgress);
					break;
				case "pix2text":
					result = await pix2textOcr(filePath, signal, onProgress);
					break;
				default:
					throw new Error(`Unknown backend "${config.backend}"`);
			}

			const totalChars = result.text.length;
			const tooLarge = totalChars > 2000;

			let outputFile: string | undefined;
			let outputMode: "inline" | "file";

			if (tooLarge) {
				const ext = extname(filePath).toLowerCase() === ".pdf" ? ".md" : ".txt";
				outputFile = join(tmpdir(), `pi-ocr-${Date.now()}${ext}`);
				writeFileSync(outputFile, result.text, "utf8");
				outputMode = "file";
			} else {
				outputMode = "inline";
			}

			const header = [
				`## OCR Result`,
				``,
				`**File:** \`${basename(filePath)}\``,
				`**Backend:** ${config.backend}`,
				`**Chars:** ${totalChars.toLocaleString()}`,
			];
			if (result.details && typeof result.details.pages === "number") {
				header.push(`**Pages:** ${result.details.pages}`);
			}
			if (outputMode === "file" && outputFile) {
				header.push(
					`**Output:** file → \`${outputFile}\` (${totalChars.toLocaleString()} chars)`,
				);
			} else {
				header.push(`**Output:** inline`);
				header.push(``);
			}

			return {
				content: [
					{
						type: "text",
						text:
							outputMode === "file"
								? header.join("\n")
								: [...header, result.text].join("\n"),
					},
				],
				details: {
					...result.details,
					task: "auto",
					path: filePath,
					fullText: result.text,
					truncated: tooLarge,
					outputMode,
					backend: config.backend,
					...(outputFile ? { outputFile } : {}),
				},
			};
		} catch (e: any) {
			const msg = e.message || String(e);
			let hint = "";
			if (
				config.backend === "ollama" &&
				(msg.includes("fetch failed") || msg.includes("ECONNREFUSED"))
			)
				hint = "\n\n💡 Is Ollama running? Start: `ollama serve`";
			else if (config.backend === "tesseract" && msg.includes("not found"))
				hint =
					"\n\n💡 Install: `brew install tesseract` (macOS) or `sudo apt install tesseract-ocr` (Linux)";
			else if (config.backend === "pix2text" && msg.includes("python3"))
				hint = "\n\n💡 Install: `pip install pix2text`";
			else if (config.backend === "mineru" && msg.includes("429"))
				hint =
					"\n\n💡 MinerU rate limit. Wait a minute or switch backend with /ocr.";
			else if (config.backend === "mineru" && msg.includes("too large"))
				hint =
					"\n\n💡 Compress at https://ilovepdf.com/compress_pdf or switch backend.";
			throw new Error(`OCR error (${config.backend}): ${msg}${hint}`);
		}
	},
});

// ── Extension Entry ─────────────────────────────────────────────────────────

export default function ocrExtension(pi: ExtensionAPI) {
	pi.registerTool(ocrTool);

	// ── /ocr command ─────────────────────────────────────────────────────────

	pi.registerCommand("ocr", {
		description: "OCR an image or PDF, or configure OCR settings",
		handler: async (args, ctx) => {
			const trimmed = (args || "").trim();

			// No args → open settings UI
			if (!trimmed) {
				await showOcrSettings(ctx);
				return;
			}

			// Args → OCR a file
			const filePath = trimmed.trim();

			if (!existsSync(filePath)) {
				ctx.ui.notify(`File not found: ${filePath}`, "error");
				return;
			}

			try {
				const result = await ocrTool.execute(
					"",
					{ path: filePath },
					undefined as any,
					undefined,
					ctx,
				);
				const textLen = (result.details as any)?.fullText?.length || 0;
				ctx.ui.notify(
					`OCR complete — ${textLen} chars via ${(result.details as any)?.backend || "?"}`,
					"info",
				);
			} catch (e: any) {
				ctx.ui.notify(e.message?.slice(0, 200) || "OCR failed", "error");
			}
		},
	});

	// ── Settings UI ────────────────────────────────────────────────────────────
	//
	// Shows a SettingsList with:
	//   1. Backend selector (toggle: ollama / mineru / pix2text)
	//   2. MinerU: Split PDF >20 pages (toggle: ON / OFF)
	//   3. Ollama model (current value shown; Enter opens model picker submenu)
	//
	// Changes are saved immediately to ~/.pi/agent/settings.json.

	async function showOcrSettings(ctx: ExtensionContext) {
		const config = getConfig();

		// Build token info line
		const tokenLabel = config.mineruToken
			? `●●● configured (${config.mineruToken.slice(-6)})`
			: "not set — save to: ~/.pi/agent/settings.json → piOcr.mineruToken";

		const items: SettingItem[] = [
			{
				id: "backend",
				label: "OCR Backend",
				description:
					"Ollama=local GPU, MinerU=free cloud API, Pix2Text=local Python",
				currentValue: config.backend,
				values: [...BACKENDS],
			},
			{
				id: "mineruSplitPdf",
				label: "MinerU: Split PDF >20 pages",
				description: "Auto-split large PDFs into ≤20-page free-tier chunks",
				currentValue: config.mineruSplitPdf ? "ON" : "OFF",
				values: ["ON", "OFF"],
			},
			{
				id: "mineruToken",
				label: "MinerU Pro Token",
				description: tokenLabel,
				currentValue: config.mineruToken ? "configured" : "not set",
				submenu: (_currentValue, done) => {
					return createTokenInput(config.mineruToken || "", ctx, (token) => {
						if (token !== undefined) {
							saveOcrConfig({ mineruToken: token || undefined });
							settingsListRef?.updateValue(
								"mineruToken",
								token ? "configured" : "not set",
							);
						}
						done(token);
					});
				},
			},
			{
				id: "model",
				label: "Ollama Model",
				description:
					"Vision model used for OCR (only applies to Ollama backend)",
				currentValue: config.model,
				submenu: (_currentValue, done) => {
					return createModelSelector(config.model, ctx, (selected) => {
						if (selected) {
							saveOcrConfig({ model: selected });
							process.env.OCR_MODEL = selected;
							updateStatus(ctx);
							settingsListRef?.updateValue("model", selected);
						}
						done(selected);
					});
				},
			},
			{
				id: "numCtx",
				label: "Ollama num_ctx",
				description:
					"Context window size for OCR (only applies to Ollama backend)",
				currentValue: config.numCtx ? String(config.numCtx) : "default",
				submenu: (_currentValue, done) => {
					return createNumCtxInput(config.numCtx, ctx, (value) => {
						if (value !== undefined) {
							saveOcrConfig({ numCtx: value });
							if (value) process.env.OCR_NUM_CTX = String(value);
							else delete process.env.OCR_NUM_CTX;
							settingsListRef?.updateValue(
								"numCtx",
								value ? String(value) : "default",
							);
						}
						done(value !== undefined ? String(value) : undefined);
					});
				},
			},
			{
				id: "clearCache",
				label: "Clear OCR temp files",
				description: clearCacheLabel(),
				currentValue: "",
				submenu: (_currentValue, done) => {
					return createClearCacheDialog(ctx, () => {
						done(undefined);
					});
				},
			},
		];

		let settingsListRef: SettingsList | null = null;

		await ctx.ui.custom((tui, theme, _kb, done) => {
			const settingsList = new SettingsList(
				items,
				8, // max visible items
				getSettingsListTheme(),
				(id, newValue) => {
					// onChange — save immediately
					switch (id) {
						case "backend": {
							const backend = BACKENDS.includes(newValue as Backend)
								? (newValue as Backend)
								: "ollama";
							saveOcrConfig({ backend });
							updateStatus(ctx);
							// Show hints when switching
							if (backend === "mineru-pro") {
								ctx.ui.notify(
									"☁️ MinerU Pro: vlm model, ≤200MB, ≤200 pages.\nToken → ~/.pi/agent/settings.json → piOcr.mineruToken\nGet one at https://mineru.net/apiManage",
									"info",
								);
							} else if (backend === "mineru") {
								ctx.ui.notify(
									"☁️ MinerU: free for ≤10MB & ≤20 pages. Auto-split " +
										(config.mineruSplitPdf
											? "ON"
											: "OFF — enable in settings") +
										".\nLarge files? Compress at https://ilovepdf.com/compress_pdf",
									"info",
								);
							} else if (backend === "tesseract") {
								ctx.ui.notify(
									"🔤 Tesseract: `brew install tesseract` (macOS) or `sudo apt install tesseract-ocr` (Linux). ~30MB, CPU-only.",
									"warning",
								);
							} else if (backend === "pix2text") {
								ctx.ui.notify(
									"🐍 Pix2Text: needs `pip install pix2text`",
									"warning",
								);
							}
							break;
						}
						case "mineruSplitPdf":
							saveOcrConfig({ mineruSplitPdf: newValue === "ON" });
							break;
					}
				},
				() => done(undefined), // onCancel
			);

			settingsListRef = settingsList;

			const container = new Container();
			container.addChild(
				new Text(theme.fg("accent", theme.bold("OCR Settings")), 1, 0),
			);
			container.addChild(settingsList);
			container.addChild(
				new Text(
					theme.fg(
						"dim",
						"↑↓ navigate • ← → toggle • enter select • esc close",
					),
					1,
					0,
				),
			);

			return {
				render(width: number) {
					return container.render(width);
				},
				invalidate() {
					container.invalidate();
				},
				handleInput(data: string) {
					settingsList.handleInput(data);
					tui.requestRender();
				},
			};
		});
	}

	// ── Model selector submenu ─────────────────────────────────────────────────

	// Submenu states: "list" | "input" | "confirm-pull" | "pulling"
	interface ModelSubmenuState {
		mode: "list" | "input" | "confirm-pull" | "pulling";
		selectedModel: string;
		confirmYes: boolean;
	}

	function createModelSelector(
		currentModel: string,
		ctx: ExtensionContext,
		onDone: (selected: string | undefined) => void,
	) {
		const items: SelectItem[] = RECOMMENDED_MODELS.map((m) => ({
			value: m.name,
			label: m.name === currentModel ? `${m.name} ✓` : m.name,
			description: m.desc,
		}));
		items.push({
			value: "__custom__",
			label: "Type a custom name…",
			description: "Enter any Ollama model name",
		});

		const state: ModelSubmenuState = {
			mode: "list",
			selectedModel: "",
			confirmYes: true,
		};

		// Notifications to fire AFTER component is dismissed (avoid modal-in-modal)
		const pendingNotify: Array<{
			msg: string;
			level: "info" | "error" | "warning";
		}> = [];

		const theme = ctx.ui.theme;

		// Inline input for custom model name
		const nameInput = new Input();

		const container = new Container();
		container.addChild(new Text("Choose Ollama Model", 1, 0));

		const selectList = new SelectList(items, Math.min(items.length, 8), {
			selectedPrefix: (text) => theme.fg("accent", text),
			selectedText: (text) => theme.fg("accent", text),
			description: (text) => theme.fg("muted", text),
			scrollInfo: (text) => theme.fg("dim", text),
			noMatch: (text) => theme.fg("warning", text),
		});

		// Fire pending notifications and call onDone
		function finalize(selected: string | undefined) {
			for (const n of pendingNotify) {
				ctx.ui.notify(n.msg, n.level);
			}
			onDone(selected);
		}

		selectList.onSelect = async (item) => {
			if (item.value === "__custom__") {
				state.mode = "input";
				nameInput.setValue(currentModel);
				return;
			}
			// Check if model needs pulling
			const config = getConfig();
			const exists = await ollamaCheckModel(config.ollamaHost, item.value);
			if (!exists) {
				state.mode = "confirm-pull";
				state.selectedModel = item.value;
				state.confirmYes = true;
				return;
			}
			finalize(item.value);
		};

		selectList.onCancel = () => finalize(undefined);
		container.addChild(selectList);

		return {
			render(width: number) {
				if (state.mode === "input") {
					const lines: string[] = [];
					const add = (s: string) => lines.push(s);
					add(theme.fg("accent", "─".repeat(width)));
					add("");
					add(theme.fg("text", " Enter custom Ollama model name:"));
					add("");
					for (const line of nameInput.render(width - 4)) {
						add(`  ${line}`);
					}
					add("");
					add(theme.fg("dim", " Enter to confirm · Esc to cancel"));
					add(theme.fg("accent", "─".repeat(width)));
					return lines;
				}

				if (state.mode === "confirm-pull") {
					const lines: string[] = [];
					const add = (s: string) => lines.push(s);
					add(theme.fg("accent", "─".repeat(width)));
					add("");
					add(
						theme.fg(
							"warning",
							` Model "${state.selectedModel}" is not pulled locally.`,
						),
					);
					add("");
					add(
						theme.fg(
							"text",
							` Pull it now? (ollama pull ${state.selectedModel})`,
						),
					);
					add("");
					const yesStyle = state.confirmYes ? theme.fg("accent", ">") : " ";
					const noStyle = !state.confirmYes ? theme.fg("accent", ">") : " ";
					add(
						`  ${yesStyle} ${state.confirmYes ? theme.fg("accent", "Yes") : theme.fg("muted", "Yes")}  ${noStyle} ${!state.confirmYes ? theme.fg("accent", "No") : theme.fg("muted", "No")}`,
					);
					add("");
					add(theme.fg("dim", " ← → toggle · Enter confirm · Esc cancel"));
					add(theme.fg("accent", "─".repeat(width)));
					return lines;
				}

				if (state.mode === "pulling") {
					const lines: string[] = [];
					const add = (s: string) => lines.push(s);
					add(theme.fg("accent", "─".repeat(width)));
					add("");
					add(theme.fg("text", ` Pulling ${state.selectedModel}…`));
					add("");
					add(theme.fg("dim", " Please wait…"));
					add(theme.fg("accent", "─".repeat(width)));
					return lines;
				}

				return container.render(width);
			},
			invalidate() {
				container.invalidate();
			},
			handleInput(data: string) {
				if (state.mode === "input") {
					if (matchesKey(data, Key.escape)) {
						state.mode = "list";
						nameInput.setValue("");
						return;
					}
					if (matchesKey(data, Key.enter)) {
						const value = nameInput.getValue().trim();
						if (!value) {
							state.mode = "list";
							return;
						}
						state.selectedModel = value;
						// Check if model exists — async, but we'll handle in next frame
						const config = getConfig();
						ollamaCheckModel(config.ollamaHost, value).then((exists) => {
							if (!exists) {
								state.mode = "confirm-pull";
								state.confirmYes = true;
							} else {
								finalize(value);
							}
						});
						state.mode = "pulling"; // brief loading state
						return;
					}
					nameInput.handleInput(data);
					return;
				}

				if (state.mode === "confirm-pull") {
					if (matchesKey(data, Key.escape)) {
						state.mode = "list";
						return;
					}
					if (matchesKey(data, Key.left) || matchesKey(data, Key.right)) {
						state.confirmYes = !state.confirmYes;
						return;
					}
					if (matchesKey(data, Key.enter)) {
						if (state.confirmYes) {
							pendingNotify.push({
								msg: `Pulling ${state.selectedModel}…`,
								level: "info",
							});
							ollamaPullModel(state.selectedModel)
								.then(() => {
									ctx.ui.notify(`${state.selectedModel} ready`, "info");
								})
								.catch((e: any) => {
									ctx.ui.notify(
										`Pull failed: ${e.message}`.slice(0, 200),
										"error",
									);
								});
							finalize(state.selectedModel);
						} else {
							state.mode = "list";
						}
						return;
					}
					return;
				}

				if (state.mode === "pulling") {
					// Block all input while loading
					return;
				}

				selectList.handleInput(data);
			},
		};
	}

	// ── Token input submenu (inline Input — no ctx.ui.input) ──────────────

	function createTokenInput(
		currentToken: string,
		ctx: ExtensionContext,
		onDone: (token: string | undefined) => void,
	) {
		const theme = ctx.ui.theme;
		const input = new Input();
		// Leave input blank so user can paste new token directly — current token shown above as masked

		return {
			render(width: number) {
				const lines: string[] = [];
				const add = (s: string) => lines.push(s);
				const masked = currentToken
					? currentToken.slice(0, 10) + "…" + currentToken.slice(-6)
					: "not set";
				add(theme.fg("accent", "─".repeat(width)));
				add("");
				add(
					theme.fg("text", " MinerU Pro API token from mineru.net/apiManage"),
				);
				add("");
				if (currentToken) {
					add(theme.fg("muted", ` Current: ${masked}`));
					add("");
				}
				add(theme.fg("text", " Paste new token:"));
				add("");
				for (const line of input.render(width - 4)) {
					add(`  ${line}`);
				}
				add("");
				add(theme.fg("dim", " Enter to confirm · Esc to cancel"));
				add(theme.fg("accent", "─".repeat(width)));
				return lines;
			},
			invalidate() {},
			handleInput(data: string) {
				if (matchesKey(data, Key.escape)) {
					onDone(undefined);
					return;
				}
				if (matchesKey(data, Key.enter)) {
					const value = input.getValue().trim();
					onDone(value || undefined);
					return;
				}
				input.handleInput(data);
			},
		};
	}

	function createNumCtxInput(
		currentNumCtx: number | undefined,
		ctx: ExtensionContext,
		onDone: (value: number | undefined) => void,
	) {
		const theme = ctx.ui.theme;
		const input = new Input();
		if (currentNumCtx) input.setValue(String(currentNumCtx));

		return {
			render(width: number) {
				const lines: string[] = [];
				const add = (s: string) => lines.push(s);
				add(theme.fg("accent", "─".repeat(width)));
				add("");
				add(
					theme.fg(
						"text",
						" Ollama num_ctx — context window size for OCR requests",
					),
				);
				add("");
				add(theme.fg("muted", ` Current: ${currentNumCtx ?? "default (Ollama decides)"}`));
				add("");
				add(theme.fg("text", " New value (blank = use Ollama default):"));
				add("");
				for (const line of input.render(width - 4)) {
					add(`  ${line}`);
				}
				add("");
				add(theme.fg("dim", " Enter to confirm · Esc to cancel"));
				add(theme.fg("accent", "─".repeat(width)));
				return lines;
			},
			invalidate() {},
			handleInput(data: string) {
				if (matchesKey(data, Key.escape)) {
					onDone(undefined);
					return;
				}
				if (matchesKey(data, Key.enter)) {
					const raw = input.getValue().trim();
					if (!raw) {
						onDone(undefined); // blank = keep default (unset)
						return;
					}
					const n = Number(raw);
					if (isNaN(n) || n <= 0 || !Number.isFinite(n)) {
						ctx.ui.notify("num_ctx must be a positive integer", "warning");
						return;
					}
					onDone(Math.floor(n));
					return;
				}
				input.handleInput(data);
			},
		};
	}

	// ── Cache cleanup ──────────────────────────────────────────────────────────

	/** Only scan tmpdir(), only match pi-ocr-*.md / pi-ocr-*.txt, only unlink files. */
	function clearCacheLabel(): string {
		try {
			const files = readdirSync(tmpdir()).filter(
				(f) =>
					f.startsWith("pi-ocr-") && (f.endsWith(".md") || f.endsWith(".txt")),
			);
			return files.length === 0
				? "no temp files"
				: `${files.length} file${files.length > 1 ? "s" : ""}`;
		} catch {
			return "unable to scan";
		}
	}

	function createClearCacheDialog(
		ctx: ExtensionContext,
		onCleared: () => void,
	) {
		const theme = ctx.ui.theme;
		let confirmed = false;

		function countFiles(): { files: string[]; totalSize: number } {
			try {
				const dir = tmpdir();
				const files = readdirSync(dir).filter(
					(f) =>
						f.startsWith("pi-ocr-") &&
						(f.endsWith(".md") || f.endsWith(".txt")),
				);
				let totalSize = 0;
				const validFiles: string[] = [];
				for (const f of files) {
					try {
						const full = join(dir, f);
						const s = readFileSync(full);
						totalSize += s.length;
						validFiles.push(f);
					} catch {
						/* skip stale/unreadable */
					}
				}
				return { files: validFiles, totalSize };
			} catch {
				return { files: [], totalSize: 0 };
			}
		}

		function doClear() {
			const dir = tmpdir();
			let removed = 0;
			try {
				for (const f of readdirSync(dir)) {
					if (!f.startsWith("pi-ocr-")) continue;
					if (!f.endsWith(".md") && !f.endsWith(".txt")) continue;
					try {
						unlinkSync(join(dir, f));
						removed++;
					} catch {
						/* skip locked files */
					}
				}
			} catch {
				/* dir gone */
			}
			ctx.ui.notify(
				`Cleared ${removed} OCR temp file${removed !== 1 ? "s" : ""}`,
				"info",
			);
			onCleared();
		}

		return {
			render(width: number) {
				const { files, totalSize } = countFiles();
				const sizeStr =
					totalSize > 1024 * 1024
						? `${(totalSize / 1024 / 1024).toFixed(1)} MB`
						: totalSize > 1024
							? `${(totalSize / 1024).toFixed(1)} KB`
							: `${totalSize} B`;

				const lines: string[] = [];
				const add = (s: string) => lines.push(s);
				add(theme.fg("accent", "─".repeat(width)));
				add("");
				if (files.length === 0) {
					add(theme.fg("muted", " No OCR temp files found."));
				} else {
					add(
						theme.fg(
							"text",
							` ${files.length} OCR temp file${files.length > 1 ? "s" : ""} (${sizeStr})`,
						),
					);
					add("");
					add(
						theme.fg(
							"text",
							" Delete these files? They contain cached OCR output.",
						),
					);
					add("");
					const yesStyle = confirmed ? theme.fg("accent", ">") : " ";
					const noStyle = !confirmed ? theme.fg("accent", ">") : " ";
					add(
						`  ${yesStyle} ${confirmed ? theme.fg("accent", "Yes") : theme.fg("muted", "Yes")}  ${noStyle} ${!confirmed ? theme.fg("accent", "No") : theme.fg("muted", "No")}`,
					);
				}
				add("");
				add(theme.fg("dim", " Enter to confirm · Esc to cancel"));
				add(theme.fg("accent", "─".repeat(width)));
				return lines;
			},
			invalidate() {},
			handleInput(data: string) {
				if (matchesKey(data, Key.escape)) {
					onCleared(); // close without clearing
					return;
				}
				if (matchesKey(data, Key.left) || matchesKey(data, Key.right)) {
					confirmed = !confirmed;
					return;
				}
				if (matchesKey(data, Key.enter)) {
					if (confirmed) {
						doClear();
					} else {
						onCleared(); // close without clearing
					}
					return;
				}
			},
		};
	}

	// ── Status bar ─────────────────────────────────────────────────────────────

	function updateStatus(ctx: ExtensionContext) {
		const config = getConfig();
		const text =
			config.backend === "ollama"
				? `OCR: ollama ${config.model}`
				: config.backend === "mineru-pro"
					? "OCR: mineru-pro (vlm)"
					: `OCR: ${config.backend}`;
		ctx.ui.setStatus("pi-ocr", text);
	}

	// ── Startup ────────────────────────────────────────────────────────────────

	pi.on("session_start", async (_event, ctx) => {
		updateStatus(ctx);

		// Proactive check: macOS multi-page PDF support
		if (process.platform === "darwin" && getConfig().backend === "ollama") {
			const { spawn } = await import("node:child_process");
			const hasPdftoppm = await new Promise<boolean>((resolve) => {
				const child = spawn("pdftoppm", ["-v"], { stdio: "ignore" });
				child.on("close", (code) => resolve(code === 0));
				child.on("error", () => resolve(false));
			});
			if (!hasPdftoppm) {
				ctx.ui.notify(
					"💡 Multi-page PDF via Ollama needs pdftoppm: brew install poppler",
					"warning",
				);
			}
		}
	});
}
