# Ad Copy Studio (CM3070 Final Project · Template 4.1)

End-to-end prototype: **product image → selling points → audience profile → ranked points → multi-platform ad copy**, with **timestamped JSON logs** for every stage.

| Phase | What it does | Model / technique |
|-------|----------------|-------------------|
| **0** | Vision + heuristics → English selling-point phrases | **MobileNet V2** (TensorFlow / Keras) |
| **1** | Structured **audience JSON** from selling-point text | **Ollama** chat API (`requests`) |
| **2** | Rank top-*k* selling points vs audience tags | Rule-based matcher (keywords; no extra NN) |
| **3** | Per-platform ad copy | **Hugging Face** causal LM (PyTorch `transformers`) |
| **5 (UI)** | Web upload + same orchestration as CLI | `simple_web_ui_phase5.py` (stdlib `http.server`) or optional FastAPI variant |

The single integration entry used by both CLI and web is **`run_phase4_for_one_image`** in **`phase4_runner.py`**.

---

## Requirements

- **Python** 3.10+ recommended (3.8+ may work; tested with 3.10).
- **Ollama** running locally for Phase 1 (default chat endpoint `http://localhost:11434/api/chat`).
- **Disk / RAM** for TensorFlow + first-time Hugging Face model download (default copy model: `Qwen/Qwen2.5-3B-Instruct`).
- **GPU** optional: TensorFlow and PyTorch can use CPU; expect slower runs on laptops.

---

## Installation

From the `prototype/` directory:

```bash
pip install -r requirements.txt
```

Pull the Ollama model you will use (defaults below):

```bash
ollama pull llama3.2:1b
```

---

## Environment variables (optional)

| Variable | Purpose | Default |
|----------|---------|---------|
| `OLLAMA_ENDPOINT` | Ollama HTTP chat URL | `http://localhost:11434/api/chat` |
| `OLLAMA_MODEL` | Model name for audience inference | `llama3.2:1b` |
| `HF_COPY_MODEL` | Hugging Face model id for copy generation | `Qwen/Qwen2.5-3B-Instruct` |
| `HF_COPY_MAX_NEW_TOKENS` | Max new tokens for copy | `220` |
| `HF_COPY_TEMPERATURE` | Sampling temperature | `0.7` |
| `HF_COPY_TOP_P` | Top-*p* sampling | `0.9` |
| `CM3070_WEB_HOST` | Web UI bind address | `127.0.0.1` |
| `CM3070_WEB_PORT` | Web UI port | `8000` |

---

## Run the full pipeline (CLI)

**Single image or directory** of images; **comma-separated platforms**:

Allowed platforms: `xiaohongshu`, `douyin`, `taobao`, `youtube`, `instagram`, `tiktok`, `other`.

```bash
cd prototype
python run_phase4_cli.py --image-dir /path/to/image.jpg --platforms xiaohongshu,youtube,instagram
```

Optional: write aggregated JSON to a file:

```bash
python run_phase4_cli.py --image-dir /path/to/images --platforms taobao --top-k 3 --output-json /tmp/run.json
```

Equivalent: run **`phase4_runner.py`** directly and print JSON to stdout:

```bash
python phase4_runner.py --image-dir /path/to/image.jpg --platforms xiaohongshu,douyin --top-k 3
```

Logs are written under **`prototype/debug_logs/`** (`phase_0_*`, `phase_1_*`, `phase_2_*`, `phase_3_*` per platform).

---

## Run the Web UI (recommended for demo)

**Stdlib server** (no FastAPI required for this entrypoint):

```bash
cd prototype
python simple_web_ui_phase5.py
```

Open **`http://127.0.0.1:8000`** (or your `CM3070_WEB_HOST` / `CM3070_WEB_PORT`).

- Upload an image, select one or more platforms, submit.
- On success, the UI shows audience, selling points, and tabbed copy; **`debug_logs/*_phase_5_web_ui_success.json`** records the run.

---

## Alternative Web UI (FastAPI)

If you prefer FastAPI + Uvicorn (dependencies in `requirements.txt`):

```bash
cd prototype
uvicorn web_ui_phase5:app --host 127.0.0.1 --port 8001
```

Same backend contract: Phase 4 orchestration via `phase4_runner`.

---

## Artifacts

| Path | Description |
|------|-------------|
| `debug_logs/` | Timestamped JSON for phases 0–3 (and web success/error logs) |
| `uploads/` | Images saved from the web UI when using `simple_web_ui_phase5.py` |

---

## Legacy / Phase-0-only CLI

For **vision-only** analysis (MobileNet + selling points, no LLM stages), the older entry point is still available:

```bash
python main.py path/to/image.jpg
python main.py path/to/folder --batch
```

See `main.py` help for batch and evaluation options.

---

## Project layout (high level)

```
prototype/
├── phase4_runner.py          # Core: run_phase4_for_one_image, run_phase4_for_image_dir
├── run_phase4_cli.py         # Thin CLI wrapper
├── image_analyzer.py         # Phase 0
├── feature_extractor.py      # MobileNet features
├── selling_point_converter.py
├── audience_analyzer.py      # Phase 1 (Ollama)
├── selling_point_matcher.py  # Phase 2
├── copywriter.py             # Phase 3 (HF)
├── simple_web_ui_phase5.py   # Phase 5 UI (stdlib)
├── web_ui_phase5.py          # Phase 5 UI (FastAPI, optional)
├── debug_logs/               # Created at runtime
├── uploads/                  # Web uploads
├── main.py                   # Legacy single-stage analyzer
├── requirements.txt
└── README.md
```

---

## Troubleshooting

1. **Phase 1 errors** — Ensure Ollama is running (`ollama serve`) and `OLLAMA_MODEL` is pulled.
2. **Phase 3 slow or OOM** — Use a smaller `HF_COPY_MODEL`, reduce `HF_COPY_MAX_NEW_TOKENS`, or run on GPU if available.
3. **TensorFlow / CUDA** — CPU execution is supported; first import can be slow.

---

## Course context

This repository supports **CM3070 Computer Science Final Project — Project Idea 4.1** (*Orchestrating AI models to achieve a goal*): three pre-trained model roles (vision, LLM understanding, text generation) plus explicit orchestration and evaluation artefacts.
