"""
Phase 5 Web UI (zero external dependencies).

Why:
  FastAPI/Flask dependencies may not be available in some environments.
  This server uses only Python standard library:
    - http.server
    - cgi for multipart form parsing

UI:
  - GET / : HTML form (upload image + select platforms)
  - POST /run : runs Phase 4 pipeline and returns JSON

Core logic is reused from `phase4_runner.py`.
"""

from __future__ import annotations

import cgi
import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
DEBUG_DIR = APP_DIR / "debug_logs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_filename(name: str) -> str:
    return Path(name).name.replace(" ", "_")


def _page_html() -> str:
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>CM3070 Computer Science Final Project · Template: 4.1 Project Idea 1: Orchestrating AI models to achieve a goal · Jie Lu</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
      :root {
        --bg: #0b1020;
        --bg2: #111a33;
        --card: rgba(255, 255, 255, 0.08);
        --line: rgba(255, 255, 255, 0.16);
        --text: #ecf0ff;
        --sub: #b8c2e8;
        --brand: #7c9cff;
        --brand2: #45d0ff;
        --ok: #25d48a;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Manrope", sans-serif;
        color: var(--text);
        background:
          radial-gradient(1200px 600px at -10% -20%, #304781 0%, transparent 60%),
          radial-gradient(900px 500px at 110% -10%, #12406e 0%, transparent 55%),
          linear-gradient(160deg, var(--bg) 0%, var(--bg2) 100%);
        min-height: 100vh;
      }
      .wrap {
        max-width: 1080px;
        margin: 0 auto;
        padding: 28px 20px 40px;
      }
      .hero {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 16px;
        text-align: center;
      }
      .title {
        font-weight: 800;
        font-size: clamp(24px, 4vw, 38px);
        letter-spacing: 0.2px;
        margin: 0;
        text-align: center;
      }
      .sub {
        color: var(--sub);
        margin: 6px 0 0;
        font-size: 14px;
        text-align: center;
      }
      .meta {
        color: #d9e2ff;
        margin: 8px 0 0;
        font-size: 13px;
      }
      .tag {
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 6px 12px;
        color: #dbe4ff;
        font-size: 12px;
        backdrop-filter: blur(8px);
      }
      .tag-group {
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        justify-content: center;
        margin-top: 10px;
      }
      .grid {
        display: grid;
        grid-template-columns: 360px 1fr;
        gap: 16px;
      }
      .card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.24);
        backdrop-filter: blur(8px);
      }
      .card h3 {
        margin: 0 0 12px;
        font-size: 16px;
      }
      .row { margin: 12px 0; }
      .label {
        display: block;
        margin-bottom: 8px;
        font-weight: 600;
      }
      .file-input {
        width: 100%;
        border: 1px dashed var(--line);
        border-radius: 12px;
        padding: 12px;
        background: rgba(255,255,255,0.04);
        color: var(--sub);
      }
      .image-preview-wrap {
        margin-top: 10px;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 8px;
        background: rgba(0, 0, 0, 0.2);
      }
      .image-preview {
        display: none;
        width: 100%;
        max-height: 220px;
        object-fit: contain;
        border-radius: 8px;
      }
      .image-placeholder {
        color: var(--sub);
        font-size: 12px;
        text-align: center;
        padding: 20px 8px;
      }
      .platforms {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
      }
      .p-item {
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 8px 10px;
        font-size: 13px;
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(255,255,255,0.04);
      }
      .actions {
        display: flex;
        align-items: center;
        gap: 10px;
      }
      button {
        border: none;
        border-radius: 12px;
        padding: 10px 14px;
        color: #08122f;
        font-weight: 700;
        cursor: pointer;
        background: linear-gradient(90deg, var(--brand), var(--brand2));
      }
      button:disabled { opacity: 0.5; cursor: not-allowed; }
      .hint { color: var(--sub); font-size: 12px; }
      .status {
        font-size: 13px;
        padding: 8px 10px;
        border-radius: 10px;
        background: rgba(255,255,255,0.06);
        border: 1px solid var(--line);
        display: inline-flex;
        align-items: center;
        gap: 8px;
      }
      .dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--ok);
        box-shadow: 0 0 10px var(--ok);
      }
      pre {
        white-space: pre-wrap;
        margin: 0;
        font-family: "JetBrains Mono", monospace;
        font-size: 12px;
        color: #d7e2ff;
        max-height: 240px;
        overflow: auto;
      }
      .audience-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin-bottom: 10px;
      }
      .metric {
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 10px;
        background: rgba(255,255,255,0.04);
      }
      .metric .k {
        font-size: 11px;
        color: var(--sub);
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }
      .metric .v {
        margin-top: 4px;
        font-weight: 700;
        color: #f1f5ff;
        overflow-wrap: anywhere;
        word-break: break-word;
        line-height: 1.35;
      }
      .chips {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }
      .chip {
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 4px 8px;
        font-size: 12px;
        color: #dbe4ff;
        background: rgba(255,255,255,0.05);
      }
      .section-title {
        margin: 10px 0 6px;
        font-size: 12px;
        color: var(--sub);
        text-transform: uppercase;
        letter-spacing: 0.6px;
      }
      .note-box {
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 10px;
        background: rgba(255,255,255,0.04);
        color: #dbe4ff;
        font-size: 13px;
      }
      details {
        margin-top: 10px;
        border: 1px solid var(--line);
        border-radius: 10px;
        background: rgba(255,255,255,0.03);
        padding: 8px 10px;
      }
      summary {
        cursor: pointer;
        color: var(--sub);
        font-size: 12px;
      }
      .copies {
        margin-top: 12px;
      }
      .tabs {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 10px;
      }
      .tab-btn {
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 6px 12px;
        background: rgba(255,255,255,0.04);
        color: #dbe4ff;
        font-size: 12px;
        cursor: pointer;
      }
      .tab-btn.active {
        background: linear-gradient(90deg, rgba(124,156,255,0.25), rgba(69,208,255,0.25));
        border-color: rgba(124,156,255,0.8);
        color: #ffffff;
      }
      .copy-panel {
        display: none;
      }
      .copy-panel.active {
        display: block;
      }
      .copy-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
      }
      .pill {
        border: 1px solid var(--line);
        padding: 4px 8px;
        border-radius: 999px;
        font-size: 12px;
        color: #dbe4ff;
      }
      textarea {
        width: 100%;
        height: 170px;
        border-radius: 10px;
        border: 1px solid var(--line);
        background: rgba(0,0,0,0.2);
        color: var(--text);
        padding: 10px;
        resize: vertical;
        font-family: "Manrope", sans-serif;
      }
      .copy-meta {
        margin-top: 6px;
        color: var(--sub);
        font-size: 12px;
        word-break: break-all;
      }
      .btn-ghost {
        background: transparent;
        color: #dbe4ff;
        border: 1px solid var(--line);
        padding: 6px 10px;
        border-radius: 8px;
        font-size: 12px;
      }
      .loading {
        display: none;
        align-items: center;
        gap: 8px;
        color: var(--sub);
        font-size: 13px;
      }
      .spinner {
        width: 14px; height: 14px;
        border: 2px solid rgba(255,255,255,0.2);
        border-top-color: var(--brand2);
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
      }
      @keyframes spin { to { transform: rotate(360deg); } }
      @media (max-width: 900px) {
        .grid { grid-template-columns: 1fr; }
        .audience-grid { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="hero">
        <div>
          <h1 class="title">Ad Copy Studio</h1>
          <p class="sub">Image -> Audience -> Selling Points -> Multi-platform Copies</p>
          <div class="tag-group">
            <div class="tag">CM3070 Computer Science Final Project</div>
            <div class="tag">Template: 4.1 Project Idea 1: Orchestrating AI models to achieve a goal</div>
            <div class="tag">Jie Lu</div>
          </div>
        </div>
      </div>

      <div class="grid">
        <div class="card">
          <h3>Input</h3>
          <div class="row">
            <label class="label">Upload product image</label>
            <input class="file-input" type="file" id="image" accept="image/*" />
            <div class="image-preview-wrap">
              <img id="imagePreview" class="image-preview" alt="Selected image preview" />
              <div id="imagePlaceholder" class="image-placeholder">No image selected</div>
            </div>
          </div>

          <div class="row">
            <label class="label">Target platforms</label>
            <div class="platforms">
              <label class="p-item"><input type="checkbox" name="platforms" value="xiaohongshu" checked /> xiaohongshu</label>
              <label class="p-item"><input type="checkbox" name="platforms" value="douyin" /> douyin</label>
              <label class="p-item"><input type="checkbox" name="platforms" value="taobao" /> taobao</label>
              <label class="p-item"><input type="checkbox" name="platforms" value="youtube" /> youtube</label>
              <label class="p-item"><input type="checkbox" name="platforms" value="instagram" /> instagram</label>
              <label class="p-item"><input type="checkbox" name="platforms" value="tiktok" /> tiktok</label>
              <label class="p-item"><input type="checkbox" name="platforms" value="other" /> other</label>
            </div>
          </div>

          <div class="actions">
            <button id="runBtn" onclick="runPipeline()">Generate</button>
            <div class="loading" id="loading"><span class="spinner"></span> running pipeline...</div>
          </div>
          <p class="hint">Tip: first run may be slower due to model warm-up.</p>
        </div>

        <div class="card">
          <h3>Output</h3>
          <div class="status"><span class="dot"></span><span id="status">Waiting...</span></div>
          <div style="margin-top:12px;">
            <div class="label">Audience Profile</div>
            <div id="audience"></div>
          </div>
          <div style="margin-top:12px;">
            <div class="label">Selling Points</div>
            <div id="sellingPoints" class="chips"></div>
          </div>
          <div class="label" style="margin-top:14px;">Generated Copies</div>
          <div id="copies" class="copies"></div>
        </div>
      </div>
    </div>

<script>
      function copyText(text) {
        navigator.clipboard.writeText(text || '');
      }

      function setupImagePreview() {
        const fileInput = document.getElementById('image');
        const preview = document.getElementById('imagePreview');
        const placeholder = document.getElementById('imagePlaceholder');
        if (!fileInput || !preview || !placeholder) return;

        fileInput.addEventListener('change', () => {
          if (!fileInput.files || fileInput.files.length === 0) {
            preview.style.display = 'none';
            preview.src = '';
            placeholder.style.display = 'block';
            placeholder.innerText = 'No image selected';
            return;
          }

          const file = fileInput.files[0];
          if (!file.type.startsWith('image/')) {
            preview.style.display = 'none';
            preview.src = '';
            placeholder.style.display = 'block';
            placeholder.innerText = 'Selected file is not an image';
            return;
          }

          const reader = new FileReader();
          reader.onload = (e) => {
            preview.src = String(e.target?.result || '');
            preview.style.display = 'block';
            placeholder.style.display = 'none';
          };
          reader.readAsDataURL(file);
        });
      }

      function toList(x) {
        if (!x) return [];
        return Array.isArray(x) ? x.filter(Boolean) : [String(x)];
      }

      function esc(s) {
        return String(s ?? '')
          .replaceAll('&', '&amp;')
          .replaceAll('<', '&lt;')
          .replaceAll('>', '&gt;');
      }

      function renderChips(items) {
        const list = toList(items);
        if (!list.length) return '<span class="chip">N/A</span>';
        return list.map(v => `<span class="chip">${esc(v)}</span>`).join('');
      }

      function renderAudienceProfile(profile) {
        const p = profile || {};
        const pref = p.platform_preferences || {};
        const notes = (p.notes || '').trim();
        const raw = p.raw_params || {};
        const pickPrimary = (v) => {
          const arr = String(v || '').split('|').map(x => x.trim()).filter(Boolean);
          return arr.length ? arr[0] : 'Not provided';
        };
        const pickDisplay = (v) => {
          const arr = String(v || '').split('|').map(x => x.trim()).filter(Boolean);
          if (!arr.length) return 'Not provided';
          return arr.join(' / ');
        };
        const tone = pickPrimary(pref.tone);
        const length = pickPrimary(pref.length);
        const style = pickDisplay(pref.style);
        const targetHint = raw.target_platform || 'Not provided';

        return `
          <div class="audience-grid">
            <div class="metric">
              <div class="k">Age Group</div>
              <div class="v">${esc(p.age_group || 'N/A')}</div>
            </div>
            <div class="metric">
              <div class="k">Target Platform Hint</div>
              <div class="v">${esc(targetHint)}</div>
            </div>
            <div class="metric">
              <div class="k">Style</div>
              <div class="v">${esc(style)}</div>
            </div>
            <div class="metric">
              <div class="k">Tone / Length</div>
              <div class="v">${esc(tone + ' / ' + length)}</div>
            </div>
          </div>

          <div class="section-title">Audience Segments</div>
          <div class="chips">${renderChips(p.audience_segments)}</div>

          <div class="section-title">Tags</div>
          <div class="chips">${renderChips(p.tags)}</div>

          <div class="section-title">Source Selling Points</div>
          <div class="chips">${renderChips(raw.selling_points || [])}</div>

          <div class="section-title">Notes</div>
          <div class="note-box">${notes ? esc(notes) : 'No additional notes.'}</div>

          <details>
            <summary>View raw JSON</summary>
            <pre>${esc(JSON.stringify(p, null, 2))}</pre>
          </details>
        `;
      }

      async function runPipeline() {
        const fileInput = document.getElementById('image');
        const runBtn = document.getElementById('runBtn');
        const loading = document.getElementById('loading');
        if (!fileInput.files || fileInput.files.length === 0) {
          alert('Please upload an image.');
          return;
        }

        const platforms = [];
        document.querySelectorAll('input[name="platforms"]:checked').forEach(cb => platforms.push(cb.value));
        if (platforms.length === 0) {
          alert('Please select at least one platform.');
          return;
        }

        runBtn.disabled = true;
        loading.style.display = 'inline-flex';
        const fd = new FormData();
        fd.append('image', fileInput.files[0]);
        platforms.forEach(p => fd.append('platforms', p));

        document.getElementById('status').innerText = 'Running...';
        document.getElementById('audience').innerHTML = '';
        document.getElementById('sellingPoints').innerHTML = '';
        document.getElementById('copies').innerHTML = '';

        const resp = await fetch('/run', { method: 'POST', body: fd });
        const data = await resp.json();

        if (!resp.ok) {
          document.getElementById('status').innerText = 'Error: ' + (data.error || resp.status);
          runBtn.disabled = false;
          loading.style.display = 'none';
          return;
        }

        document.getElementById('status').innerText = 'Done.';
        document.getElementById('audience').innerHTML = renderAudienceProfile(data.audience_profile);
        document.getElementById('sellingPoints').innerHTML = renderChips(data.recommended_selling_points || []);

        const copiesDiv = document.getElementById('copies');
        copiesDiv.innerHTML = '';
        const copies = Array.isArray(data.copies) ? data.copies : [];
        if (!copies.length) {
          copiesDiv.innerHTML = '<div class="note-box">No copies generated.</div>';
        } else {
          const tabs = document.createElement('div');
          tabs.className = 'tabs';
          copiesDiv.appendChild(tabs);

          copies.forEach((item, idx) => {
            const tab = document.createElement('button');
            tab.className = 'tab-btn' + (idx === 0 ? ' active' : '');
            tab.innerText = item.platform || ('copy_' + idx);
            tab.onclick = () => {
              copiesDiv.querySelectorAll('.tab-btn').forEach(x => x.classList.remove('active'));
              copiesDiv.querySelectorAll('.copy-panel').forEach(x => x.classList.remove('active'));
              tab.classList.add('active');
              const panel = copiesDiv.querySelector(`#panel-${idx}`);
              if (panel) panel.classList.add('active');
            };
            tabs.appendChild(tab);
          });

          copies.forEach((item, idx) => {
            const block = document.createElement('div');
            block.className = 'copy-panel card' + (idx === 0 ? ' active' : '');
            block.id = `panel-${idx}`;

            const head = document.createElement('div');
            head.className = 'copy-head';
            const title = document.createElement('div');
            title.className = 'pill';
            title.innerText = item.platform;
            const btn = document.createElement('button');
            btn.className = 'btn-ghost';
            btn.innerText = 'Copy';
            btn.onclick = () => copyText(item.copy_text || '');
            head.appendChild(title);
            head.appendChild(btn);
            block.appendChild(head);

            const ta = document.createElement('textarea');
            ta.value = item.copy_text || '';
            block.appendChild(ta);

            const meta = document.createElement('div');
            meta.className = 'copy-meta';
            meta.innerText = 'log: ' + item.copy_log_path;
            block.appendChild(meta);

            copiesDiv.appendChild(block);
          });
        }
        runBtn.disabled = false;
        loading.style.display = 'none';
      }

      setupImagePreview();
</script>
  </body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "CM3070Phase5/0.1"

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, code: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/" or path == "":
            return self._send_html(200, _page_html())
        return self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/run":
            return self._send_json(404, {"error": "Not found"})

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST"},
        )

        image_item = form["image"] if "image" in form else None
        if image_item is None or not getattr(image_item, "filename", None):
            return self._send_json(400, {"error": "Missing image upload"})

        # cgi may return multiple values; normalize to list[str]
        platforms: List[str] = []
        if "platforms" in form:
            val = form["platforms"]
            if isinstance(val, list):
                platforms = [str(x.value) for x in val]
            else:
                platforms = [str(val.value)]
        if not platforms:
            platforms = ["xiaohongshu"]

        ts = _now_ts()
        upload_name = _safe_filename(image_item.filename)
        suffix = Path(upload_name).suffix or ".png"
        saved_path = UPLOAD_DIR / f"{ts}_{upload_name}"
        saved_path.write_bytes(image_item.file.read())

        try:
            from phase4_runner import run_phase4_for_one_image

            result = run_phase4_for_one_image(
                image_path=str(saved_path),
                platforms=platforms,
                top_k=3,
                debug_dir=DEBUG_DIR,
            )

            out_log = DEBUG_DIR / f"{ts}_phase_5_web_ui_success.json"
            out_log.write_text(
                json.dumps(
                    {
                        "timestamp": ts,
                        "input": {"uploaded_image_path": str(saved_path), "platforms": platforms},
                        "output": result,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            return self._send_json(
                200,
                {
                    "audience_profile": result.get("audience_profile", {}),
                    "recommended_selling_points": result.get("recommended_selling_points", []),
                    "copies": result.get("copies", []),
                    "phase0_log_path": result.get("phase0_log_path"),
                    "phase1_log_path": result.get("phase1_log_path"),
                    "phase2_log_path": result.get("phase2_log_path"),
                },
            )
        except Exception as e:
            err_log = DEBUG_DIR / f"{ts}_phase_5_web_ui_error.json"
            err_log.write_text(
                json.dumps(
                    {
                        "timestamp": ts,
                        "input": {"uploaded_image_path": str(saved_path), "platforms": platforms},
                        "error": {"type": type(e).__name__, "message": str(e)},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return self._send_json(500, {"error": str(e), "log_path": str(err_log)})


def main() -> None:
    port = int(os.environ.get("CM3070_WEB_PORT", "8000")) if "os" in globals() else 8000
    host = os.environ.get("CM3070_WEB_HOST", "127.0.0.1") if "os" in globals() else "127.0.0.1"
    print(f"Serving Phase 5 Web UI at http://{host}:{port}")
    httpd = HTTPServer((host, port), Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    import os

    main()

