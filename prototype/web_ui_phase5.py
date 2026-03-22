"""
Phase 5: Web UI entry point.

Implements a minimal FastAPI web app:
  - Upload one product image
  - Select one or more platforms
  - Click run -> backend calls Phase 4 full pipeline
  - Render: audience profile + generated copies per platform

All algorithm logic is reused from `phase4_runner.py`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse
except ModuleNotFoundError as e:
    raise RuntimeError(
        "Missing web dependencies. Install with:\n"
        "  pip install -r /home/lujie/uolProjects/CM3070Final/V1/prototype/requirements.txt"
    ) from e


APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
DEBUG_DIR = APP_DIR / "debug_logs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_filename(name: str) -> str:
    # Keep only basename to avoid path traversal.
    return Path(name).name.replace(" ", "_")


def _page_html() -> str:
    return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>CM3070 Ad Copy Tool - Phase 5</title>
    <style>
      body {{ font-family: sans-serif; margin: 24px; }}
      .row {{ margin: 12px 0; }}
      .box {{ border: 1px solid #ddd; padding: 12px; border-radius: 8px; }}
      textarea {{ width: 100%; height: 160px; }}
      .copy-block {{ margin-top: 14px; }}
      button {{ padding: 10px 16px; }}
    </style>
  </head>
  <body>
    <h2>CM3070 Ad Copy Tool (Phase 5)</h2>
    <div class="box">
      <div class="row">
        <label>Upload product image:</label><br/>
        <input type="file" id="image" accept="image/*" />
      </div>

      <div class="row">
        <label>Select platforms:</label><br/>
        <label><input type="checkbox" name="platforms" value="xiaohongshu" checked /> xiaohongshu</label><br/>
        <label><input type="checkbox" name="platforms" value="douyin" /> douyin</label><br/>
        <label><input type="checkbox" name="platforms" value="taobao" /> taobao</label><br/>
        <label><input type="checkbox" name="platforms" value="other" /> other</label><br/>
      </div>

      <div class="row">
        <button onclick="runPipeline()">Run</button>
      </div>
    </div>

    <div class="box" style="margin-top:16px;">
      <h3>Result</h3>
      <div id="status">Waiting...</div>
      <pre id="audience" style="white-space:pre-wrap;"></pre>
      <div id="copies"></div>
    </div>

    <script>
      async function runPipeline() {{
        const fileInput = document.getElementById('image');
        if (!fileInput.files || fileInput.files.length === 0) {{
          alert('Please upload an image.');
          return;
        }}

        const platforms = [];
        document.querySelectorAll('input[name="platforms"]:checked').forEach(cb => platforms.push(cb.value));
        if (platforms.length === 0) {{
          alert('Please select at least one platform.');
          return;
        }}

        const fd = new FormData();
        fd.append('image', fileInput.files[0]);
        platforms.forEach(p => fd.append('platforms', p));

        document.getElementById('status').innerText = 'Running...';
        document.getElementById('audience').innerText = '';
        document.getElementById('copies').innerHTML = '';

        const resp = await fetch('/run', {{
          method: 'POST',
          body: fd
        }});

        if (!resp.ok) {{
          document.getElementById('status').innerText = 'Error: ' + resp.status;
          return;
        }}
        const data = await resp.json();

        document.getElementById('status').innerText = 'Done.';
        document.getElementById('audience').innerText = JSON.stringify(data.audience_profile, null, 2);

        const copiesDiv = document.getElementById('copies');
        copiesDiv.innerHTML = '';
        data.copies.forEach(item => {{
          const block = document.createElement('div');
          block.className = 'copy-block box';

          const title = document.createElement('div');
          title.innerText = 'Platform: ' + item.platform;
          block.appendChild(title);

          const ta = document.createElement('textarea');
          ta.value = item.copy_text || '';
          block.appendChild(ta);

          const meta = document.createElement('div');
          meta.style.fontSize = '12px';
          meta.innerText = 'log: ' + item.copy_log_path;
          block.appendChild(meta);

          copiesDiv.appendChild(block);
        }});
      }}
    </script>
  </body>
</html>
"""


app = FastAPI(title="CM3070 Ad Copy Tool (Phase 5)")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _page_html()


@app.post("/run")
def run(
    image: UploadFile = File(...),
    platforms: List[str] = Form(...),
) -> Any:
    from phase4_runner import run_phase4_for_one_image

    ts = _now_ts()
    upload_name = _safe_filename(image.filename or "upload.png")
    suffix = Path(upload_name).suffix or ".png"
    saved_path = UPLOAD_DIR / f"{ts}_{upload_name}"
    saved_path.write_bytes(image.file.read())

    try:
        result = run_phase4_for_one_image(
            image_path=str(saved_path),
            platforms=platforms,
            top_k=3,
            debug_dir=DEBUG_DIR,
        )

        # Phase 5 log (for evidence)
        out_log = DEBUG_DIR / f"{ts}_phase_5_web_ui_success.json"
        _payload = {
            "timestamp": ts,
            "input": {"uploaded_image_path": str(saved_path), "platforms": platforms},
            "output": {
                "phase0_log_path": result.get("phase0_log_path"),
                "phase1_log_path": result.get("phase1_log_path"),
                "phase2_log_path": result.get("phase2_log_path"),
                "copies": result.get("copies", []),
            },
        }
        out_log.write_text(json.dumps(_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return JSONResponse(
            {
                "audience_profile": result.get("audience_profile", {}),
                "copies": result.get("copies", []),
                "phase0_log_path": result.get("phase0_log_path"),
                "phase1_log_path": result.get("phase1_log_path"),
                "phase2_log_path": result.get("phase2_log_path"),
            }
        )
    except Exception as e:
        err_log = DEBUG_DIR / f"{ts}_phase_5_web_ui_error.json"
        err_payload = {
            "timestamp": ts,
            "input": {"uploaded_image_path": str(saved_path), "platforms": platforms},
            "error": {"type": type(e).__name__, "message": str(e)},
        }
        err_log.write_text(json.dumps(err_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return JSONResponse({"error": str(e), "log_path": str(err_log)}, status_code=500)

