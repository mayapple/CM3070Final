"""
Final Phase 1: Infer audience from product selling points (LLM-based)

Input:
  selling_points (derived from product images by the midterm MobileNet prototype)
Output:
  structured audience profile/tags (age_group, audience_segments, tags, platform_preferences, notes)
"""

import json
import os
import re
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import requests

_ALLOWED_AGE_GROUPS = {
    "teen",
    "young_adult",
    "adult",
    "middle_aged",
    "senior",
}


class AudienceAnalyzerDebugLogError(Exception):
    """
    Raised when we saved a debug log file but the program still failed
    to parse/normalize the LLM output.
    """

    def __init__(self, log_path: Path, message: str = "AudienceAnalyzer debug error"):
        super().__init__(message)
        self.log_path = log_path


@dataclass
class AudienceParams:
    """
    Input parameters for audience inference from selling points.
    """

    selling_points: List[str]
    product_name: Optional[str] = None
    product_category: Optional[str] = None
    target_platform: Optional[str] = None  # optional hint for platform_preferences.platform
    notes: Optional[str] = None  # optional extra context


class AudienceAnalyzer:
    """
    Audience analyzer.

    Phase 1 requirement (LLM version):
    - Provide a callable interface that accepts audience parameters
      and returns a structured audience profile and tags.
    - Internally this implementation calls an LLM (Ollama HTTP API)
      with a JSON-only response format.
    """

    def __init__(self):
        # Model name can be overridden via environment variable.
        self._model_name: str = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
        # Ollama HTTP endpoint
        self._endpoint: str = os.environ.get(
            "OLLAMA_ENDPOINT", "http://localhost:11434/api/chat"
        )

    # Public API ---------------------------------------------------------
    def analyze(self, params: AudienceParams) -> Dict[str, Any]:
        """
        Analyze audience parameters and return a structured profile.

        Args:
            params: AudienceParams instance with basic audience info.

        Returns（由 LLM 生成并返回的 JSON）:
            Dict with at least keys:
              - \"age_group\": inferred age group label
              - \"audience_segments\": high-level segment labels
              - \"tags\": flat list of tags
              - \"platform_preferences\": platform-related hints
              - \"notes\": free-text summary (short)
            And we will always add:
              - \"raw_params\": original parameters as dict
        """
        raw = asdict(params)
        llm_result, _raw_content = self._call_llm(params)
        llm_result = self._normalize_result(llm_result)

        # 确保一定带上原始输入，便于后续阶段使用
        llm_result["raw_params"] = raw
        return llm_result

    def analyze_with_debug(self, params: AudienceParams) -> Dict[str, Any]:
        """
        Like `analyze`, but print:
        - which model is used
        - the exact prompt sent to the LLM
        - the LLM raw output content
        - the parsed JSON before normalization
        - the final normalized result
        """
        raw = asdict(params)
        prompt = self._build_prompt(params)
        print("\n=== LLM Request ===")
        print(f"Model: {self._model_name}")
        print(f"Endpoint: {self._endpoint}")
        print("\n--- Prompt ---")
        print(prompt)

        llm_result, raw_content = self._call_llm(params)
        print("\n--- LLM Raw Output (message.content) ---")
        print(raw_content)
        print("\n--- Parsed JSON (before normalization) ---")
        # llm_result is already json-decoded
        print(json.dumps(llm_result, ensure_ascii=False, indent=2))

        llm_result = self._normalize_result(llm_result)
        llm_result["raw_params"] = raw
        print("\n=== Normalized Result ===")
        print(json.dumps(llm_result, ensure_ascii=False, indent=2))
        return llm_result

    def analyze_to_file(
        self,
        params: AudienceParams,
        midterm_log_path: Optional[Path] = None,
    ) -> Path:
        """
        Run phase-1 LLM audience analysis and write all intermediate artifacts
        into a debug log file.

        Success path:
          - Save prompt + model + input params + LLM raw output
          - Save parsed JSON (before normalization)
          - Save normalized result (after normalization + raw_params)

        Failure path:
          - _call_llm will save an error log and raise AudienceAnalyzerDebugLogError.
        """
        raw = asdict(params)
        prompt = self._build_prompt(params)

        llm_result, raw_content = self._call_llm(params)
        parsed_before_norm = llm_result

        normalized = self._normalize_result(llm_result)
        normalized["raw_params"] = raw

        debug_dir = Path(__file__).resolve().parent / "debug_logs"
        debug_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = debug_dir / f"{ts}_phase_1_audience_analyzer_success.json"

        payload = {
            "timestamp": ts,
            "model": self._model_name,
            "endpoint": self._endpoint,
            "input_params": raw,
            "midterm_log_path": str(midterm_log_path) if midterm_log_path else None,
            "prompt": prompt,
            "llm_raw_content": raw_content,
            "parsed_json_before_normalization": parsed_before_norm,
            "normalized_result": normalized,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    # Internal helpers ---------------------------------------------------
    def _build_prompt(self, params: AudienceParams) -> str:
        """
        Build a prompt instructing the LLM to output ONLY JSON.
        """
        raw = asdict(params)
        if not raw.get("selling_points"):
            raise ValueError("selling_points must be a non-empty list")
        # 只让模型输出 JSON，便于后续解析
        prompt = f"""
You are an expert marketing analyst.
Given the following product selling points (in JSON), infer the most likely target audience
and return ONLY a JSON object EXACTLY matching this schema (types must match).

Rules:
- age_group must be EXACTLY ONE single value chosen from: teen, young_adult, adult, middle_aged, senior
- Do NOT output multiple candidates or use separators like '|'
- Do NOT output placeholder literals like 'segment_1'/'segment_2' or 'tag_1'/'tag_2'
- If you are unsure, still choose the closest ONE value
{{
  "age_group": "teen|young_adult|adult|middle_aged|senior",   // string, NOT array
  "audience_segments": ["price_sensitive", "beauty_oriented"],      // array of strings (snake_case)
  "tags": ["beauty", "fitness"],                                    // array of strings (snake_case or short tokens)
  "platform_preferences": {{                                          // object
    "platform": "xiaohongshu|douyin|taobao|other",
    "style": "storytelling_lifestyle|short_video_hook|product_feature_focused|generic",
    "length": "short|medium|long",
    "tone": "warm|energetic|informative|neutral"
  }},
  "notes": "..."                                                      // string
}}

Important:
- Respond with JSON ONLY, no extra explanation, no markdown.
- Use English for field names and tags, but you may mention Chinese platforms in values.
- If you are unsure, still output valid JSON with the required keys and correct types.

Product input:
{json.dumps(raw, ensure_ascii=False, indent=2)}

Now produce the JSON response.
"""
        return prompt.strip()

    def _call_llm(self, params: AudienceParams) -> Tuple[Dict[str, Any], str]:
        """
        Call the Ollama chat API and parse the JSON response.
        """
        prompt = self._build_prompt(params)

        payload = {
            "model": self._model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
        }

        response = requests.post(self._endpoint, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        # Ollama's chat API: content usually in data["message"]["content"]
        content = data.get("message", {}).get("content", "")
        if not content:
            raise ValueError("Empty response content from LLM")

        # Try to extract JSON more robustly:
        # 1) Prefer ```json ... ``` fenced blocks
        # 2) Balanced-brace extraction from the first '{' to the matching end
        #    (handles cases where the outermost '}' is missing by auto-appending closers)
        json_str: Optional[str] = None

        fenced = re.search(r"```json\s*(\{.*?\})\s*```", content, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            json_str = fenced.group(1).strip()
        else:
            start = content.find("{")
            if start != -1:
                # Balanced extraction with a tiny JSON-ish state machine.
                # This avoids rfind('}') truncating at an inner object when the outermost
                # JSON is cut off by the model.
                stack: List[str] = []
                in_str = False
                escape = False
                end_idx: Optional[int] = None

                for i in range(start, len(content)):
                    ch = content[i]
                    if in_str:
                        if escape:
                            escape = False
                        elif ch == "\\":
                            escape = True
                        elif ch == '"':
                            in_str = False
                        continue

                    if ch == '"':
                        in_str = True
                        continue

                    if ch in "{[":
                        stack.append(ch)
                    elif ch == "}":
                        if stack and stack[-1] == "{":
                            stack.pop()
                        if not stack:
                            end_idx = i
                            break
                    elif ch == "]":
                        if stack and stack[-1] == "[":
                            stack.pop()

                if end_idx is not None:
                    json_str = content[start : end_idx + 1].strip()
                else:
                    # Incomplete JSON: append required closing braces/brackets.
                    if stack:
                        closers = "".join(("}" if c == "{" else "]") for c in reversed(stack))
                        json_str = (content[start:].strip() + closers).strip()
                    else:
                        json_str = content[start:].strip()

        if not json_str:
            raise ValueError(
                "LLM response does not contain a JSON object. "
                f"Raw content (truncated): {content[:800]!r}"
            )

        # Best-effort sanitization for common formatting issues
        # (still expecting keys/strings to be in double quotes per schema)
        json_str_sanitized = json_str
        # remove trailing commas before } or ]
        json_str_sanitized = re.sub(r",(\s*[}\]])", r"\1", json_str_sanitized)
        # remove // and /* */ comments if the model ever emits them
        json_str_sanitized = re.sub(r"//.*?$", "", json_str_sanitized, flags=re.MULTILINE)
        json_str_sanitized = re.sub(r"/\*.*?\*/", "", json_str_sanitized, flags=re.DOTALL)

        try:
            parsed = json.loads(json_str_sanitized)
            return parsed, content
        except json.JSONDecodeError as e:
            preview = json_str_sanitized
            debug_dir = Path(__file__).resolve().parent / "debug_logs"
            debug_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = debug_dir / f"{ts}_phase_1_audience_analyzer_json_parse_error.json"

            error_payload = {
                "timestamp": ts,
                "model": self._model_name,
                "endpoint": self._endpoint,
                "prompt": prompt,
                "llm_raw_content": content,
                "extracted_json": json_str_sanitized,
                "json_decode_error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "pos": getattr(e, "pos", None),
                    "lineno": getattr(e, "lineno", None),
                    "colno": getattr(e, "colno", None),
                },
            }
            out_path.write_text(
                json.dumps(error_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            # Do not print large prompt/output to console.
            # Let caller print only the saved log path.
            raise AudienceAnalyzerDebugLogError(
                log_path=out_path,
                message=f"JSON parse failed, debug log saved to {out_path}",
            ) from e

    def _normalize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize/validate the LLM result so downstream code sees stable types.

        We do NOT implement a non-LLM fallback. If the result is unusable,
        raise an error so the caller can fix the prompt/model.
        """
        if not isinstance(result, dict):
            raise TypeError("LLM result must be a JSON object (dict)")

        age_group = result.get("age_group", None)
        if isinstance(age_group, list):
            age_group = age_group[0] if age_group else "unknown"
        if age_group is None:
            raise ValueError("LLM result missing required field: age_group")
        if not isinstance(age_group, str):
            age_group = str(age_group)
        age_group = age_group.strip()
        if age_group not in _ALLOWED_AGE_GROUPS:
            # If the model returns multiple candidates like "young_adult|adult|senior",
            # pick the first candidate we can find from the allowed set.
            candidates: List[str] = []
            for allowed in _ALLOWED_AGE_GROUPS:
                if allowed in age_group:
                    candidates.append(allowed)
            if candidates:
                age_group = candidates[0]
            else:
                raise ValueError(f"LLM returned invalid age_group: {age_group!r}")

        segments = result.get("audience_segments", [])
        if not isinstance(segments, list):
            segments = [segments]
        segments = [str(x).strip() for x in segments if str(x).strip()]

        tags = result.get("tags", [])
        if not isinstance(tags, list):
            tags = [tags]
        tags = [str(x).strip() for x in tags if str(x).strip()]

        prefs = result.get("platform_preferences", {})
        if not isinstance(prefs, dict):
            prefs = {}
        # Normalize common alternate keys that smaller models may emit
        key_map = {
            "content_style": "style",
            "content_length": "length",
        }
        for old_k, new_k in key_map.items():
            if old_k in prefs and new_k not in prefs:
                prefs[new_k] = prefs.get(old_k)
        # Coerce preference fields to strings
        for k in ("platform", "style", "length", "tone"):
            if k in prefs and prefs[k] is not None:
                prefs[k] = str(prefs[k]).strip()

        notes = result.get("notes", "")
        if notes is None:
            notes = ""
        if not isinstance(notes, str):
            notes = str(notes)
        notes = notes.strip()
        if not notes:
            # Fallback summary to avoid empty notes in downstream UI/logs.
            seg_text = ", ".join(segments[:3]) if segments else "general audience"
            tag_text = ", ".join(tags[:4]) if tags else "no specific tags"
            notes = (
                f"Likely {age_group} audience, mainly {seg_text}. "
                f"Key interests/signals: {tag_text}."
            )

        return {
            "age_group": age_group,
            "audience_segments": segments,
            "tags": tags,
            "platform_preferences": prefs,
            "notes": notes,
        }


def analyze_audience(params: AudienceParams) -> Dict[str, Any]:
    """
    Functional wrapper around AudienceAnalyzer.analyze for quick use.

    Example:
        from audience_analyzer import AudienceParams, analyze_audience

        params = AudienceParams(
            age_min=20,
            age_max=28,
            platform="xiaohongshu",
            interests=["beauty", "fitness"],
            budget_level="medium",
        )
        profile = analyze_audience(params)
    """
    analyzer = AudienceAnalyzer()
    return analyzer.analyze(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Final Phase 1: infer audience from selling_points (derived from product images)."
    )
    parser.add_argument("--image", type=str, default=None, help="Path to a product image (optional).")
    parser.add_argument(
        "--selling-points-json",
        type=str,
        default=None,
        help="Path to a JSON file: either a list of selling_points, or an object with key 'selling_points'.",
    )
    parser.add_argument("--platform", type=str, default=None, help="Optional platform hint (e.g. xiaohongshu/douyin/taobao).")
    parser.add_argument("--product-name", type=str, default=None, help="Optional product name.")
    parser.add_argument("--product-category", type=str, default=None, help="Optional product category.")
    args = parser.parse_args()

    # 1) Get selling_points ------------------------------------------------
    selling_points: List[str] = []
    midterm_log_path: Optional[Path] = None
    if args.image:
        from image_analyzer import ImageAnalyzer

        ia = ImageAnalyzer()
        result = ia.analyze(args.image)
        if "error" in result:
            midterm_debug_dir = Path(__file__).resolve().parent / "debug_logs"
            midterm_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            midterm_log_path = midterm_debug_dir / f"{midterm_ts}_phase_0_image_analyzer_error.json"
            midterm_debug_dir.mkdir(parents=True, exist_ok=True)
            midterm_payload = {
                "timestamp": midterm_ts,
                "input_image_path": args.image,
                "image_analyzer_error": result["error"],
            }
            midterm_log_path.write_text(
                json.dumps(midterm_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            raise RuntimeError(f"Image analysis failed: {result['error']}")
        selling_points = result.get("selling_points", [])
        midterm_debug_dir = Path(__file__).resolve().parent / "debug_logs"
        midterm_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        midterm_log_path = midterm_debug_dir / f"{midterm_ts}_phase_0_image_analyzer_success.json"
        midterm_payload = {
            "timestamp": midterm_ts,
            "input_image_path": args.image,
            "image_analyzer_result": result,
            "selling_points": selling_points,
        }
        midterm_debug_dir.mkdir(parents=True, exist_ok=True)
        midterm_log_path.write_text(
            json.dumps(midterm_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif args.selling_points_json:
        raw_payload = json.loads(Path(args.selling_points_json).read_text(encoding="utf-8"))
        if isinstance(raw_payload, list):
            selling_points = [str(x) for x in raw_payload]
        elif isinstance(raw_payload, dict) and "selling_points" in raw_payload:
            selling_points = [str(x) for x in raw_payload["selling_points"]]
        else:
            raise ValueError("selling-points-json must be a list or an object with key 'selling_points'.")
    else:
        # Demo selling points (replace by real selling_points from image analyzer later).
        selling_points = ["Vibrant Red", "Premium Texture", "Fine Craftsmanship"]

    # 2) LLM audience inference -------------------------------------------
    params = AudienceParams(
        selling_points=selling_points,
        product_name=args.product_name,
        product_category=args.product_category,
        target_platform=args.platform,
    )

    analyzer = AudienceAnalyzer()
    try:
        out_path = analyzer.analyze_to_file(params, midterm_log_path=midterm_log_path)
        print(f"{out_path.parent} {out_path.name}")
    except AudienceAnalyzerDebugLogError as e:
        out_path = e.log_path
        # Best-effort: link to midterm log path (metadata only).
        try:
            err_payload = json.loads(out_path.read_text(encoding="utf-8"))
            err_payload["midterm_log_path"] = str(midterm_log_path) if midterm_log_path else None
            out_path.write_text(
                json.dumps(err_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        print(f"{out_path.parent} {out_path.name}")

