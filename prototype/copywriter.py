"""
Phase 3: Generate ad copy for a platform (LLM-based).

Input:
  - recommended_selling_points (from phase 2)
  - audience_profile (from phase 1, normalized)
  - platform (derived from audience_profile.platform_preferences.platform)

Output:
  - platform + copy (structured JSON), written to a debug log file.
"""

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_ALLOWED_PLATFORMS = {
    "xiaohongshu",
    "douyin",
    "taobao",
    "youtube",
    "instagram",
    "tiktok",
    "other",
}


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class CopywriterParams:
    recommended_selling_points: List[str]
    audience_profile: Dict[str, Any]
    platform: str
    product_name: Optional[str] = None
    notes: Optional[str] = None


class CopywriterDebugLogError(Exception):
    def __init__(self, log_path: Path, message: str = "Copywriter debug error"):
        super().__init__(message)
        self.log_path = log_path


class Copywriter:
    """
    Phase 3 generator.

    Use Hugging Face `transformers` text-generation pipeline.
    Output is normalized into {"platform": "...", "copy": "..."}.
    """

    def __init__(self):
        self._model_name: str = os.environ.get("HF_COPY_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        self._max_new_tokens: int = int(os.environ.get("HF_COPY_MAX_NEW_TOKENS", "220"))
        self._temperature: float = float(os.environ.get("HF_COPY_TEMPERATURE", "0.7"))
        self._top_p: float = float(os.environ.get("HF_COPY_TOP_P", "0.9"))
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import pipeline
        except Exception as e:
            raise RuntimeError(
                "Hugging Face dependencies not installed. "
                "Please install `transformers` and `torch` in your environment."
            ) from e

        self._pipeline = pipeline(
            task="text-generation",
            model=self._model_name,
            device_map="auto",
        )

    def _postprocess_copy(self, text: str, platform: str) -> str:
        """
        收敛/清洗生成文本，尽量只保留“最终可展示的广告文案”。
        由于模型有时会在文案后追加解释/模板残留，我们用启发式规则截断。
        """
        if text is None:
            return ""

        t = str(text)
        t = t.replace("\r\n", "\n").strip()

        # Remove common prompt fragments
        t = re.sub(r"^\s*of\s+`[^`]+`\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"^\s*Here['’]s\b[^\n]*\n+", "", t, flags=re.IGNORECASE)
        t = re.sub(r"^\s*Here['’]s\s+the\s+ad\s+copy[^\n]*\n+", "", t, flags=re.IGNORECASE)

        # Cut off everything after explanation sections.
        # 1) Prefer separator lines like "---"
        m = re.search(r"(?m)^\s*---\s*$", t)
        if m:
            t = t[: m.start()].strip()

        # 2) Cut by common trailing sections
        cut_markers = [
            r"(?mi)\bNote:\b",
            r"(?mi)\bLet me know\b",
            r"(?mi)\bThis ad copy is tailored\b",
            r"(?mi)\bThe provided ad copy\b",
            r"(?mi)\bIf you need any adjustments\b",
            r"(?mi)\bIf further customization\b",
        ]
        cut_pos = None
        for pat in cut_markers:
            mm = re.search(pat, t)
            if mm:
                cut_pos = mm.start() if cut_pos is None else min(cut_pos, mm.start())
        if cut_pos is not None:
            t = t[:cut_pos].strip()

        # Collapse excessive blank lines
        t = re.sub(r"\n{3,}", "\n\n", t)

        # Remove lines that look like cross-platform headings, e.g. "Taobao Ad Copy"
        other_platforms = {
            "xiaohongshu": ["taobao", "douyin", "youtube", "instagram", "tiktok", "other"],
            "douyin": ["taobao", "xiaohongshu", "youtube", "instagram", "tiktok", "other"],
            "taobao": ["douyin", "xiaohongshu", "youtube", "instagram", "tiktok", "other"],
            "youtube": ["taobao", "douyin", "xiaohongshu", "instagram", "tiktok", "other"],
            "instagram": ["taobao", "douyin", "xiaohongshu", "youtube", "tiktok", "other"],
            "tiktok": ["taobao", "douyin", "xiaohongshu", "youtube", "instagram", "other"],
            "other": ["taobao", "douyin", "xiaohongshu", "youtube", "instagram", "tiktok"],
        }
        banned = other_platforms.get(platform, [])
        cleaned_lines: List[str] = []
        for line in t.split("\n"):
            lower = line.lower()
            if "ad copy" in lower and any(bp in lower for bp in banned):
                continue
            cleaned_lines.append(line)
        t = "\n".join(cleaned_lines).strip()

        # Final safety: if it still ends with an obvious truncated tail, drop after last full stop.
        t = t.strip()
        if re.search(r"(?i)\blet me\b\s*$", t):
            # truncate to last sentence end if present
            last_end = max(t.rfind("."), t.rfind("!"), t.rfind("?"))
            if last_end > 0:
                t = t[: last_end + 1].strip()

        return t

    def _build_prompt(self, params: CopywriterParams) -> str:
        platform = params.platform
        prefs = params.audience_profile.get("platform_preferences", {}) or {}
        age_group = params.audience_profile.get("age_group", "")
        segments = params.audience_profile.get("audience_segments", []) or []
        tags = params.audience_profile.get("tags", []) or []
        notes = params.audience_profile.get("notes", "")

        # Ask for plain ad text; we structure final output in code.
        prompt = f"""
You are an expert advertising copywriter.
Write ONE platform-specific ad copy using:
1) Recommended selling points: {json.dumps(params.recommended_selling_points, ensure_ascii=False)}
2) Audience profile:
   - age_group: {age_group}
   - audience_segments: {json.dumps(segments, ensure_ascii=False)}
   - tags: {json.dumps(tags, ensure_ascii=False)}
   - notes: {notes}
3) Platform preferences:
   - platform: {prefs.get('platform', platform)}
   - style: {prefs.get('style', 'generic')}
   - length: {prefs.get('length', 'short')}
   - tone: {prefs.get('tone', 'neutral')}

Platform rules (platform = "{platform}"):
- If platform is xiaohongshu: prefer storytelling + lifestyle hook, include 3-6 hashtags at the end.
- If platform is douyin: prefer short hook + benefit bullets, keep it punchy.
- If platform is taobao: emphasize value/benefits, clearer CTA, less poetic.
- If platform is youtube: use a creator-style hook + concise call-to-action; can include one line break structure.
- If platform is instagram: visually-driven tone, concise and trendy, include 3-8 hashtags.
- If platform is tiktok: viral short-hook style, energetic, punchy, CTA for interaction.
- If platform is other: generic but still persuasive.

Output requirements:
- Return plain ad copy text only
- Do NOT return JSON
- Do NOT include explanations about your process
- Do NOT mention other platforms (e.g. if platform is xiaohongshu, do not mention taobao/douyin)
- Keep length aligned with `length` preference
"""
        return prompt.strip()

    def _call_llm(self, params: CopywriterParams) -> Tuple[Dict[str, Any], str]:
        self._ensure_pipeline()
        prompt = self._build_prompt(params)
        outputs = self._pipeline(
            prompt,
            max_new_tokens=self._max_new_tokens,
            do_sample=True,
            temperature=self._temperature,
            top_p=self._top_p,
            return_full_text=False,
        )
        if not outputs or "generated_text" not in outputs[0]:
            raise ValueError("Empty generation result from Hugging Face pipeline")
        generated_text = str(outputs[0]["generated_text"]).strip()
        if not generated_text:
            raise ValueError("Generated ad copy is empty")

        raw_result = {"platform": params.platform, "copy": generated_text}
        return raw_result, generated_text

    def _normalize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(result, dict):
            raise TypeError("LLM result must be a JSON object (dict)")
        platform = result.get("platform", "")
        if platform is None:
            platform = ""
        platform = str(platform).strip()
        if platform not in _ALLOWED_PLATFORMS:
            # allow "taobao|other" kind of mistakes
            for p in _ALLOWED_PLATFORMS:
                if p in platform:
                    platform = p
                    break
        if platform not in _ALLOWED_PLATFORMS:
            raise ValueError(f"Invalid platform: {platform!r}")

        copy = result.get("copy", "")
        if copy is None:
            copy = ""
        if not isinstance(copy, str):
            copy = str(copy)
        copy = copy.strip()
        copy = self._postprocess_copy(copy, platform=platform)
        if not copy:
            raise ValueError("Empty copy in LLM result")

        return {"platform": platform, "copy": copy}

    def generate_to_file(
        self,
        params: CopywriterParams,
        debug_dir: Path,
        phase2_log_path: Optional[Path] = None,
    ) -> Path:
        debug_dir.mkdir(parents=True, exist_ok=True)
        ts = _now_ts()
        safe_platform = str(params.platform).replace("/", "_")
        out_path = debug_dir / f"{ts}_phase_3_{safe_platform}_ad_copy_success.json"

        prompt = self._build_prompt(params)
        try:
            raw_result, raw_content = self._call_llm(params)
            normalized = self._normalize_result(raw_result)
        except Exception as e:
            err_path = debug_dir / f"{ts}_phase_3_{safe_platform}_ad_copy_error.json"
            err_payload = {
                "timestamp": ts,
                "backend": "huggingface",
                "platform": params.platform,
                "phase2_log_path": str(phase2_log_path) if phase2_log_path else None,
                "model": self._model_name,
                "generation_config": {
                    "max_new_tokens": self._max_new_tokens,
                    "temperature": self._temperature,
                    "top_p": self._top_p,
                },
                "prompt": prompt,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                },
            }
            err_path.write_text(json.dumps(err_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            raise CopywriterDebugLogError(log_path=err_path) from e

        payload = {
            "timestamp": ts,
            "backend": "huggingface",
            "model": self._model_name,
            "generation_config": {
                "max_new_tokens": self._max_new_tokens,
                "temperature": self._temperature,
                "top_p": self._top_p,
            },
            "phase2_log_path": str(phase2_log_path) if phase2_log_path else None,
            "input": {
                "platform": params.platform,
                "recommended_selling_points": params.recommended_selling_points,
                "audience_profile": params.audience_profile,
            },
            "prompt": prompt,
            # Neutral names (backend-agnostic)
            "raw_model_result": raw_result,
            "normalized_result": normalized,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path


def extract_from_phase2_log(phase2_log: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any], str]:
    recommended = phase2_log.get("recommended_selling_points", []) or []
    if not isinstance(recommended, list):
        recommended = [str(recommended)]

    audience_profile = phase2_log.get("audience_profile", {}) or {}

    prefs = audience_profile.get("platform_preferences", {}) or {}
    raw_platform = str(prefs.get("platform", "xiaohongshu")).strip()
    tokens = [t.strip() for t in raw_platform.split("|") if t.strip()]
    platform = "xiaohongshu"
    for t in tokens:
        if t in _ALLOWED_PLATFORMS:
            platform = t
            break
    return recommended, audience_profile, platform

