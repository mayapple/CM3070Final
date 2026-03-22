"""
Phase 2: Match selling points with audience profile (rule-based).

Input:
  - selling_points: List[str] (from midterm ImageAnalyzer)
  - audience_profile: Dict (from phase_1 audience_analyzer normalized_result)

Output:
  - recommended_selling_points: top-k selling points with scores/reasons

No LLM is used in this phase by design.
"""

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _clean_text(s: str) -> str:
    return (s or "").strip()


def _extract_tags_from_audience_profile(audience_profile: Dict[str, Any]) -> List[str]:
    tags = audience_profile.get("tags", [])
    if isinstance(tags, list):
        return [str(x).strip() for x in tags if str(x).strip()]
    if tags is None:
        return []
    return [str(tags).strip()]


def _tag_to_category(tag: str) -> Optional[str]:
    """
    Map audience tags to coarse categories for keyword matching.
    """
    t = tag.lower()
    if any(k in t for k in ["beauty", "appearance", "fashion", "style", "颜值"]):
        return "beauty"
    if any(k in t for k in ["premium", "luxury", "high_end", "质感", "高级"]):
        return "premium"
    if any(k in t for k in ["price_sensitive", "value", "性价比", "实惠", "划算"]):
        return "value"
    if any(k in t for k in ["fitness", "wellness", "health", "健康", "养生", "舒适"]):
        return "wellness"
    if any(k in t for k in ["tech", "technology", "innovation", "科技", "智能", "创新"]):
        return "tech"
    return None


_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    # These keywords are intended to match Chinese selling points
    # produced by prototype/selling_point_converter.py.
    "beauty": ["配色", "颜色", "颜值", "好看", "时尚", "靓丽", "好看", "流行"],
    "premium": ["高级", "质感", "精致", "工艺", "细腻", "奢华"],
    "value": ["性价比", "实惠", "划算", "省钱", "耐用"],
    "wellness": ["健康", "养生", "舒适", "透气", "轻盈", "清爽"],
    "tech": ["科技", "智能", "创新", "未来", "高科技"],
}

# Extend keywords to cover the project's current selling_point outputs.
# `prototype/selling_point_converter.py` currently returns English selling points.
_CATEGORY_KEYWORDS_EN: Dict[str, List[str]] = {
    "beauty": [
        "Color",
        "Eye-catching",
        "Fashionable",
        "Appearance",
        "Trendy",
        "Youthful",
        "Elegant",
        "Classic",
        "Mysterious",
        "Unique",
        "Distinctive",
        "Design",
        "Style",
        "Bright",
        "Energetic",
        "Sunny",
        "Fresh",
        "Warm",
        "Vintage",
        "Modern",
        "Minimalist",
        "Streamlined",
    ],
    "premium": [
        "Premium",
        "Fine",
        "Craftsmanship",
        "Quality",
        "Quality Assurance",
        "Refined",
        "Delicate",
        "Premium Texture",
        "Premium Color",
    ],
    "value": [
        "Value",
        "Budget",
        "Affordable",
        "Affordable",
        "Durable",
        "Reliable",
    ],
    "wellness": [
        "Comfortable",
        "Comfortable Color",
        "Comfortable Feel",
        "Comfortable Touch",
        "Soft",
        "Balanced",
        "Natural",
        "Balanced Color",
        "Soft Appearance",
    ],
    "tech": [
        "Tech",
        "Tech Feel",
        "Technology",
        "Smart",
        "Tech",
    ],
}


def _merge_keywords() -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {}
    for cat in _CATEGORY_KEYWORDS.keys():
        merged[cat] = list(_CATEGORY_KEYWORDS[cat])
        merged[cat].extend(_CATEGORY_KEYWORDS_EN.get(cat, []))
    return merged


_CATEGORY_KEYWORDS = _merge_keywords()


@dataclass
class ScoringResult:
    selling_point: str
    score: float
    matched_tags: List[str]
    matched_keywords: List[str]


def score_selling_point(
    selling_point: str, audience_tags: List[str], audience_segments: List[str]
) -> ScoringResult:
    sp = _clean_text(selling_point)
    matched_keywords: List[str] = []
    matched_tags: List[str] = []

    # Primary driver: tags.
    score = 0.0
    for tag in audience_tags:
        category = _tag_to_category(tag)
        if not category:
            continue
        keywords = _CATEGORY_KEYWORDS.get(category, [])
        hit_any = False
        for kw in keywords:
            if kw and kw in sp:
                matched_keywords.append(kw)
                hit_any = True
        if hit_any:
            matched_tags.append(tag)
            # Weight by number of keyword hits for that category.
            # Keep simple and explainable.
            score += 1.0 + len([k for k in keywords if k in sp]) * 0.2

    # Secondary driver: segments (very lightweight).
    seg_l = [s.lower() for s in (audience_segments or [])]
    if any(s in "young_digital_native" or "teen" in s for s in seg_l):
        # Youth tends to respond to beauty/style
        if any(k in sp for k in _CATEGORY_KEYWORDS["beauty"]):
            score += 0.5
    if any("working_professional" in s or "adult" in s for s in seg_l):
        if any(k in sp for k in _CATEGORY_KEYWORDS["premium"]):
            score += 0.3

    # Normalize to keep scores stable
    score = round(score, 3)
    return ScoringResult(
        selling_point=selling_point,
        score=score,
        matched_tags=sorted(set(matched_tags)),
        matched_keywords=sorted(set(matched_keywords)),
    )


def recommend_top_k(
    selling_points: List[str],
    audience_profile: Dict[str, Any],
    top_k: int = 3,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Returns:
      - scoring_details: list of dicts for logging/reporting
      - recommended_selling_points: list[str]
    """
    audience_tags = _extract_tags_from_audience_profile(audience_profile)
    audience_segments_raw = audience_profile.get("audience_segments", [])
    if isinstance(audience_segments_raw, list):
        audience_segments = [str(x).strip() for x in audience_segments_raw if str(x).strip()]
    else:
        audience_segments = [str(audience_segments_raw).strip()] if audience_segments_raw else []

    scoring: List[ScoringResult] = []
    for sp in selling_points:
        scoring.append(score_selling_point(sp, audience_tags, audience_segments))

    # Sort by score desc; tie-breaker by original order (stable sort)
    scored_sorted = sorted(
        enumerate(scoring),
        key=lambda x: x[1].score,
        reverse=True,
    )

    # Deduplicate by exact string
    recommended: List[str] = []
    for _, item in scored_sorted:
        sp = item.selling_point
        if sp not in recommended:
            recommended.append(sp)
        if len(recommended) >= top_k:
            break

    scoring_details: List[Dict[str, Any]] = []
    for item in scoring:
        scoring_details.append(
            {
                "selling_point": item.selling_point,
                "score": item.score,
                "matched_tags": item.matched_tags,
                "matched_keywords": item.matched_keywords,
            }
        )

    return scoring_details, recommended


def extract_from_phase1_log(phase1_log: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    """
    Try to extract:
      - selling_points: from phase1 log normalized_result.raw_params or input_params
      - audience_profile: normalized_result
    """
    normalized = phase1_log.get("normalized_result", {})
    raw_params = None
    if isinstance(normalized, dict):
        raw_params = normalized.get("raw_params")
    selling_points: List[str] = []

    # Common locations we store in our phase 1 logs:
    # 1) normalized_result.raw_params.selling_points
    if isinstance(raw_params, dict) and "selling_points" in raw_params:
        sps = raw_params["selling_points"]
        if isinstance(sps, list):
            selling_points = [str(x) for x in sps]

    # 2) input_params.selling_points
    if not selling_points and isinstance(phase1_log.get("input_params"), dict):
        inp = phase1_log["input_params"]
        if "selling_points" in inp and isinstance(inp["selling_points"], list):
            selling_points = [str(x) for x in inp["selling_points"]]

    return selling_points, normalized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: match selling points with audience profile (rule-based)."
    )
    parser.add_argument(
        "--phase1-log",
        type=str,
        default=None,
        help="Path to a phase_1 audience_analyzer success json log.",
    )
    parser.add_argument(
        "--selling-points-json",
        type=str,
        default=None,
        help="Path to a JSON file containing selling_points (list or {selling_points:[...]}).",
    )
    parser.add_argument(
        "--audience-profile-json",
        type=str,
        default=None,
        help="Path to a JSON file containing audience_profile (normalized_result).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of recommended selling points.",
    )
    args = parser.parse_args()

    selling_points: List[str] = []
    audience_profile: Dict[str, Any] = {}
    phase1_log_path: Optional[str] = None

    if args.phase1_log:
        phase1_log_path = args.phase1_log
        phase1_log = _load_json(args.phase1_log)
        selling_points, audience_profile = extract_from_phase1_log(phase1_log)

    if args.selling_points_json and not selling_points:
        payload = _load_json(args.selling_points_json)
        if isinstance(payload, list):
            selling_points = [str(x) for x in payload]
        elif isinstance(payload, dict) and "selling_points" in payload:
            selling_points = [str(x) for x in payload["selling_points"]]

    if args.audience_profile_json and not audience_profile:
        payload = _load_json(args.audience_profile_json)
        if isinstance(payload, dict):
            audience_profile = payload

    if not selling_points:
        raise ValueError("selling_points is empty. Provide --phase1-log or --selling-points-json.")
    if not audience_profile:
        raise ValueError("audience_profile is empty. Provide --phase1-log or --audience-profile-json.")

    scoring_details, recommended = recommend_top_k(
        selling_points=selling_points,
        audience_profile=audience_profile,
        top_k=args.top_k,
    )

    # Persist log -------------------------------------------------------
    debug_dir = Path(__file__).resolve().parent / "debug_logs"
    debug_dir.mkdir(parents=True, exist_ok=True)
    ts = _now_ts()
    out_path = debug_dir / f"{ts}_phase_2_selling_point_match_success.json"

    payload = {
        "timestamp": ts,
        "phase1_log_path": phase1_log_path,
        "top_k": args.top_k,
        "selling_points": selling_points,
        "audience_profile": audience_profile,
        "scoring_details": scoring_details,
        "recommended_selling_points": recommended,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Only print file location & name
    print(f"{out_path.parent} {out_path.name}")


if __name__ == "__main__":
    main()

