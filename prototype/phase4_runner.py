"""
Phase 4: End-to-end orchestration (CLI first).

Goal:
  image (+ multiple platforms) -> phase0 -> phase1 -> phase2 -> phase3 (per platform)

This module is intended to be reused by Phase 5 Web UI.
It writes debug logs for phases 0/1/2/3 into `prototype/debug_logs/`.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import get_image_files


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_platform_list(platforms: List[str]) -> List[str]:
    allowed = {"xiaohongshu", "douyin", "taobao", "youtube", "instagram", "tiktok", "other"}
    out: List[str] = []
    for p in platforms:
        pp = str(p).strip()
        if not pp:
            continue
        # Allow accidental pipe-separated values.
        for token in [t.strip() for t in pp.split("|") if t.strip()]:
            if token in allowed:
                if token not in out:
                    out.append(token)
                break
        else:
            if "other" not in out:
                out.append("other")
    return out or ["xiaohongshu"]


def _phase0_analyze_one_image(
    image_path: str,
    image_analyzer,
    debug_dir: Path,
) -> Tuple[Path, List[str]]:
    ts = _now_ts()
    result = image_analyzer.analyze(image_path)

    if "error" in result:
        out_path = debug_dir / f"{ts}_phase_0_image_analyzer_error.json"
        _write_json(
            out_path,
            {"timestamp": ts, "input_image_path": image_path, "image_analyzer_error": result["error"]},
        )
        raise RuntimeError(f"Phase 0 failed for image: {image_path}")

    selling_points = result.get("selling_points", []) or []
    out_path = debug_dir / f"{ts}_phase_0_image_analyzer_success.json"
    _write_json(
        out_path,
        {
            "timestamp": ts,
            "input_image_path": image_path,
            "image_analyzer_result": result,
            "selling_points": selling_points,
        },
    )
    return out_path, selling_points


def run_phase4_for_one_image(
    image_path: str,
    platforms: List[str],
    top_k: int = 3,
    debug_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Returns:
      {
        phase0_log_path, phase1_log_path, phase2_log_path,
        audience_profile, recommended_selling_points,
        copies: [{platform, copy_log_path, copy_text}]
      }
    """

    platforms = _normalize_platform_list(platforms)
    debug_dir = debug_dir or (Path(__file__).resolve().parent / "debug_logs")
    debug_dir.mkdir(parents=True, exist_ok=True)

    from image_analyzer import ImageAnalyzer
    from audience_analyzer import AudienceAnalyzer, AudienceParams
    from selling_point_matcher import recommend_top_k, extract_from_phase1_log
    from copywriter import Copywriter, CopywriterParams

    image_analyzer = ImageAnalyzer()
    audience_analyzer = AudienceAnalyzer()
    copywriter = Copywriter()

    # Phase 0 -------------------------------------------------------------
    phase0_log_path, selling_points = _phase0_analyze_one_image(
        image_path=image_path,
        image_analyzer=image_analyzer,
        debug_dir=debug_dir,
    )

    # Phase 1 -------------------------------------------------------------
    # Provide one platform hint to the audience analyzer (helps LLM style hints),
    # but final copy is generated for each requested platform below.
    platform_hint = platforms[0]
    params = AudienceParams(selling_points=selling_points, target_platform=platform_hint)
    phase1_log_path = audience_analyzer.analyze_to_file(params, midterm_log_path=phase0_log_path)
    phase1_log = _load_json(phase1_log_path)
    _, audience_profile = extract_from_phase1_log(phase1_log)

    # Phase 2 -------------------------------------------------------------
    scoring_details, recommended = recommend_top_k(
        selling_points=selling_points, audience_profile=audience_profile, top_k=top_k
    )
    ts = _now_ts()
    phase2_log_path = debug_dir / f"{ts}_phase_2_selling_point_match_success.json"
    _write_json(
        phase2_log_path,
        {
            "timestamp": ts,
            "phase1_log_path": str(phase1_log_path),
            "top_k": top_k,
            "selling_points": selling_points,
            "audience_profile": audience_profile,
            "scoring_details": scoring_details,
            "recommended_selling_points": recommended,
        },
    )

    # Phase 3 (per platform) ---------------------------------------------
    copies: List[Dict[str, Any]] = []
    for platform in platforms:
        params3 = CopywriterParams(
            recommended_selling_points=recommended,
            audience_profile=audience_profile,
            platform=platform,
        )
        copy_log_path = copywriter.generate_to_file(
            params3, debug_dir=debug_dir, phase2_log_path=phase2_log_path
        )
        copy_log = _load_json(copy_log_path)
        normalized = copy_log.get("normalized_result", {}) or {}
        copy_text = normalized.get("copy", "") if isinstance(normalized, dict) else ""
        copies.append(
            {
                "platform": platform,
                "copy_log_path": str(copy_log_path),
                "copy_text": copy_text,
            }
        )

    return {
        "phase0_log_path": str(phase0_log_path),
        "phase1_log_path": str(phase1_log_path),
        "phase2_log_path": str(phase2_log_path),
        "audience_profile": audience_profile,
        "recommended_selling_points": recommended,
        "copies": copies,
    }


def run_phase4_for_image_dir(
    image_dir: str,
    platforms: List[str],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    image_path_obj = Path(image_dir)
    if image_path_obj.is_file():
        image_paths = [str(image_path_obj)]
    else:
        image_paths = get_image_files(image_dir)

    results: List[Dict[str, Any]] = []
    for p in image_paths:
        results.append(run_phase4_for_one_image(image_path=p, platforms=platforms, top_k=top_k))
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4 runner: image + platforms -> copies.")
    parser.add_argument("--image-dir", type=str, required=True, help="Image file path or directory")
    parser.add_argument(
        "--platforms",
        type=str,
        default="xiaohongshu",
        help="Comma-separated list: xiaohongshu,douyin,taobao,youtube,instagram,tiktok,other",
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    platforms = [p.strip() for p in args.platforms.split(",") if p.strip()]
    out = run_phase4_for_image_dir(args.image_dir, platforms=platforms, top_k=args.top_k)
    print(json.dumps(out, ensure_ascii=False, indent=2))

