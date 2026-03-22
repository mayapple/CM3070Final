"""
Run pipeline: phase 0 (image->selling_points) -> phase 1 (selling_points->audience profile)
-> phase 2 (selling point matching)
-> phase 3 (platform ad copy generation).

User only needs to provide an image directory. All other parameters use defaults:
- platform hint: "xiaohongshu"
- phase 2 top_k: 3

This script writes phase_0 / phase_1 / phase_2 / phase_3 logs into `prototype/debug_logs/`.
"""

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils import get_image_files


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def phase_0_analyze_images(
    image_paths: List[str], image_analyzer, debug_dir: Path
) -> List[Tuple[str, Dict[str, Any], Path, List[str]]]:
    """
    Returns items per image:
      (image_path, image_analyzer_result, phase0_log_path, selling_points)
    """
    outputs: List[Tuple[str, Dict[str, Any], Path, List[str]]] = []

    for idx, image_path in enumerate(image_paths, 1):
        ts = _now_ts()
        result = image_analyzer.analyze(image_path)

        if "error" in result:
            out_path = debug_dir / f"{ts}_phase_0_image_analyzer_error.json"
            _write_json(
                out_path,
                {
                    "timestamp": ts,
                    "input_image_path": image_path,
                    "image_analyzer_error": result["error"],
                },
            )
            raise RuntimeError(f"Phase 0 failed for image #{idx}: {image_path}")

        selling_points = result.get("selling_points", [])

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
        outputs.append((image_path, result, out_path, selling_points))

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline runner: image-dir -> phase0 -> phase1 -> phase2 -> phase3"
    )
    parser.add_argument("--image-dir", type=str, required=True, help="Directory of product images")
    args = parser.parse_args()
    top_k = 3

    image_dir = args.image_dir
    debug_dir = Path(__file__).resolve().parent / "debug_logs"
    debug_dir.mkdir(parents=True, exist_ok=True)

    image_path_obj = Path(image_dir)
    if image_path_obj.is_file():
        image_paths = [str(image_path_obj)]
    else:
        image_paths = get_image_files(image_dir)

    if not image_paths:
        raise ValueError(f"No images found at: {image_dir}")

    # Initialize reusable components once
    from image_analyzer import ImageAnalyzer
    from audience_analyzer import AudienceAnalyzer, AudienceParams, AudienceAnalyzerDebugLogError
    from selling_point_matcher import recommend_top_k, extract_from_phase1_log
    from copywriter import Copywriter, CopywriterParams, CopywriterDebugLogError

    image_analyzer = ImageAnalyzer()
    audience_analyzer = AudienceAnalyzer()
    copywriter = Copywriter()

    # Phase 0: images -> selling_points
    phase0_outputs = phase_0_analyze_images(image_paths, image_analyzer, debug_dir)

    # Phase 1: selling_points -> audience profile (via LLM) and logs
    phase1_logs: List[Path] = []
    for image_path, _, phase0_log_path, selling_points in phase0_outputs:
        params = AudienceParams(
            selling_points=selling_points,
            target_platform="xiaohongshu",
        )
        try:
            out1_path = audience_analyzer.analyze_to_file(
                params, midterm_log_path=phase0_log_path
            )
            phase1_logs.append(out1_path)
            print(f"✓ Phase 1 complete {out1_path.parent} {out1_path.name}")
        except AudienceAnalyzerDebugLogError as e:
            # Keep the pipeline running for other images.
            print(f"✗ Phase 1 failed {e.log_path.parent} {e.log_path.name}")

    # Phase 2: phase1 logs -> recommend selling points and logs
    recommended_paths: List[Path] = []
    for phase1_log_path in phase1_logs:
        try:
            phase1_log = _load_json(phase1_log_path)
            selling_points, audience_profile = extract_from_phase1_log(phase1_log)

            scoring_details, recommended = recommend_top_k(
                selling_points=selling_points,
                audience_profile=audience_profile,
                top_k=top_k,
            )

            ts = _now_ts()
            out2_path = debug_dir / f"{ts}_phase_2_selling_point_match_success.json"
            payload = {
                "timestamp": ts,
                "phase1_log_path": str(phase1_log_path.relative_to(Path.cwd())),
                "top_k": top_k,
                "selling_points": selling_points,
                "audience_profile": audience_profile,
                "scoring_details": scoring_details,
                "recommended_selling_points": recommended,
            }
            _write_json(out2_path, payload)
            recommended_paths.append(out2_path)
            print(f"✓ Phase 2 complete {out2_path.parent} {out2_path.name}")

            # Phase 3: recommended selling points + audience -> platform ad copy
            prefs = (audience_profile or {}).get("platform_preferences", {}) or {}
            raw_platform = str(prefs.get("platform", "xiaohongshu")).strip() or "xiaohongshu"
            tokens = [t.strip() for t in raw_platform.split("|") if t.strip()]
            allowed = {"xiaohongshu", "douyin", "taobao", "other"}
            platform = "xiaohongshu"
            for t in tokens:
                if t in allowed:
                    platform = t
                    break
            platform_params = CopywriterParams(
                recommended_selling_points=recommended,
                audience_profile=audience_profile,
                platform=platform,
            )
            try:
                out3_path = copywriter.generate_to_file(
                    platform_params, debug_dir=debug_dir, phase2_log_path=out2_path
                )
                print(f"✓ Phase 3 complete {out3_path.parent} {out3_path.name}")
            except CopywriterDebugLogError as e:
                print(f"✗ Phase 3 failed {e.log_path.parent} {e.log_path.name}")
        except Exception:
            # Phase 2 errors are not expected for a valid phase1 log,
            # but we avoid stopping the whole run.
            print(f"✗ Phase 2 failed {phase1_log_path.parent} {phase1_log_path.name}")

    # Phase 2 completion already printed per image.


if __name__ == "__main__":
    main()

