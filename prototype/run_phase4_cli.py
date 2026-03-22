"""
CLI for Phase 4:
image(s) + platforms -> phase0 -> phase1 -> phase2 -> phase3(per platform)

Main purpose:
  Provide a command-line entry for Stage 4 (reused by Phase 5 Web UI).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from phase4_runner import run_phase4_for_image_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 CLI: image -> multi-platform ad copies")
    parser.add_argument("--image-dir", type=str, required=True, help="Image file path or directory")
    parser.add_argument(
        "--platforms",
        type=str,
        default="xiaohongshu",
        help="Comma-separated platforms: xiaohongshu,douyin,taobao,youtube,instagram,tiktok,other",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    platforms: List[str] = [p.strip() for p in args.platforms.split(",") if p.strip()]
    results = run_phase4_for_image_dir(image_dir=args.image_dir, platforms=platforms, top_k=args.top_k)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(out_path))
        return

    # Keep CLI output short: print per image final phase3 logs
    for item in results:
        for c in item.get("copies", []):
            print(f"{c.get('platform')} {c.get('copy_log_path')}")


if __name__ == "__main__":
    main()

