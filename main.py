"""Soccer play difficulty analyzer — CLI.

Feed it a ~15 second clip; it tracks the players and ball, extracts the
events (passes, dribbles, shots), computes a 0-100 difficulty rating from
five measured dimensions, and renders the dashboard + 3D visualizations.

Usage:
    python main.py path/to/clip.mp4    # full pipeline (annotated video + dashboard)
    python main.py clip.mp4 --vlm      # + VLM semantic analysis
    python main.py --demo              # synthetic play, no video needed

For the web app (upload clips from the browser, interactive dashboards):
    uvicorn server:app --port 8000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import analyze_clip


def main() -> None:
    ap = argparse.ArgumentParser(description="Rate the difficulty of a soccer play (0-100)")
    ap.add_argument("video", nargs="?", help="path to a ~15s clip (mp4/mov/...)")
    ap.add_argument("--demo", action="store_true", help="run on a synthetic play, no video needed")
    ap.add_argument("--vlm", action="store_true", help="add VLM vision analysis (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--name", help="play name for reports (default: video filename)")
    args = ap.parse_args()

    if not args.demo and not args.video:
        ap.error("provide a video path or --demo")

    name = args.name or (Path(args.video).stem if args.video else "demo_play")
    report = analyze_clip(args.video if not args.demo else None, name, args.vlm)

    print("\n" + "=" * 46)
    print(f"  PLAY DIFFICULTY: {report['score']:.0f} / 100")
    print("=" * 46)
    for k, v in report["subscores"].items():
        bar = "#" * int(v / 4)
        print(f"  {k:<11} {v:5.1f}  {bar}")
    print(f"\n  DASHBOARD: results/{name}/dashboard.html  <- open this")
    print(f"  breakdown: results/{name}/report.json")
    print("  3D space:  results/difficulty_space.html")


if __name__ == "__main__":
    main()
