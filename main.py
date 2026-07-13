"""Soccer play difficulty analyzer.

Feed it a ~15 second clip; it tracks the players and ball, extracts the
events (passes, dribbles, shots), computes a 0-100 difficulty rating from
five measured dimensions, and renders 3D visualizations.

Usage:
    python main.py path/to/clip.mp4    # full pipeline (annotated video + dashboard)
    python main.py clip.mp4 --vlm      # + VLM semantic analysis
    python main.py --demo              # synthetic play, no video needed

Outputs land in results/<clip-name>/ and every scored play is appended to
results/plays.csv, which feeds the cross-play difficulty_space.html chart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.events import extract_events
from src.features import compute_features
from src.scoring import score_play, fuse_with_vlm

RESULTS = Path(__file__).parent / "results"


def analyze(video_path: str | None, name: str, use_vlm: bool) -> dict:
    out_dir = RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. track
    if video_path is None:
        from src.tracking import synthetic_demo_tracks
        print("[1/5] generating synthetic demo play (no video)...")
        tracks = synthetic_demo_tracks()
    else:
        from src.tracking import track_video
        print(f"[1/5] tracking players and ball in {video_path} (CPU, ~1-3 min)...")
        tracks = track_video(video_path)
    print(f"      {tracks.n_frames} frames @ {tracks.fps:.0f}fps, "
          f"{len(tracks.players)} tracked players")

    # 2. events
    print("[2/5] extracting events...")
    events = extract_events(tracks)
    print(f"      {len(events.spells)} possession spells, {len(events.passes)} passes, "
          f"{len(events.dribbles)} dribbles, {len(events.shots)} shot attempts")

    # 3. features + score
    print("[3/5] computing difficulty features...")
    features = compute_features(tracks, events)
    result = score_play(features)
    report = result.as_dict()
    report["name"] = name

    # 4. optional VLM
    if use_vlm and video_path is not None:
        print("[4/5] asking Claude to watch the play...")
        try:
            from src.vlm import analyze_play
            analysis = analyze_play(video_path)
            report["vlm"] = analysis.model_dump()
            report["metric_score"] = report["score"]
            report["score"] = round(
                fuse_with_vlm(result.score, analysis.difficulty_estimate), 1)
            print(f"      VLM: {analysis.play_type} -> {analysis.difficulty_estimate}/100")
        except Exception as e:  # missing key, network, refusal — keep metric score
            print(f"      VLM step skipped: {e}")
    else:
        print("[4/5] VLM step skipped" + (" (needs a real video)" if use_vlm else ""))

    # 5. visualize + dashboard
    print("[5/5] rendering visualizations and dashboard...")
    from src.frontend import build_dashboard
    from src.momentum import compute_momentum
    from src.visualize import plot_play_3d, plot_difficulty_space, annotate_video

    plot_play_3d(tracks, str(out_dir / "play_3d.html"),
                 title=f"{name} — difficulty {report['score']:.0f}/100")

    # append to the cross-play gallery and re-render difficulty space
    gallery_csv = RESULTS / "plays.csv"
    row = {"name": name, "score": report["score"], **report["subscores"]}
    gallery = pd.read_csv(gallery_csv) if gallery_csv.exists() else pd.DataFrame()
    gallery = pd.concat([gallery[gallery["name"] != name] if not gallery.empty else gallery,
                         pd.DataFrame([row])], ignore_index=True)
    gallery.to_csv(gallery_csv, index=False)
    plot_difficulty_space(gallery, str(RESULTS / "difficulty_space.html"))

    momentum = compute_momentum(tracks, events)
    video_rel = None
    if video_path is not None:
        print("      writing annotated video (tracking overlay)...")
        annotate_video(video_path, tracks, events, str(out_dir / "annotated.mp4"))
        video_rel = "annotated.mp4"
    build_dashboard(tracks, events, momentum, report,
                    str(out_dir / "dashboard.html"), video_rel)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


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
    report = analyze(args.video if not args.demo else None, name, args.vlm)

    print("\n" + "=" * 46)
    print(f"  PLAY DIFFICULTY: {report['score']:.0f} / 100")
    print("=" * 46)
    for k, v in report["subscores"].items():
        bar = "#" * int(v / 4)
        print(f"  {k:<11} {v:5.1f}  {bar}")
    print(f"\n  DASHBOARD: results/{name}/dashboard.html  <- open this")
    print(f"  breakdown: results/{name}/report.json")
    print(f"  3D play:   results/{name}/play_3d.html")
    print("  3D space:  results/difficulty_space.html")


if __name__ == "__main__":
    main()
