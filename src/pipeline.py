"""The full analysis pipeline as a callable — shared by the CLI (main.py)
and the web API (server.py).

analyze_clip() runs tracking -> events -> features -> score -> visuals and
writes everything under results/<name>/, including play_data.json (the dict
the React frontend consumes).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"


def analyze_clip(video_path: str | None, name: str, use_vlm: bool = False,
                 log: Callable[[str], None] = print) -> dict:
    """Analyze one clip (or the synthetic demo when video_path is None).

    Returns the report dict; writes report.json, play_data.json,
    dashboard.html, play_3d.html, annotated.mp4 (for real clips), and updates
    the cross-play gallery.
    """
    from .events import extract_events
    from .features import compute_features
    from .scoring import score_play, fuse_with_vlm

    out_dir = RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. track
    if video_path is None:
        from .tracking import synthetic_demo_tracks
        log("[1/5] generating synthetic demo play (no video)...")
        tracks = synthetic_demo_tracks()
    else:
        from .tracking import track_video
        log(f"[1/5] tracking players and ball in {video_path} (CPU, a few min)...")
        tracks = track_video(video_path, progress=False)
    log(f"      {tracks.n_frames} frames @ {tracks.fps:.0f}fps, "
        f"{len(tracks.players)} tracked players")

    # 2. events
    log("[2/5] extracting events...")
    events = extract_events(tracks)
    log(f"      {len(events.spells)} possession spells, {len(events.passes)} passes, "
        f"{len(events.dribbles)} dribbles, {len(events.shots)} shot attempts")

    # 3. features + score
    log("[3/5] computing difficulty features...")
    features = compute_features(tracks, events)
    result = score_play(features)
    report = result.as_dict()
    report["name"] = name

    # 4. optional VLM
    if use_vlm and video_path is not None:
        log("[4/5] running VLM analysis...")
        try:
            from .vlm import analyze_play
            analysis = analyze_play(video_path)
            report["vlm"] = analysis.model_dump()
            report["metric_score"] = report["score"]
            report["score"] = round(
                fuse_with_vlm(result.score, analysis.difficulty_estimate), 1)
            log(f"      VLM: {analysis.play_type} -> {analysis.difficulty_estimate}/100")
        except Exception as e:  # missing key, network, refusal — keep metric score
            log(f"      VLM step skipped: {e}")
    else:
        log("[4/5] VLM step skipped" + (" (needs a real video)" if use_vlm else ""))

    # 5. visuals + data for the web frontend
    log("[5/5] rendering visualizations and dashboard...")
    from .frontend import build_dashboard, build_play_data
    from .momentum import compute_momentum
    from .visualize import plot_play_3d, plot_difficulty_space, annotate_video

    plot_play_3d(tracks, str(out_dir / "play_3d.html"),
                 title=f"{name} — difficulty {report['score']:.0f}/100")

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
        log("      writing annotated video (tracking overlay)...")
        annotate_video(video_path, tracks, events, str(out_dir / "annotated.mp4"))
        video_rel = "annotated.mp4"

    data = build_play_data(tracks, events, momentum, report, video_rel)
    (out_dir / "play_data.json").write_text(json.dumps(data))
    build_dashboard(tracks, events, momentum, report,
                    str(out_dir / "dashboard.html"), video_rel)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    log("done")
    return report


def list_plays() -> list[dict]:
    """Summaries of every analyzed play (for the web API)."""
    plays = []
    if not RESULTS.exists():
        return plays
    for d in sorted(RESULTS.iterdir()):
        rep = d / "report.json"
        if rep.exists():
            r = json.loads(rep.read_text())
            plays.append({
                "name": r.get("name", d.name),
                "score": r.get("score"),
                "subscores": r.get("subscores", {}),
                "has_video": (d / "annotated.mp4").exists(),
            })
    return plays
