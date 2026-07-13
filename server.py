"""FastAPI backend for the soccer-analyze web app.

Run:  uvicorn server:app --port 8000
Then open http://localhost:8000 (serves the built React frontend) or run the
frontend dev server (cd webapp && npm run dev) which proxies /api here.

Endpoints:
    POST /api/analyze            upload a clip -> {job_id}; analysis runs in a
                                 background thread
    GET  /api/jobs/{job_id}      job status + progress log
    GET  /api/plays              summaries of all analyzed plays
    GET  /api/plays/{name}       full play data (tracks, momentum, score...)
    GET  /api/plays/{name}/video the annotated clip (H.264 mp4)
"""

from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.pipeline import RESULTS, analyze_clip, list_plays

ROOT = Path(__file__).parent
CLIPS = ROOT / "clips"
WEBAPP_DIST = ROOT / "webapp" / "dist"

app = FastAPI(title="soccer-analyze")

JOBS: dict[str, dict] = {}   # job_id -> {status, log, name, error}
_ANALYZE_LOCK = threading.Lock()   # one CPU-heavy analysis at a time


def _safe_name(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "clip"
    name, k = stem, 2
    while (RESULTS / name / "report.json").exists():
        name = f"{stem}_{k}"
        k += 1
    return name


def _run_job(job_id: str, video_path: str, name: str, use_vlm: bool) -> None:
    job = JOBS[job_id]

    def log(msg: str) -> None:
        job["log"].append(msg)

    try:
        with _ANALYZE_LOCK:
            job["status"] = "running"
            report = analyze_clip(video_path, name, use_vlm, log=log)
        job["score"] = report["score"]
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


@app.post("/api/analyze")
async def analyze(file: UploadFile, vlm: bool = False) -> dict:
    if not file.filename or not file.filename.lower().endswith(
            (".mp4", ".mov", ".avi", ".mkv", ".webm")):
        raise HTTPException(400, "upload a video file (.mp4/.mov/.avi/.mkv/.webm)")
    name = _safe_name(file.filename)
    CLIPS.mkdir(exist_ok=True)
    dest = CLIPS / f"{name}{Path(file.filename).suffix.lower()}"
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "queued", "log": [], "name": name, "error": None}
    threading.Thread(target=_run_job, args=(job_id, str(dest), name, vlm),
                     daemon=True).start()
    return {"job_id": job_id, "name": name}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


@app.get("/api/plays")
def plays() -> list[dict]:
    return list_plays()


def _play_file(name: str, filename: str) -> Path:
    """Resolve a per-play file, rejecting path traversal."""
    p = (RESULTS / name / filename).resolve()
    if not p.is_file() or not p.is_relative_to(RESULTS.resolve()):
        raise HTTPException(404, "unknown play")
    return p


@app.get("/api/plays/{name}")
def play_data(name: str):
    return FileResponse(_play_file(name, "play_data.json"),
                        media_type="application/json")


@app.get("/api/plays/{name}/video")
def play_video(name: str):
    return FileResponse(_play_file(name, "annotated.mp4"), media_type="video/mp4")


# serve the built React frontend (npm run build) when present
if WEBAPP_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEBAPP_DIST, html=True), name="frontend")
