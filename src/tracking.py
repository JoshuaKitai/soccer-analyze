"""Player and ball tracking from raw video.

The perception layer, tuned for real broadcast footage:

  players - YOLOv8 person detections at 960px through ByteTrack with long
            occlusion memory (custom yaml), then junk filtering (crowd,
            cutaways) and TRACK STITCHING: fragments that end/begin close in
            space+time with matching jersey color are merged back into one
            identity.
  ball    - a dedicated second pass at 1280px with a very low confidence
            threshold (small fast balls score low), then MOTION-GATED
            LINKING: a candidate is accepted only if it fits the ball's
            predicted trajectory, so one strong detection seeds a track and
            weak detections extend it while random false positives get
            rejected.
  camera  - global camera motion (pan/zoom) is estimated per frame from
            background optical flow, so speeds are measured relative to the
            pitch rather than the moving camera.
  teams   - k-means on jersey colors with outlier rejection (referees and
            goalkeepers whose color fits neither cluster get team -1).

All positions are normalized to [0, 1] in image space; a player's "ground
point" is the bottom-center of their bounding box. Per-player bounding-box
heights are kept so downstream logic (possession radius) can scale with how
large a player appears on screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_TRACKER_YAML = str(Path(__file__).parent / "bytetrack_soccer.yaml")

# detection settings
PLAYER_IMGSZ = 960
PLAYER_CONF = 0.30
BALL_IMGSZ = 1280
BALL_CONF = 0.05
BALL_MAX_BOX_H = 0.05      # candidates taller than 5% of frame aren't the ball

# track hygiene
MIN_TRACK_S = 0.5          # tracks shorter than this are noise
MIN_MEDIAN_H = 0.02        # tinier than this = crowd
MAX_MEDIAN_H = 0.55        # bigger than this = close-up cutaway
STITCH_MAX_GAP_S = 1.5
STITCH_BASE_DIST = 0.035   # merge threshold at zero gap...
STITCH_DIST_PER_S = 0.10   # ...growing with gap length
STITCH_COLOR_DIST = 60.0   # max BGR distance for "same jersey"

# ball linking
BALL_GATE = 0.05           # base acceptance radius around prediction
BALL_GATE_GROWTH = 0.012   # per missing frame
BALL_REACQUIRE_CONF = 0.30 # a strong detection can restart a lost track
BALL_REACQUIRE_GAP = 8     # after this many blind frames


@dataclass
class TrackData:
    """Normalized tracking output for one clip."""

    fps: float
    n_frames: int
    ball: np.ndarray                      # (n, 2), NaN where unseen
    players: dict[int, np.ndarray]        # tid -> (n, 2) ground positions
    teams: dict[int, int]                 # tid -> 0/1, -1 unknown/referee
    frame_size: tuple[int, int]
    heights: dict[int, np.ndarray] | None = None    # tid -> (n,) bbox heights
    cam_affines: np.ndarray | None = None           # (n, 2, 3) frame i-1 -> i
    colors: dict[int, np.ndarray] | None = None     # tid -> mean jersey BGR

    def ball_speed(self) -> np.ndarray:
        return self._speed(self.ball)

    def player_speed(self, tid: int) -> np.ndarray:
        return self._speed(self.players[tid])

    def _speed(self, pos: np.ndarray) -> np.ndarray:
        """Per-frame speed in normalized units/second, camera-compensated."""
        n = len(pos)
        sp = np.full(n, np.nan)
        for i in range(1, n):
            p0, p1 = pos[i - 1], pos[i]
            if np.isnan(p0[0]) or np.isnan(p1[0]):
                continue
            if self.cam_affines is not None:
                A = self.cam_affines[i]
                # where a static point at p0 would appear this frame
                p0 = A[:, :2] @ p0 + A[:, 2]
            sp[i] = float(np.linalg.norm(p1 - p0)) * self.fps
        return sp

    def height_at(self, tid: int, frame: int, default: float = 0.10) -> float:
        if self.heights is None or tid not in self.heights:
            return default
        h = self.heights[tid][frame]
        if np.isnan(h):
            valid = self.heights[tid][~np.isnan(self.heights[tid])]
            return float(np.median(valid)) if len(valid) else default
        return float(h)


# ---------------------------------------------------------------- utilities

def _interpolate_gaps(pos: np.ndarray, max_gap: int = 15) -> np.ndarray:
    """Linearly interpolate NaN gaps up to max_gap frames."""
    out = pos.copy()
    n = len(out)
    if out.ndim == 1:
        out = out[:, None]
    valid = ~np.isnan(out[:, 0])
    if valid.sum() < 2:
        return pos
    idx = np.arange(n)
    gap_len = np.zeros(n)
    run = 0
    for i in range(n):
        run = run + 1 if not valid[i] else 0
        gap_len[i] = run
    for i in range(n - 2, -1, -1):
        if gap_len[i] > 0 and gap_len[i + 1] > 0:
            gap_len[i] = max(gap_len[i], gap_len[i + 1])
    fill = (~valid) & (gap_len <= max_gap)
    for dim in range(out.shape[1]):
        col = out[:, dim]
        col[fill] = np.interp(idx, idx[valid], col[valid])[fill]
    return out.reshape(pos.shape)


def _kmeans2(colors: np.ndarray, iters: int = 25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = colors[rng.choice(len(colors), 2, replace=False)].astype(float)
    labels = np.zeros(len(colors), dtype=int)
    for _ in range(iters):
        d = np.linalg.norm(colors[:, None, :] - centers[None, :, :], axis=2)
        labels = d.argmin(axis=1)
        for k in range(2):
            if (labels == k).any():
                centers[k] = colors[labels == k].mean(axis=0)
    return labels, centers


def _pixel_affine_to_norm(A: np.ndarray, w: int, h: int) -> np.ndarray:
    """Convert a 2x3 pixel-space affine to normalized-coordinate space."""
    An = A.copy().astype(float)
    An[0, 1] *= h / w
    An[1, 0] *= w / h
    An[0, 2] /= w
    An[1, 2] /= h
    return An


_IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _camera_motion(prev_gray, gray, boxes_px, w, h):
    """Estimate the camera's frame-to-frame affine from background flow."""
    import cv2

    mask = np.full(prev_gray.shape, 255, np.uint8)
    sx = prev_gray.shape[1] / w
    sy = prev_gray.shape[0] / h
    for (x1, y1, x2, y2) in boxes_px:
        mask[int(y1 * sy):int(y2 * sy) + 1, int(x1 * sx):int(x2 * sx) + 1] = 0
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=180, qualityLevel=0.01,
                                  minDistance=16, mask=mask)
    if pts is None or len(pts) < 12:
        return _IDENTITY
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
    good = st.ravel() == 1
    if good.sum() < 12:
        return _IDENTITY
    A, _ = cv2.estimateAffinePartial2D(pts[good], nxt[good],
                                       method=cv2.RANSAC, ransacReprojThreshold=3)
    if A is None:
        return _IDENTITY
    # scale from the downsampled flow image back to full pixels, then normalize
    A_px = A.copy()
    A_px[0, 2] /= sx
    A_px[1, 2] /= sy
    return _pixel_affine_to_norm(A_px, w, h)


def _link_ball(candidates: list[list[tuple[float, float, float]]],
               n: int) -> np.ndarray:
    """Turn per-frame ball candidates into one trajectory via motion gating.

    Seed on a confident detection; each subsequent frame accepts the candidate
    nearest the constant-velocity prediction if it's inside a gate that grows
    while the ball is unseen. A lost track can be re-acquired by a strong
    detection. Random false positives away from the trajectory are ignored.
    """
    pos = np.full((n, 2), np.nan)
    vel = np.zeros(2)
    last: np.ndarray | None = None
    missing = 0

    for i in range(n):
        cands = candidates[i]
        chosen = None
        if last is not None:
            pred = last + vel * min(missing + 1, 6)
            gate = BALL_GATE + BALL_GATE_GROWTH * missing
            best_d = gate
            for (x, y, conf) in cands:
                d = float(np.hypot(x - pred[0], y - pred[1]))
                if d < best_d:
                    best_d, chosen = d, (x, y)
        if chosen is None and cands:
            # re-acquire: strongest detection, if we've been blind a while
            x, y, conf = max(cands, key=lambda c: c[2])
            if last is None or (missing >= BALL_REACQUIRE_GAP and conf >= BALL_REACQUIRE_CONF):
                chosen = (x, y)
                vel = np.zeros(2)
        if chosen is not None:
            p = np.array(chosen)
            if last is not None and missing == 0:
                vel = 0.6 * vel + 0.4 * (p - last)
            last, missing = p, 0
            pos[i] = p
        else:
            missing += 1
    return _interpolate_gaps(pos, max_gap=15)


def _stitch_tracks(players: dict[int, np.ndarray],
                   heights: dict[int, np.ndarray],
                   colors: dict[int, np.ndarray],
                   fps: float) -> tuple[dict, dict, dict]:
    """Merge track fragments belonging to the same player.

    Two fragments merge when one ends shortly before the other starts, the
    positions line up (tolerance grows with the gap), and jersey colors agree.
    """
    def span(arr):
        v = np.where(~np.isnan(arr[:, 0]))[0]
        return (int(v[0]), int(v[-1])) if len(v) else (None, None)

    max_gap = int(STITCH_MAX_GAP_S * fps)
    merged = True
    while merged:
        merged = False
        tids = sorted(players, key=lambda t: span(players[t])[0] or 0)
        for a in tids:
            sa, ea = span(players[a])
            if sa is None:
                continue
            best = None
            for b in tids:
                if b == a:
                    continue
                sb, eb = span(players[b])
                if sb is None or not (ea < sb <= ea + max_gap):
                    continue
                gap_s = (sb - ea) / fps
                d = float(np.linalg.norm(players[b][sb] - players[a][ea]))
                if d > STITCH_BASE_DIST + STITCH_DIST_PER_S * gap_s:
                    continue
                if a in colors and b in colors:
                    if float(np.linalg.norm(colors[a] - colors[b])) > STITCH_COLOR_DIST:
                        continue
                if best is None or d < best[1]:
                    best = (b, d)
            if best is not None:
                b = best[0]
                keep = ~np.isnan(players[b][:, 0])
                players[a][keep] = players[b][keep]
                hk = ~np.isnan(heights[b])
                heights[a][hk] = heights[b][hk]
                if a in colors and b in colors:
                    colors[a] = (colors[a] + colors[b]) / 2
                del players[b], heights[b]
                colors.pop(b, None)
                merged = True
                break
    return players, heights, colors


# ------------------------------------------------------------------ main API

def track_video(video_path: str, model_name: str = "yolov8n.pt",
                progress: bool = True) -> TrackData:
    """Run the full perception stack over a clip."""
    import cv2
    from ultralytics import YOLO

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    player_yolo = YOLO(model_name)
    ball_yolo = YOLO(model_name)   # separate instance: its own detect pass

    frames_players: list[dict[int, tuple[float, float, float]]] = []  # tid -> (x, y, height)
    ball_candidates: list[list[tuple[float, float, float]]] = []
    jersey_colors: dict[int, list[np.ndarray]] = {}
    affines: list[np.ndarray] = []
    prev_gray = None

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # --- players (tracked) ---
        rA = player_yolo.track(frame, persist=True, conf=PLAYER_CONF, classes=[0],
                               imgsz=PLAYER_IMGSZ, tracker=_TRACKER_YAML,
                               verbose=False)[0]
        players_here: dict[int, tuple[float, float, float]] = {}
        boxes_px = []
        bA = rA.boxes
        if bA is not None and len(bA) > 0 and bA.id is not None:
            xyxy = bA.xyxy.cpu().numpy()
            ids = bA.id.cpu().numpy().astype(int)
            for box, tid in zip(xyxy, ids):
                x1, y1, x2, y2 = box
                boxes_px.append(box)
                players_here[int(tid)] = ((x1 + x2) / 2 / w, y2 / h, (y2 - y1) / h)
                if i % 2 == 0:  # jersey sample every other frame
                    cy1 = int(y1 + 0.15 * (y2 - y1))
                    cy2 = int(y1 + 0.50 * (y2 - y1))
                    cx1 = int(x1 + 0.25 * (x2 - x1))
                    cx2 = int(x1 + 0.75 * (x2 - x1))
                    crop = frame[max(cy1, 0):cy2, max(cx1, 0):cx2]
                    if crop.size > 0:
                        jersey_colors.setdefault(int(tid), []).append(
                            crop.reshape(-1, 3).mean(axis=0))
        frames_players.append(players_here)

        # --- ball candidates (high-res, low-threshold detect) ---
        rB = ball_yolo.predict(frame, conf=BALL_CONF, classes=[32],
                               imgsz=BALL_IMGSZ, verbose=False)[0]
        cands = []
        bB = rB.boxes
        if bB is not None and len(bB) > 0:
            for box, conf in zip(bB.xyxy.cpu().numpy(), bB.conf.cpu().numpy()):
                x1, y1, x2, y2 = box
                if (y2 - y1) / h > BALL_MAX_BOX_H:
                    continue
                cands.append(((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, float(conf)))
        ball_candidates.append(cands)

        # --- camera motion (background optical flow) ---
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (w // 2, h // 2))
        if prev_gray is not None:
            affines.append(_camera_motion(prev_gray, gray, boxes_px, w, h))
        else:
            affines.append(_IDENTITY)
        prev_gray = gray

        i += 1
        if progress and i % 60 == 0:
            pct = f" ({100 * i // total}%)" if total > 0 else ""
            print(f"  tracked {i} frames{pct}...")

    cap.release()
    n = len(frames_players)

    # assemble ball trajectory
    ball = _link_ball(ball_candidates, n)

    # assemble player arrays + heights
    all_ids = sorted({tid for f in frames_players for tid in f})
    players: dict[int, np.ndarray] = {}
    heights: dict[int, np.ndarray] = {}
    for tid in all_ids:
        arr = np.full((n, 2), np.nan)
        hgt = np.full(n, np.nan)
        for f, d in enumerate(frames_players):
            if tid in d:
                arr[f] = d[tid][:2]
                hgt[f] = d[tid][2]
        vis = (~np.isnan(arr[:, 0])).sum()
        med_h = float(np.nanmedian(hgt)) if vis else 0.0
        if vis < fps * MIN_TRACK_S or not (MIN_MEDIAN_H <= med_h <= MAX_MEDIAN_H):
            continue
        players[tid] = arr
        heights[tid] = hgt

    colors = {t: np.mean(jersey_colors[t], axis=0)
              for t in players if t in jersey_colors}
    players, heights, colors = _stitch_tracks(players, heights, colors, fps)

    # smooth small holes left by occlusion
    for tid in players:
        players[tid] = _interpolate_gaps(players[tid], max_gap=10)
        heights[tid] = _interpolate_gaps(heights[tid], max_gap=10)

    # team assignment with referee/keeper rejection. The outlier bar is
    # deliberately high (1.15x the inter-kit distance): wrongly exiling a real
    # player destroys pass detection for every ball he touches, while letting
    # a referee slip onto a team merely adds one phantom defender.
    teams: dict[int, int] = {tid: -1 for tid in players}
    tids_c = [t for t in players if t in colors]
    if len(tids_c) >= 2:
        mat = np.array([colors[t] for t in tids_c])
        labels, centers = _kmeans2(mat)
        sep = float(np.linalg.norm(centers[0] - centers[1]))
        for t, lab in zip(tids_c, labels):
            if sep > 1e-6 and np.linalg.norm(colors[t] - centers[lab]) > 1.15 * sep:
                teams[t] = -1   # far outside both kits: referee or goalkeeper
            else:
                teams[t] = int(lab)

    return TrackData(fps=fps, n_frames=n, ball=ball, players=players,
                     teams=teams, frame_size=(w, h),
                     heights=heights, cam_affines=np.array(affines),
                     colors=colors)


def synthetic_demo_tracks(seed: int = 7) -> TrackData:
    """Generate a plausible synthetic attacking play (~12s at 30fps) so the
    events -> features -> scoring -> visualization pipeline can be exercised
    without a real video or the YOLO dependency."""
    rng = np.random.default_rng(seed)
    fps, dur = 30.0, 12.0
    n = int(fps * dur)
    t = np.linspace(0, 1, n)

    def noisy(path: np.ndarray, sigma: float = 0.004) -> np.ndarray:
        return path + rng.normal(0, sigma, path.shape)

    players: dict[int, np.ndarray] = {}
    teams: dict[int, int] = {}

    # Attacking team (0): three players building toward the right goal
    p1 = np.stack([0.15 + 0.55 * t, 0.70 - 0.25 * t], axis=1)          # winger run
    p2 = np.stack([0.25 + 0.50 * t, 0.45 + 0.05 * np.sin(6 * t)], axis=1)  # center mid
    p3 = np.stack([0.10 + 0.70 * t, 0.30 + 0.10 * t], axis=1)          # striker run
    # Defending team (1): four defenders tracking back / closing down
    d1 = np.stack([0.55 + 0.25 * t, 0.60 - 0.15 * t], axis=1)
    d2 = np.stack([0.60 + 0.22 * t, 0.40 + 0.05 * t], axis=1)
    d3 = np.stack([0.70 + 0.18 * t, 0.50 - 0.05 * np.sin(4 * t)], axis=1)
    d4 = np.stack([0.80 + 0.12 * t, 0.35 + 0.08 * t], axis=1)

    for i, (p, team) in enumerate(
            [(p1, 0), (p2, 0), (p3, 0), (d1, 1), (d2, 1), (d3, 1), (d4, 1)], start=1):
        players[i] = noisy(np.clip(p, 0, 1))
        teams[i] = team

    # Ball: carried by p2, passed to p1 (~4s), dribble, pass to p3 (~8s), shot (~11s)
    ball = np.zeros((n, 2))
    s1, s2, s3 = int(4 * fps), int(8 * fps), int(11 * fps)
    ball[:s1] = players[2][:s1]                       # p2 carries
    for i in range(s1, min(s1 + 12, n)):              # pass travels fast
        a = (i - s1) / 12
        ball[i] = (1 - a) * players[2][s1] + a * players[1][min(s1 + 12, n - 1)]
    ball[s1 + 12:s2] = players[1][s1 + 12:s2]         # p1 dribbles under pressure
    for i in range(s2, min(s2 + 10, n)):              # through ball
        a = (i - s2) / 10
        ball[i] = (1 - a) * players[1][s2] + a * players[3][min(s2 + 10, n - 1)]
    ball[s2 + 10:s3] = players[3][s2 + 10:s3]         # striker takes touch
    goal = np.array([0.98, 0.45])
    for i in range(s3, n):                            # shot toward goal
        a = min((i - s3) / (0.6 * fps), 1.0)
        ball[i] = (1 - a) * players[3][s3] + a * goal
    ball = noisy(ball, 0.003)

    heights = {tid: np.full(n, 0.12) for tid in players}
    affines = np.tile(_IDENTITY, (n, 1, 1))
    return TrackData(fps=fps, n_frames=n, ball=ball, players=players,
                     teams=teams, frame_size=(1280, 720),
                     heights=heights, cam_affines=affines)
