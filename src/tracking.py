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

# ball linking (global path optimization over candidates)
BALL_MAX_LINK_GAP = 25     # frames a trajectory may skip between detections
BALL_STEP_ALLOW = 0.045    # allowed travel per frame of gap before penalty
BALL_STEP_BASE = 0.035     # slack at zero gap
BALL_TELEPORT_COST = 9.0   # cost cap: re-acquiring far away is possible but dear
BALL_GAP_COST = 0.15       # per skipped frame
BALL_MAX_SPEED = 2.5       # units/s — faster than any real ball flight; jumps
                           # implying more are physically impossible and FORBIDDEN
                           # unless the ball has been unseen long enough
BALL_TELEPORT_MIN_GAP = 10 # frames of blindness before a re-acquisition jump
                           # may ignore the physics ceiling
STATIC_SPAN_S = 2.5        # a "ball" parked at the same pitch position this long
STATIC_TOL = 0.035         # (within this radius, camera-compensated) is a pitch
                           # marking — penalty spot, center spot — not the ball
STATIC_MIN_HITS = 12       # ...if it was detected at least this many times
BALL_CONF_W = 3.0          # weight of detection confidence
BALL_PROX_NEAR = 0.15      # full proximity bonus inside this player distance
BALL_PROX_FAR = 0.30       # beyond this from every player: penalized (penalty
                           # spots and sideline clutter live far from players)


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
    boxes: dict[int, np.ndarray] | None = None      # tid -> (n, 4) xyxy normalized
    ball_candidates: list | None = None             # per-frame raw ball detections
                                                    # (kept so linking can be re-run
                                                    # from cache without re-detecting)

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


def _kmeans(colors: np.ndarray, k: int = 2, iters: int = 25,
            seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = colors[rng.choice(len(colors), k, replace=False)].astype(float)
    labels = np.zeros(len(colors), dtype=int)
    for _ in range(iters):
        d = np.linalg.norm(colors[:, None, :] - centers[None, :, :], axis=2)
        labels = d.argmin(axis=1)
        for j in range(k):
            if (labels == j).any():
                centers[j] = colors[labels == j].mean(axis=0)
    return labels, centers


def assign_teams(players: dict[int, np.ndarray],
                 colors: dict[int, np.ndarray]) -> dict[int, int]:
    """Team assignment from kit colors, with officials isolated.

    First tries THREE clusters: referees wear their own kit, so when a small
    third color group exists it's marked as officials (-1) and the two big
    groups become the teams. Falls back to two clusters with distance-based
    outlier rejection. Frame-border dwell (sideline officials) is applied by
    the caller on top of this.
    """
    teams: dict[int, int] = {tid: -1 for tid in players}
    tids = [t for t in players if t in colors]
    if len(tids) < 2:
        return teams
    mat = np.array([colors[t] for t in tids])

    if len(tids) >= 6:
        labels3, centers3 = _kmeans(mat, k=3)
        sizes = [int((labels3 == j).sum()) for j in range(3)]
        smallest = int(np.argmin(sizes))
        # a genuinely small third cluster (distinct kit) = the officials
        if 0 < sizes[smallest] <= max(2, round(0.2 * len(tids))):
            mains = [j for j in range(3) if j != smallest]
            for t, lab in zip(tids, labels3):
                teams[t] = -1 if lab == smallest else mains.index(lab)
            return teams

    labels, centers = _kmeans(mat, k=2)
    sep = float(np.linalg.norm(centers[0] - centers[1]))
    for t, lab in zip(tids, labels):
        if sep > 1e-6 and np.linalg.norm(colors[t] - centers[lab]) > 1.0 * sep:
            teams[t] = -1   # fits neither kit: referee or goalkeeper
        else:
            teams[t] = int(lab)
    return teams


def _mark_sideline_officials(players: dict[int, np.ndarray],
                             teams: dict[int, int]) -> None:
    """Tracks living on the frame border are sideline officials/coaches."""
    for tid, arr in players.items():
        v = arr[~np.isnan(arr[:, 0])]
        if len(v) == 0:
            continue
        near_edge = ((v[:, 0] < 0.03) | (v[:, 0] > 0.97)
                     | (v[:, 1] < 0.05) | (v[:, 1] > 0.95))
        if near_edge.mean() > 0.8:
            teams[tid] = -1


def recompute_teams(tracks: "TrackData") -> None:
    """Re-run team assignment on cached tracks (after algorithm changes)."""
    if tracks.colors is None:
        return
    tracks.teams = assign_teams(tracks.players, tracks.colors)
    _mark_sideline_officials(tracks.players, tracks.teams)


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


def _remove_static_candidates(candidates: list[list[tuple[float, float, float]]],
                              cam_affines: np.ndarray | None,
                              fps: float) -> list[list[tuple[float, float, float]]]:
    """Drop ball detections that are static in pitch coordinates.

    Pitch markings (penalty spot, center spot) look exactly like a ball to the
    detector but never move relative to the pitch. Candidates are mapped into
    a camera-stabilized reference frame; any spot that keeps producing
    detections at the same stabilized position for STATIC_SPAN_S seconds is a
    marking, and every candidate near it is removed.
    """
    from collections import defaultdict

    n = len(candidates)
    # cumulative transforms: frame f coords -> frame 0 coords
    C = [np.eye(3)]
    for f in range(1, n):
        A = np.eye(3)
        if cam_affines is not None:
            A[:2] = cam_affines[f]
        # invert (frame f-1 -> f) and compose onto the running transform
        C.append(C[f - 1] @ np.linalg.inv(A))

    stab: list[tuple[int, int, float, float]] = []   # frame, idx, sx, sy
    for f, cands in enumerate(candidates):
        for idx, (x, y, _conf) in enumerate(cands):
            s = C[f] @ np.array([x, y, 1.0])
            stab.append((f, idx, float(s[0]), float(s[1])))

    grid: dict[tuple[int, int], list[tuple[int, int, float, float]]] = defaultdict(list)
    cell = STATIC_TOL
    for item in stab:
        grid[(round(item[2] / cell), round(item[3] / cell))].append(item)

    static: set[tuple[int, int]] = set()
    for items in grid.values():
        if len(items) < STATIC_MIN_HITS:
            continue
        frames = [it[0] for it in items]
        if (max(frames) - min(frames)) / fps < STATIC_SPAN_S:
            continue
        xs = np.array([it[2] for it in items])
        ys = np.array([it[3] for it in items])
        if xs.std() < STATIC_TOL and ys.std() < STATIC_TOL:
            static.update((it[0], it[1]) for it in items)

    if not static:
        return candidates
    return [[c for idx, c in enumerate(cands) if (f, idx) not in static]
            for f, cands in enumerate(candidates)]


def _remove_official_zone_candidates(
        candidates: list[list[tuple[float, float, float]]],
        boxes: dict[int, np.ndarray] | None,
        teams: dict[int, int] | None) -> list[list[tuple[float, float, float]]]:
    """Drop ball candidates inside an official's (expanded) bounding box.

    The assistant referee's flag is a small bright object that the detector
    confuses with the ball — but it lives in the referee's hands. The real
    match ball is never inside an official's silhouette, so any candidate
    overlapping an official's box (padded for the flag's reach) is removed.
    """
    if boxes is None or teams is None:
        return candidates
    official_ids = [t for t, team in teams.items() if team == -1 and t in boxes]
    if not official_ids:
        return candidates
    out = []
    for f, cands in enumerate(candidates):
        keep = []
        for (x, y, conf) in cands:
            inside = False
            for tid in official_ids:
                bx = boxes[tid][f]
                if np.isnan(bx[0]):
                    continue
                x1, y1, x2, y2 = bx
                pw, ph = 0.6 * (x2 - x1), 0.2 * (y2 - y1)   # flag reach padding
                if x1 - pw <= x <= x2 + pw and y1 - ph <= y <= y2 + ph:
                    inside = True
                    break
            if not inside:
                keep.append((x, y, conf))
        out.append(keep)
    return out


def _link_ball(candidates: list[list[tuple[float, float, float]]],
               n: int, players: dict[int, np.ndarray], fps: float,
               cam_affines: np.ndarray | None = None,
               teams: dict[int, int] | None = None,
               boxes: dict[int, np.ndarray] | None = None) -> np.ndarray:
    """Choose the ball trajectory by global path optimization.

    Every detection becomes a node scored by confidence and proximity to a
    tracked player (the real ball lives at someone's feet or in flight
    between players; penalty spots and sideline clutter don't). A dynamic
    program then finds the highest-scoring path through the whole clip, where
    moving between nodes costs more the less physically plausible the jump.
    Unlike greedy frame-by-frame linking, one bad detection can't hijack the
    track: committing to it has to beat every alternative path.
    """
    candidates = _remove_static_candidates(candidates, cam_affines, fps)
    candidates = _remove_official_zone_candidates(candidates, boxes, teams)
    nodes: list[tuple[int, float, float, float]] = []   # frame, x, y, conf
    for f, cands in enumerate(candidates):
        for (x, y, conf) in cands:
            nodes.append((f, x, y, conf))
    pos = np.full((n, 2), np.nan)
    if not nodes:
        return pos

    def emission(f: int, x: float, y: float, conf: float) -> float:
        # proximity counts TEAM players only: the real ball lives at a
        # player's feet or in flight between players. Being near only an
        # official is where false positives live (the ball is never "with"
        # the referee), so officials give no bonus.
        dmin = np.inf
        for tid, arr in players.items():
            if teams is not None and teams.get(tid, -1) == -1:
                continue
            p = arr[f]
            if not np.isnan(p[0]):
                dmin = min(dmin, float(np.hypot(x - p[0], y - p[1])))
        score = BALL_CONF_W * conf
        if np.isfinite(dmin):
            score += 1.2 * max(0.0, 1.0 - dmin / BALL_PROX_NEAR)
            if dmin > BALL_PROX_FAR:
                score -= 1.0
        return score

    nodes.sort(key=lambda t: t[0])
    em = [emission(f, x, y, c) for (f, x, y, c) in nodes]
    best = list(em)                       # best path score ending at node i
    parent = [-1] * len(nodes)

    for i, (fi, xi, yi, _) in enumerate(nodes):
        j = i - 1
        while j >= 0:
            fj, xj, yj, _ = nodes[j]
            gap = fi - fj
            if gap > BALL_MAX_LINK_GAP:
                break
            if gap >= 1:
                dist = float(np.hypot(xi - xj, yi - yj))
                speed = dist * fps / gap
                if speed > BALL_MAX_SPEED and gap < BALL_TELEPORT_MIN_GAP:
                    j -= 1
                    continue    # physically impossible jump: no such edge
                allowed = BALL_STEP_BASE + BALL_STEP_ALLOW * gap
                penalty = min(3.0 * (dist / allowed) ** 2, BALL_TELEPORT_COST)
                penalty += BALL_GAP_COST * (gap - 1)
                cand = best[j] + em[i] - penalty
                if cand > best[i]:
                    best[i] = cand
                    parent[i] = j
            j -= 1

    # backtrack from the best-scoring endpoint
    i = int(np.argmax(best))
    while i != -1:
        f, x, y, _ = nodes[i]
        pos[f] = (x, y)
        i = parent[i]

    pos = _interpolate_gaps(pos, max_gap=15)

    # sanity sweep: delete any residual physically-impossible motion
    # (e.g. interpolation across a legitimate long-gap re-acquisition)
    for i in range(1, n):
        p0, p1 = pos[i - 1], pos[i]
        if np.isnan(p0[0]) or np.isnan(p1[0]):
            continue
        if float(np.linalg.norm(p1 - p0)) * fps > 1.2 * BALL_MAX_SPEED:
            pos[i] = np.nan
    return pos


def _stitch_tracks(players: dict[int, np.ndarray],
                   heights: dict[int, np.ndarray],
                   colors: dict[int, np.ndarray],
                   boxes: dict[int, np.ndarray],
                   fps: float) -> tuple[dict, dict, dict, dict]:
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
                bk = ~np.isnan(boxes[b][:, 0])
                boxes[a][bk] = boxes[b][bk]
                if a in colors and b in colors:
                    colors[a] = (colors[a] + colors[b]) / 2
                del players[b], heights[b], boxes[b]
                colors.pop(b, None)
                merged = True
                break
    return players, heights, colors, boxes


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
        players_here: dict[int, tuple] = {}
        boxes_px = []
        bA = rA.boxes
        if bA is not None and len(bA) > 0 and bA.id is not None:
            xyxy = bA.xyxy.cpu().numpy()
            ids = bA.id.cpu().numpy().astype(int)
            for box, tid in zip(xyxy, ids):
                x1, y1, x2, y2 = box
                boxes_px.append(box)
                players_here[int(tid)] = ((x1 + x2) / 2 / w, y2 / h, (y2 - y1) / h,
                                          x1 / w, y1 / h, x2 / w, y2 / h)
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

    # assemble player arrays + heights
    all_ids = sorted({tid for f in frames_players for tid in f})
    players: dict[int, np.ndarray] = {}
    heights: dict[int, np.ndarray] = {}
    boxes: dict[int, np.ndarray] = {}
    for tid in all_ids:
        arr = np.full((n, 2), np.nan)
        hgt = np.full(n, np.nan)
        bxs = np.full((n, 4), np.nan)
        for f, d in enumerate(frames_players):
            if tid in d:
                arr[f] = d[tid][:2]
                hgt[f] = d[tid][2]
                bxs[f] = d[tid][3:7]
        vis = (~np.isnan(arr[:, 0])).sum()
        med_h = float(np.nanmedian(hgt)) if vis else 0.0
        if vis < fps * MIN_TRACK_S or not (MIN_MEDIAN_H <= med_h <= MAX_MEDIAN_H):
            continue
        players[tid] = arr
        heights[tid] = hgt
        boxes[tid] = bxs

    colors = {t: np.mean(jersey_colors[t], axis=0)
              for t in players if t in jersey_colors}
    players, heights, colors, boxes = _stitch_tracks(players, heights, colors, boxes, fps)

    # smooth small holes left by occlusion
    for tid in players:
        players[tid] = _interpolate_gaps(players[tid], max_gap=10)
        heights[tid] = _interpolate_gaps(heights[tid], max_gap=10)
        boxes[tid] = _interpolate_gaps(boxes[tid], max_gap=10)

    # team assignment BEFORE ball linking, so linking can distrust candidates
    # that are near only an official
    teams = assign_teams(players, colors)
    _mark_sideline_officials(players, teams)

    # ball trajectory: global path optimization, informed by player positions
    affines_np = np.array(affines)
    ball = _link_ball(ball_candidates, n, players, fps, affines_np, teams, boxes)

    return TrackData(fps=fps, n_frames=n, ball=ball, players=players,
                     teams=teams, frame_size=(w, h),
                     heights=heights, cam_affines=affines_np,
                     colors=colors, boxes=boxes, ball_candidates=ball_candidates)


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
    boxes = {}
    for tid, arr in players.items():
        half_w = 0.5 * 0.4 * 0.12   # box width ~40% of height
        boxes[tid] = np.stack([arr[:, 0] - half_w, arr[:, 1] - 0.12,
                               arr[:, 0] + half_w, arr[:, 1]], axis=1)
    affines = np.tile(_IDENTITY, (n, 1, 1))
    return TrackData(fps=fps, n_frames=n, ball=ball, players=players,
                     teams=teams, frame_size=(1280, 720),
                     heights=heights, cam_affines=affines, boxes=boxes)
