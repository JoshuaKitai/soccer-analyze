"""Player and ball tracking from raw video.

Uses YOLOv8 (COCO classes: person=0, sports ball=32) with ByteTrack for
persistent player IDs. Teams are assigned by k-means clustering on jersey
colors. All positions are normalized to [0, 1] in image space; a player's
"ground point" is the bottom-center of their bounding box.

Output is a TrackData object consumed by events.py / features.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackData:
    """Normalized tracking output for one clip.

    fps: frames per second of the source video
    n_frames: total frames processed
    ball: (n_frames, 2) array of normalized ball ground positions, NaN where unseen
    players: dict track_id -> (n_frames, 2) array of normalized ground positions,
             NaN where the player is not visible
    teams: dict track_id -> 0 or 1 (jersey-color cluster), -1 if unknown
    frame_size: (width, height) in pixels
    """

    fps: float
    n_frames: int
    ball: np.ndarray
    players: dict[int, np.ndarray]
    teams: dict[int, int]
    frame_size: tuple[int, int]

    def ball_speed(self) -> np.ndarray:
        """Per-frame ball speed in normalized units/second."""
        return _speed(self.ball, self.fps)

    def player_speed(self, tid: int) -> np.ndarray:
        return _speed(self.players[tid], self.fps)


def _speed(pos: np.ndarray, fps: float) -> np.ndarray:
    d = np.diff(pos, axis=0)
    sp = np.linalg.norm(d, axis=1) * fps
    return np.concatenate([[np.nan], sp])


def _interpolate_gaps(pos: np.ndarray, max_gap: int = 12) -> np.ndarray:
    """Linearly interpolate NaN gaps up to max_gap frames (ball flickers a lot)."""
    out = pos.copy()
    n = len(out)
    valid = ~np.isnan(out[:, 0])
    if valid.sum() < 2:
        return out
    idx = np.arange(n)
    for dim in range(2):
        col = out[:, dim]
        good = ~np.isnan(col)
        interp = np.interp(idx, idx[good], col[good])
        # only fill gaps shorter than max_gap
        gap_len = np.zeros(n)
        run = 0
        for i in range(n):
            run = run + 1 if not good[i] else 0
            gap_len[i] = run
        # backward pass so every frame in a gap knows the full gap length
        for i in range(n - 2, -1, -1):
            if gap_len[i] > 0 and gap_len[i + 1] > 0:
                gap_len[i] = max(gap_len[i], gap_len[i + 1])
        fill = (~good) & (gap_len <= max_gap)
        col[fill] = interp[fill]
    return out


def _kmeans2(colors: np.ndarray, iters: int = 25, seed: int = 0) -> np.ndarray:
    """Tiny 2-means over (n, 3) color vectors. Returns cluster labels."""
    rng = np.random.default_rng(seed)
    centers = colors[rng.choice(len(colors), 2, replace=False)].astype(float)
    labels = np.zeros(len(colors), dtype=int)
    for _ in range(iters):
        d = np.linalg.norm(colors[:, None, :] - centers[None, :, :], axis=2)
        labels = d.argmin(axis=1)
        for k in range(2):
            if (labels == k).any():
                centers[k] = colors[labels == k].mean(axis=0)
    return labels


def track_video(video_path: str, model_name: str = "yolov8n.pt",
                conf: float = 0.25, progress: bool = True) -> TrackData:
    """Run detection + tracking over a clip and return normalized tracks."""
    import cv2
    from ultralytics import YOLO

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    model = YOLO(model_name)
    results = model.track(
        source=video_path,
        classes=[0, 32],          # person, sports ball
        conf=conf,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False,
    )

    frames_ball: list[tuple[float, float]] = []
    frames_players: list[dict[int, tuple[float, float]]] = []
    jersey_colors: dict[int, list[np.ndarray]] = {}

    for r in results:
        ball_xy = (np.nan, np.nan)
        players_xy: dict[int, tuple[float, float]] = {}
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

            # ball: highest-confidence sports-ball detection, center point
            ball_mask = cls == 32
            if ball_mask.any():
                b = xyxy[ball_mask][confs[ball_mask].argmax()]
                ball_xy = ((b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h)

            # players: bottom-center ground point per track id
            if ids is not None:
                for box, c, tid in zip(xyxy, cls, ids):
                    if c != 0:
                        continue
                    gx = (box[0] + box[2]) / 2 / w
                    gy = box[3] / h
                    players_xy[int(tid)] = (gx, gy)
                    # jersey crop: upper-middle of the box, for team clustering
                    img = r.orig_img
                    x1, y1, x2, y2 = box.astype(int)
                    cy1 = y1 + int(0.15 * (y2 - y1))
                    cy2 = y1 + int(0.50 * (y2 - y1))
                    cx1 = x1 + int(0.25 * (x2 - x1))
                    cx2 = x1 + int(0.75 * (x2 - x1))
                    crop = img[max(cy1, 0):cy2, max(cx1, 0):cx2]
                    if crop.size > 0:
                        jersey_colors.setdefault(int(tid), []).append(
                            crop.reshape(-1, 3).mean(axis=0))

        frames_ball.append(ball_xy)
        frames_players.append(players_xy)
        if progress and len(frames_ball) % 60 == 0:
            print(f"  tracked {len(frames_ball)} frames...")

    n = len(frames_ball)
    ball = _interpolate_gaps(np.array(frames_ball, dtype=float))

    all_ids = sorted({tid for f in frames_players for tid in f})
    players: dict[int, np.ndarray] = {}
    for tid in all_ids:
        arr = np.full((n, 2), np.nan)
        for i, f in enumerate(frames_players):
            if tid in f:
                arr[i] = f[tid]
        # drop tracks visible for under half a second (spurious detections)
        if (~np.isnan(arr[:, 0])).sum() >= fps * 0.5:
            players[tid] = _interpolate_gaps(arr, max_gap=8)

    # team assignment by jersey color
    teams: dict[int, int] = {tid: -1 for tid in players}
    tids_with_color = [t for t in players if t in jersey_colors]
    if len(tids_with_color) >= 2:
        mean_colors = np.array([np.mean(jersey_colors[t], axis=0) for t in tids_with_color])
        labels = _kmeans2(mean_colors)
        for t, lab in zip(tids_with_color, labels):
            teams[t] = int(lab)

    return TrackData(fps=fps, n_frames=n, ball=ball, players=players,
                     teams=teams, frame_size=(w, h))


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

    return TrackData(fps=fps, n_frames=n, ball=ball, players=players,
                     teams=teams, frame_size=(1280, 720))
