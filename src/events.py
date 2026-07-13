"""Event extraction: turn raw tracks into soccer semantics.

From TrackData we derive:
  - possession spells (who has the ball, when)
  - passes (possession moves between teammates with a ball-speed spike)
  - dribbles (sustained carries with movement, especially under pressure)
  - shot attempts (fast ball release toward the attacking edge of the frame)

Everything operates in normalized image coordinates, so distances are
relative to the camera view, not true pitch meters. That is good enough for
comparative difficulty scoring; pitch homography is the roadmap upgrade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tracking import TrackData

# --- tunable thresholds (normalized units) ---
POSSESSION_RADIUS = 0.06      # max ball<->player distance to count as control
MIN_SPELL_FRAMES = 5          # ignore sub-1/6s possession blips
PASS_MAX_GAP_S = 1.5          # ball in flight longer than this isn't a pass
PASS_MIN_DIST = 0.04          # tiny position changes aren't passes
DRIBBLE_MIN_S = 0.8           # carry must last this long
DRIBBLE_MIN_TRAVEL = 0.05     # and cover this much ground
SHOT_SPEED = 0.55             # normalized units/sec — very fast release
SHOT_EDGE_ZONE = 0.22         # ball must be heading into the outer 22% of frame


@dataclass
class Spell:
    player: int
    team: int
    start: int   # frame index
    end: int     # inclusive


@dataclass
class Pass:
    frm: int
    to: int
    team: int
    start: int
    end: int
    distance: float
    ball_peak_speed: float
    receiver_pressure: float   # nearest-opponent distance at reception (small = tight)


@dataclass
class Dribble:
    player: int
    team: int
    start: int
    end: int
    travel: float
    mean_pressure: float       # mean nearest-opponent distance while carrying
    direction_changes: int


@dataclass
class Shot:
    player: int
    team: int
    frame: int
    ball_speed: float
    defenders_ahead: int


@dataclass
class Events:
    spells: list[Spell]
    passes: list[Pass]
    dribbles: list[Dribble]
    shots: list[Shot]
    possessing_team_frames: np.ndarray   # per-frame team in possession (-1 = none)
    owner_frames: np.ndarray             # per-frame owning player id (-1 = none)


def nearest_opponent_dist(tracks: TrackData, frame: int, player: int) -> float:
    """Distance from `player` to the closest opposing player at `frame`."""
    team = tracks.teams.get(player, -1)
    pos = tracks.players[player][frame]
    if np.isnan(pos[0]):
        return np.nan
    best = np.inf
    for tid, arr in tracks.players.items():
        if tid == player or tracks.teams.get(tid, -1) == team:
            continue
        q = arr[frame]
        if np.isnan(q[0]):
            continue
        best = min(best, float(np.linalg.norm(pos - q)))
    return best if np.isfinite(best) else np.nan


def _possession_per_frame(tracks: TrackData) -> np.ndarray:
    """Per-frame owning player id, or -1. Nearest player within radius."""
    n = tracks.n_frames
    owner = np.full(n, -1, dtype=int)
    for i in range(n):
        b = tracks.ball[i]
        if np.isnan(b[0]):
            continue
        best_d, best_t = POSSESSION_RADIUS, -1
        for tid, arr in tracks.players.items():
            p = arr[i]
            if np.isnan(p[0]):
                continue
            d = float(np.linalg.norm(b - p))
            if d < best_d:
                best_d, best_t = d, tid
        owner[i] = best_t
    return owner


def _spells(owner: np.ndarray, tracks: TrackData) -> list[Spell]:
    spells: list[Spell] = []
    start = 0
    for i in range(1, len(owner) + 1):
        if i == len(owner) or owner[i] != owner[start]:
            if owner[start] != -1 and (i - start) >= MIN_SPELL_FRAMES:
                spells.append(Spell(player=int(owner[start]),
                                    team=tracks.teams.get(int(owner[start]), -1),
                                    start=start, end=i - 1))
            start = i
    return spells


def extract_events(tracks: TrackData) -> Events:
    owner = _possession_per_frame(tracks)
    spells = _spells(owner, tracks)
    ball_speed = tracks.ball_speed()
    fps = tracks.fps

    passes: list[Pass] = []
    dribbles: list[Dribble] = []
    shots: list[Shot] = []

    # --- passes: consecutive same-team spells by different players ---
    for a, b in zip(spells, spells[1:]):
        gap_s = (b.start - a.end) / fps
        if a.player == b.player or a.team != b.team or a.team == -1:
            continue
        if gap_s > PASS_MAX_GAP_S:
            continue
        p_from = tracks.players[a.player][a.end]
        p_to = tracks.players[b.player][b.start]
        dist = float(np.linalg.norm(p_to - p_from))
        if dist < PASS_MIN_DIST:
            continue
        flight = ball_speed[a.end:b.start + 1]
        peak = float(np.nanmax(flight)) if len(flight) and not np.all(np.isnan(flight)) else 0.0
        pressure = nearest_opponent_dist(tracks, b.start, b.player)
        passes.append(Pass(frm=a.player, to=b.player, team=a.team,
                           start=a.end, end=b.start, distance=dist,
                           ball_peak_speed=peak,
                           receiver_pressure=pressure if not np.isnan(pressure) else 1.0))

    # --- dribbles: long spells with real travel ---
    for s in spells:
        dur_s = (s.end - s.start) / fps
        if dur_s < DRIBBLE_MIN_S:
            continue
        path = tracks.players[s.player][s.start:s.end + 1]
        valid = path[~np.isnan(path[:, 0])]
        if len(valid) < 2:
            continue
        travel = float(np.sum(np.linalg.norm(np.diff(valid, axis=0), axis=1)))
        if travel < DRIBBLE_MIN_TRAVEL:
            continue
        pressures = [nearest_opponent_dist(tracks, f, s.player)
                     for f in range(s.start, s.end + 1, max(1, int(fps / 6)))]
        pressures = [p for p in pressures if not np.isnan(p)]
        # direction changes: sign flips in heading over ~1/3s windows
        step = max(1, int(fps / 3))
        headings = np.diff(valid[::step], axis=0)
        angles = np.arctan2(headings[:, 1], headings[:, 0])
        turns = int(np.sum(np.abs(np.diff(np.unwrap(angles))) > np.pi / 4)) if len(angles) > 1 else 0
        dribbles.append(Dribble(player=s.player, team=s.team, start=s.start,
                                end=s.end, travel=travel,
                                mean_pressure=float(np.mean(pressures)) if pressures else 1.0,
                                direction_changes=turns))

    # --- shots: fast release out of a spell toward a lateral edge ---
    for s in spells:
        f0 = s.end
        window = ball_speed[f0:min(f0 + int(0.5 * fps), tracks.n_frames)]
        if len(window) == 0 or np.all(np.isnan(window)):
            continue
        peak = float(np.nanmax(window))
        if peak < SHOT_SPEED:
            continue
        # heading: where does the ball end up shortly after release?
        f1 = min(f0 + int(0.6 * fps), tracks.n_frames - 1)
        end_x = tracks.ball[f1][0]
        if np.isnan(end_x):
            continue
        if end_x > 1 - SHOT_EDGE_ZONE or end_x < SHOT_EDGE_ZONE:
            release = tracks.players[s.player][f0]
            defenders = 0
            for tid, arr in tracks.players.items():
                if tracks.teams.get(tid, -1) in (-1, s.team):
                    continue
                q = arr[f0]
                if np.isnan(q[0]):
                    continue
                same_side = (q[0] > release[0]) == (end_x > release[0])
                if same_side:
                    defenders += 1
            shots.append(Shot(player=s.player, team=s.team, frame=f0,
                              ball_speed=peak, defenders_ahead=defenders))

    team_frames = np.array([tracks.teams.get(int(o), -1) if o != -1 else -1 for o in owner])
    return Events(spells=spells, passes=passes, dribbles=dribbles,
                  shots=shots, possessing_team_frames=team_frames,
                  owner_frames=owner)
