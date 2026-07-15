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
# Possession radius scales with how large the player appears on screen:
# radius = POSSESSION_H_FACTOR * bbox_height, clamped. A player ~1.8m tall
# controlling a ball within ~1m maps to roughly 0.55x their height.
POSSESSION_H_FACTOR = 0.55
POSSESSION_R_MIN = 0.02
POSSESSION_R_MAX = 0.12
HYSTERESIS_KEEP = 1.5         # current owner keeps the ball within 1.5x radius
HYSTERESIS_STEAL = 0.6        # ...unless someone else is decisively closer
STEAL_CONFIRM_FRAMES = 4      # ...for this many consecutive frames
FLIGHT_SPEED = 0.35           # ball faster than this is in flight: nobody can
                              # ACQUIRE it (the incumbent may keep it)
MIN_SPELL_FRAMES = 4          # ignore sub-1/7s possession blips
SPELL_MERGE_GAP_S = 0.35      # rejoin same-player spells split by blind frames
PASS_MAX_GAP_S = 1.5          # ball in flight longer than this isn't a pass
PASS_MIN_DIST = 0.04          # tiny position changes aren't passes
PASS_MIN_FLIGHT = 0.30        # the ball must actually FLY between owners —
                              # a carried ball moves at ~0.1-0.2; without this,
                              # a track-ID switch on the dribbler reads as a
                              # "pass" between two phantom players
PASS_FLIGHT_FACTOR = 1.8      # ...and fly much faster than the giver was
                              # running: a sprint knock-on only slightly
                              # outpaces the sprinter, a real pass doesn't
DRIBBLE_MIN_S = 0.8           # carry must last this long
DRIBBLE_MIN_TRAVEL = 0.05     # and cover this much ground
SHOT_SPEED = 0.45             # normalized units/sec — very fast release
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
        other = tracks.teams.get(tid, -1)
        if tid == player or other == team or other == -1:   # skip teammates + refs
            continue
        q = arr[frame]
        if np.isnan(q[0]):
            continue
        best = min(best, float(np.linalg.norm(pos - q)))
    return best if np.isfinite(best) else np.nan


def _possession_radius(tracks: TrackData, tid: int, frame: int) -> float:
    h = tracks.height_at(tid, frame)
    return float(np.clip(POSSESSION_H_FACTOR * h, POSSESSION_R_MIN, POSSESSION_R_MAX))


def _possession_per_frame(tracks: TrackData) -> np.ndarray:
    """Per-frame owning player id, or -1.

    A player qualifies when the ball is within a radius scaled to their
    on-screen size (so a zoomed-out broadcast doesn't demand impossible
    closeness). Officials (team -1) can never own the ball. Hysteresis keeps
    possession stable, and a steal must persist for several consecutive
    frames before the owner switches — one frame of a defender lunging close
    doesn't flip possession.
    """
    n = tracks.n_frames
    owner = np.full(n, -1, dtype=int)
    ball_speed = tracks.ball_speed()
    prev = -1
    pending, pend_count = -1, 0
    for i in range(n):
        b = tracks.ball[i]
        if np.isnan(b[0]):
            owner[i] = -1
            prev, pending, pend_count = -1, -1, 0
            continue
        in_flight = not np.isnan(ball_speed[i]) and ball_speed[i] > FLIGHT_SPEED
        # ratio = distance / that player's own possession radius
        ratios: dict[int, float] = {}
        for tid, arr in tracks.players.items():
            if tracks.teams.get(tid, -1) == -1:
                continue    # referees and sideline officials can't possess
            p = arr[i]
            if np.isnan(p[0]):
                continue
            r = _possession_radius(tracks, tid, i)
            ratios[tid] = float(np.linalg.norm(b - p)) / r
        if not ratios:
            owner[i] = -1
            prev, pending, pend_count = -1, -1, 0
            continue
        best_t = min(ratios, key=ratios.get)
        best = ratios[best_t]
        chosen = -1
        if prev in ratios and ratios[prev] <= HYSTERESIS_KEEP:
            chosen = prev
            challenger = (best_t != prev and best <= HYSTERESIS_STEAL * ratios[prev]
                          and best <= 1.0 and not in_flight)
            if challenger:
                if pending == best_t:
                    pend_count += 1
                else:
                    pending, pend_count = best_t, 1
                if pend_count >= STEAL_CONFIRM_FRAMES:
                    chosen = best_t
                    pending, pend_count = -1, 0
            else:
                pending, pend_count = -1, 0
        elif best <= 1.0 and not in_flight:
            chosen = best_t
            pending, pend_count = -1, 0
        owner[i] = chosen
        prev = chosen
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
    # rejoin spells of the same player split by short blind gaps
    merge_gap = int(SPELL_MERGE_GAP_S * tracks.fps)
    merged: list[Spell] = []
    for s in spells:
        if merged and merged[-1].player == s.player and s.start - merged[-1].end <= merge_gap:
            merged[-1].end = s.end
        else:
            merged.append(s)
    return merged


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
        giver_sp = tracks.player_speed(a.player)[max(a.end - 3, 0):a.end + 1]
        giver_sp = giver_sp[~np.isnan(giver_sp)]
        giver_speed = float(np.median(giver_sp)) if len(giver_sp) else 0.0
        if peak < max(PASS_MIN_FLIGHT, PASS_FLIGHT_FACTOR * giver_speed):
            continue    # ball never outran the carrier: an ID switch or a
                        # dribble knock-on, not a pass
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

    # --- shots: fast release where possession never resumes ---
    # (an edge-of-frame heading requirement fails on broadcast footage because
    # the camera pans with the ball; "released hard and nobody controls it
    # again" is camera-proof)
    pass_release_frames = {p.start for p in passes}
    for k, s in enumerate(spells):
        f0 = s.end
        if f0 in pass_release_frames:
            continue    # this release already resolved into a completed pass
        window = ball_speed[f0:min(f0 + int(0.5 * fps), tracks.n_frames)]
        if len(window) == 0 or np.all(np.isnan(window)):
            continue
        peak = float(np.nanmax(window))
        if peak < SHOT_SPEED:
            continue
        next_start = spells[k + 1].start if k + 1 < len(spells) else np.inf
        dead_time_s = (next_start - f0) / fps
        f1 = min(f0 + int(0.6 * fps), tracks.n_frames - 1)
        end_x = tracks.ball[f1][0]
        toward_edge = (not np.isnan(end_x)
                       and (end_x > 1 - SHOT_EDGE_ZONE or end_x < SHOT_EDGE_ZONE))
        if not (dead_time_s > 1.0 or toward_edge):
            continue
        release = tracks.players[s.player][f0]
        shot_dir_x = (end_x - release[0]) if not np.isnan(end_x) else 0.0
        defenders = 0
        for tid, arr in tracks.players.items():
            if tracks.teams.get(tid, -1) in (-1, s.team):
                continue
            q = arr[f0]
            if np.isnan(q[0]) or np.isnan(release[0]):
                continue
            if (q[0] - release[0]) * shot_dir_x > 0:   # between shooter and target side
                defenders += 1
        shots.append(Shot(player=s.player, team=s.team, frame=f0,
                          ball_speed=peak, defenders_ahead=defenders))

    team_frames = np.array([tracks.teams.get(int(o), -1) if o != -1 else -1 for o in owner])
    return Events(spells=spells, passes=passes, dribbles=dribbles,
                  shots=shots, possessing_team_frames=team_frames,
                  owner_frames=owner)
