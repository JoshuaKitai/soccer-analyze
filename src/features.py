"""Difficulty features: turn events + tracks into five 0-1 subscores.

The five dimensions (each squashed to [0, 1]):

  technical   - how much skill the actions demand: pass length/speed/count,
                dribble turns, one-touch play
  pressure    - how tight the space was: opponent proximity to the ball
                carrier, defenders inside the pressure radius
  speed       - tempo of the play: ball speed percentiles, carrier sprint speed
  complexity  - structural difficulty: number of events chained together,
                distinct players involved, possession retention under transition
  finish      - how hard the final action was: shot release speed, defenders
                between shooter and target, whether a shot happened at all

Each raw metric is normalized by a soft reference scale (`_soft`), which maps
"typical amateur clip" values near 0.4-0.6 and elite values toward 1.0 without
hard clipping artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .events import Events, nearest_opponent_dist
from .tracking import TrackData


def _soft(x: float, scale: float) -> float:
    """Saturating map [0, inf) -> [0, 1); x == scale gives 0.5."""
    if x <= 0 or np.isnan(x):
        return 0.0
    return float(x / (x + scale))


@dataclass
class DifficultyFeatures:
    technical: float
    pressure: float
    speed: float
    complexity: float
    finish: float
    detail: dict

    def as_dict(self) -> dict:
        d = asdict(self)
        return d


def compute_features(tracks: TrackData, events: Events) -> DifficultyFeatures:
    fps = tracks.fps
    dur_s = tracks.n_frames / fps
    ball_speed = tracks.ball_speed()

    # ---------- technical ----------
    n_passes = len(events.passes)
    pass_diff = 0.0
    for p in events.passes:
        # longer, faster passes into tighter space are harder
        pass_diff += (_soft(p.distance, 0.25) * 0.4
                      + _soft(p.ball_peak_speed, 0.8) * 0.3
                      + (1 - _soft(p.receiver_pressure, 0.10)) * 0.3)
    pass_diff = pass_diff / max(n_passes, 1)

    dribble_diff = 0.0
    for d in events.dribbles:
        dribble_diff += (_soft(d.travel, 0.20) * 0.35
                         + _soft(d.direction_changes, 2.5) * 0.35
                         + (1 - _soft(d.mean_pressure, 0.12)) * 0.30)
    dribble_diff = dribble_diff / max(len(events.dribbles), 1)

    # one-touch play: short possession spells followed by a pass
    quick_release = sum(1 for s in events.spells if (s.end - s.start) / fps < 0.6)
    technical = float(np.clip(
        0.40 * pass_diff + 0.35 * dribble_diff
        + 0.15 * _soft(n_passes, 3.0) + 0.10 * _soft(quick_release, 2.0), 0, 1))

    # ---------- pressure ----------
    carrier_pressures = []
    crowd = []
    for s in events.spells:
        for f in range(s.start, s.end + 1, max(1, int(fps / 5))):
            d = nearest_opponent_dist(tracks, f, s.player)
            if not np.isnan(d):
                carrier_pressures.append(d)
            pos = tracks.players[s.player][f]
            if not np.isnan(pos[0]):
                near = 0
                for tid, arr in tracks.players.items():
                    if tracks.teams.get(tid, -1) in (-1, s.team) or tid == s.player:
                        continue
                    q = arr[f]
                    if not np.isnan(q[0]) and np.linalg.norm(pos - q) < 0.15:
                        near += 1
                crowd.append(near)
    mean_press_dist = float(np.mean(carrier_pressures)) if carrier_pressures else 1.0
    mean_crowd = float(np.mean(crowd)) if crowd else 0.0
    if not carrier_pressures and not crowd:
        pressure = 0.0   # no possession observed — no evidence of pressure
    else:
        pressure = float(np.clip(
            0.6 * (1 - _soft(mean_press_dist, 0.10)) + 0.4 * _soft(mean_crowd, 1.5), 0, 1))

    # ---------- speed ----------
    bs = ball_speed[~np.isnan(ball_speed)]
    p75 = float(np.percentile(bs, 75)) if len(bs) else 0.0
    carrier_speeds = []
    for s in events.spells:
        sp = tracks.player_speed(s.player)[s.start:s.end + 1]
        sp = sp[~np.isnan(sp)]
        if len(sp):
            carrier_speeds.append(float(np.percentile(sp, 90)))
    carrier_p90 = float(np.mean(carrier_speeds)) if carrier_speeds else 0.0
    speed = float(np.clip(0.55 * _soft(p75, 0.30) + 0.45 * _soft(carrier_p90, 0.25), 0, 1))

    # ---------- complexity ----------
    n_events = n_passes + len(events.dribbles) + len(events.shots)
    involved = len({s.player for s in events.spells})
    events_per_10s = n_events / max(dur_s / 10.0, 0.1)
    complexity = float(np.clip(
        0.45 * _soft(events_per_10s, 3.0) + 0.35 * _soft(involved, 3.0)
        + 0.20 * _soft(len(events.spells), 4.0), 0, 1))

    # ---------- finish ----------
    if events.shots:
        best = max(events.shots, key=lambda s: s.ball_speed)
        finish = float(np.clip(
            0.45 * _soft(best.ball_speed, 0.7)
            + 0.35 * _soft(best.defenders_ahead, 2.0)
            + 0.20, 0, 1))  # base credit for producing a shot at all
    else:
        finish = 0.0

    detail = {
        "duration_s": round(dur_s, 2),
        "n_passes": n_passes,
        "mean_pass_difficulty": round(pass_diff, 3),
        "n_dribbles": len(events.dribbles),
        "mean_dribble_difficulty": round(dribble_diff, 3),
        "n_shots": len(events.shots),
        "quick_release_actions": quick_release,
        "mean_pressure_distance": round(mean_press_dist, 3),
        "mean_defenders_in_radius": round(mean_crowd, 2),
        "ball_speed_p75": round(p75, 3),
        "carrier_sprint_p90": round(carrier_p90, 3),
        "players_involved": involved,
        "possession_spells": len(events.spells),
    }

    return DifficultyFeatures(technical=technical, pressure=pressure, speed=speed,
                              complexity=complexity, finish=finish, detail=detail)
