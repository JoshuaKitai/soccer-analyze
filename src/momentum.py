"""Momentum: a per-frame reading of who controls the play and how threateningly.

Output is a smoothed series in [-1, +1]:
    +1  = Team A fully in control and progressing toward their attacking goal
     0  = contested / dead ball
    -1  = Team B fully in control and progressing

Built from three signals while a team possesses the ball:
    base       - simply having the ball (0.4)
    territory  - how deep the ball is in the attacking half (0.3)
    progress   - ball velocity toward the attacking goal (0.3)

Attack direction is inferred from the data: whichever way Team A's
possessions net-move the ball is treated as their attacking direction, and
Team B attacks the other way. An exponential moving average (~0.9 s) turns
the raw per-frame signal into the kind of swinging momentum curve you'd
narrate watching the game.
"""

from __future__ import annotations

import numpy as np

from .events import Events
from .tracking import TrackData


def compute_momentum(tracks: TrackData, events: Events, tau_s: float = 0.9) -> np.ndarray:
    n = tracks.n_frames
    fps = tracks.fps
    ball_x = tracks.ball[:, 0]

    # ball x-velocity (normalized units / s), NaN-safe
    vx = np.full(n, 0.0)
    with np.errstate(invalid="ignore"):
        dv = np.diff(ball_x) * fps
    dv[np.isnan(dv)] = 0.0
    vx[1:] = dv

    team_frames = events.possessing_team_frames

    # infer Team A's attacking direction from net ball progression while they possess
    net_a = float(np.nansum(vx[team_frames == 0]))
    dir_a = 1.0 if net_a >= 0 else -1.0
    dirs = {0: dir_a, 1: -dir_a}

    raw = np.zeros(n)
    for f in range(n):
        t = int(team_frames[f])
        if t not in (0, 1) or np.isnan(ball_x[f]):
            continue
        sign = 1.0 if t == 0 else -1.0
        d = dirs[t]
        territory = np.clip((ball_x[f] - 0.5) * d * 2.0, -1, 1)   # deep in attack = +1
        progress = np.clip(vx[f] * d / 0.30, -1, 1)               # pushing forward = +1
        raw[f] = sign * np.clip(0.4 + 0.3 * territory + 0.3 * progress, -1, 1)

    # exponential smoothing so momentum swings rather than flickers
    alpha = 1.0 - np.exp(-1.0 / (fps * tau_s))
    out = np.zeros(n)
    m = 0.0
    for f in range(n):
        m = m + alpha * (raw[f] - m)
        out[f] = m
    return out
