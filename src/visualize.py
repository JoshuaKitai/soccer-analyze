"""3D visualization of plays and their difficulty.

Two figures, both written as self-contained HTML (plotly.js embedded):

  play_3d.html          - the play itself: ball + player trajectories in
                          (x, y, time) space, colored by team
  difficulty_space.html - every analyzed play as a point in
                          (technical, pressure, speed) space, colored by
                          its 0-100 difficulty on a sequential blue ramp

Colors follow a validated categorical/sequential palette: team A blue #2a78d6,
team B red #e34948 (the diverging warm pole - visually "opponents"), ball
yellow #eda100; magnitude uses the one-hue blue ramp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .tracking import TrackData

# palette (light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
TEAM_A = "#2a78d6"   # blue
TEAM_B = "#e34948"   # red
BALL = "#eda100"     # yellow
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

_AXIS = dict(
    backgroundcolor=SURFACE,
    gridcolor=GRID,
    zerolinecolor=GRID,
    color=MUTED,
    title_font=dict(color=INK, size=13),
)

_FONT = dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK)


def _base_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=16, **{k: v for k, v in _FONT.items() if k != "size"})),
        paper_bgcolor=SURFACE,
        font=_FONT,
        legend=dict(font=dict(color=INK, size=12), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=50, b=10),
    )


def plot_play_3d(tracks: TrackData, out_html: str, title: str = "Play trajectories") -> None:
    """Ball + player paths through (x, y, time). Time rises on the z axis."""
    t = np.arange(tracks.n_frames) / tracks.fps
    fig = go.Figure()

    shown_legend = {0: False, 1: False, -1: False}
    team_names = {0: "Team A", 1: "Team B", -1: "Unassigned"}
    team_colors = {0: TEAM_A, 1: TEAM_B, -1: MUTED}

    for tid, arr in tracks.players.items():
        team = tracks.teams.get(tid, -1)
        valid = ~np.isnan(arr[:, 0])
        if valid.sum() < 2:
            continue
        fig.add_trace(go.Scatter3d(
            x=arr[valid, 0], y=1 - arr[valid, 1], z=t[valid],
            mode="lines",
            line=dict(color=team_colors[team], width=3),
            opacity=0.75,
            name=team_names[team],
            legendgroup=team_names[team],
            showlegend=not shown_legend[team],
            hovertemplate=(f"Player {tid} ({team_names[team]})<br>"
                           "x %{x:.2f} · y %{y:.2f}<br>t %{z:.1f}s<extra></extra>"),
        ))
        shown_legend[team] = True

    bv = ~np.isnan(tracks.ball[:, 0])
    if bv.sum() >= 2:
        fig.add_trace(go.Scatter3d(
            x=tracks.ball[bv, 0], y=1 - tracks.ball[bv, 1], z=t[bv],
            mode="lines",
            line=dict(color=BALL, width=6),
            name="Ball",
            hovertemplate="Ball<br>x %{x:.2f} · y %{y:.2f}<br>t %{z:.1f}s<extra></extra>",
        ))

    fig.update_layout(
        **_base_layout(title),
        scene=dict(
            xaxis=dict(title="field x (frame-normalized)", range=[0, 1], **_AXIS),
            yaxis=dict(title="field y (frame-normalized)", range=[0, 1], **_AXIS),
            zaxis=dict(title="time (s)", **_AXIS),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
    )
    fig.write_html(out_html, include_plotlyjs=True, full_html=True)


def plot_difficulty_space(plays: pd.DataFrame, out_html: str) -> None:
    """All analyzed plays in (technical, pressure, speed) space.

    Marker color encodes the 0-100 difficulty (sequential blue ramp);
    marker size encodes structural complexity. Hover shows the full breakdown.
    """
    if plays.empty:
        raise ValueError("No plays to plot")

    hover = [
        (f"<b>{r['name']}</b><br>"
         f"difficulty {r['score']:.0f}/100<br>"
         f"technical {r['technical']:.0f} · pressure {r['pressure']:.0f}<br>"
         f"speed {r['speed']:.0f} · complexity {r['complexity']:.0f} · finish {r['finish']:.0f}")
        for _, r in plays.iterrows()
    ]

    fig = go.Figure(go.Scatter3d(
        x=plays["technical"], y=plays["pressure"], z=plays["speed"],
        mode="markers+text",
        text=plays["name"],
        textposition="top center",
        textfont=dict(color=INK, size=11),
        marker=dict(
            size=8 + 10 * plays["complexity"] / 100,
            color=plays["score"],
            colorscale=[[i / (len(SEQ_BLUES) - 1), c] for i, c in enumerate(SEQ_BLUES)],
            cmin=0, cmax=100,
            colorbar=dict(title=dict(text="Difficulty", font=dict(color=INK)),
                          tickfont=dict(color=MUTED), thickness=14),
            line=dict(color=SURFACE, width=1),
        ),
        hovertext=hover,
        hoverinfo="text",
    ))

    fig.update_layout(
        **_base_layout("Plays in difficulty space"),
        scene=dict(
            xaxis=dict(title="technical (0-100)", range=[0, 100], **_AXIS),
            yaxis=dict(title="pressure (0-100)", range=[0, 100], **_AXIS),
            zaxis=dict(title="speed (0-100)", range=[0, 100], **_AXIS),
            aspectmode="cube",
        ),
    )
    fig.write_html(out_html, include_plotlyjs=True, full_html=True)


def annotate_video(video_path: str, tracks: TrackData, out_path: str) -> None:
    """Write a copy of the clip with player IDs, team colors and ball marker."""
    import cv2

    hex2bgr = lambda h: tuple(int(h[i:i + 2], 16) for i in (5, 3, 1))
    colors = {0: hex2bgr(TEAM_A), 1: hex2bgr(TEAM_B), -1: hex2bgr(MUTED)}
    ball_c = hex2bgr(BALL)

    cap = cv2.VideoCapture(video_path)
    w, h = tracks.frame_size
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), tracks.fps, (w, h))
    i = 0
    while i < tracks.n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, arr in tracks.players.items():
            p = arr[i]
            if np.isnan(p[0]):
                continue
            c = colors[tracks.teams.get(tid, -1)]
            pt = (int(p[0] * w), int(p[1] * h))
            cv2.ellipse(frame, pt, (18, 7), 0, 0, 360, c, 2)
            cv2.putText(frame, str(tid), (pt[0] - 8, pt[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        b = tracks.ball[i]
        if not np.isnan(b[0]):
            cv2.circle(frame, (int(b[0] * w), int(b[1] * h)), 6, ball_c, -1)
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()
