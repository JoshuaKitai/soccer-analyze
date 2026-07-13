"""Interactive dashboard: video side-by-side with a synced 3D play graph.

Generates a single self-contained dashboard.html per play:

  left   - the annotated clip (tracking overlay), plus the score card
  right  - 3D trajectories in (x, y, time); live markers move through the
           graph in sync with video playback, the ball's path is colored by
           momentum (blue = Team A control, red = Team B)
  bottom - the momentum timeline with a playback cursor and event markers

If the play has no video (demo mode), a play/pause button drives a simulated
clock instead, so the 3D graph and momentum cursor still animate.
"""

from __future__ import annotations

import json

import numpy as np

from .events import Events
from .tracking import TrackData


def _clean(arr) -> list:
    """numpy -> JSON-safe list, NaN -> null."""
    out = []
    for v in np.asarray(arr, dtype=float).ravel():
        out.append(None if np.isnan(v) else round(float(v), 4))
    return out


def build_dashboard(tracks: TrackData, events: Events, momentum: np.ndarray,
                    report: dict, out_html: str, video_file: str | None = None) -> None:
    t = np.arange(tracks.n_frames) / tracks.fps

    players = []
    for tid, arr in tracks.players.items():
        players.append({
            "id": int(tid),
            "team": int(tracks.teams.get(tid, -1)),
            "x": _clean(arr[:, 0]),
            "y": _clean(1 - arr[:, 1]),   # flip so up is up
        })

    ev = ([{"t": round(p.start / tracks.fps, 2), "label": "PASS"} for p in events.passes]
          + [{"t": round(d.start / tracks.fps, 2), "label": "DRIBBLE"} for d in events.dribbles]
          + [{"t": round(s.frame / tracks.fps, 2), "label": "SHOT"} for s in events.shots])
    ev.sort(key=lambda e: e["t"])

    data = {
        "name": report.get("name", "play"),
        "fps": tracks.fps,
        "n": tracks.n_frames,
        "duration": round(tracks.n_frames / tracks.fps, 2),
        "t": _clean(t),
        "bx": _clean(tracks.ball[:, 0]),
        "by": _clean(1 - tracks.ball[:, 1]),
        "players": players,
        "owners": [int(o) for o in events.owner_frames],
        "momentum": _clean(momentum),
        "score": report["score"],
        "subscores": report["subscores"],
        "events": ev,
        "video": video_file,
    }

    from plotly.offline import get_plotlyjs

    html = (_TEMPLATE
            .replace("__TITLE__", f"{data['name']} — play dashboard")
            .replace("__PLOTLYJS__", get_plotlyjs())
            .replace("__DATA__", json.dumps(data)))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink2: #52514e;
    --muted: #898781; --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
    --team-a: #2a78d6; --team-b: #e34948; --ball: #eda100;
  }
  * { box-sizing: border-box; margin: 0; }
  body {
    background: var(--page); color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    padding: 20px; max-width: 1500px; margin: 0 auto;
  }
  h1 { font-size: 20px; margin-bottom: 4px; }
  .sub { color: var(--ink2); font-size: 13px; margin-bottom: 16px; }
  .grid { display: grid; grid-template-columns: minmax(340px, 5fr) minmax(380px, 6fr); gap: 16px; }
  @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px;
  }
  video, .novideo { width: 100%; border-radius: 6px; background: #111; display: block; }
  .novideo { aspect-ratio: 16/9; color: #ddd; display: flex; align-items: center;
             justify-content: center; flex-direction: column; gap: 10px; font-size: 14px; }
  .novideo button {
    font: inherit; padding: 8px 22px; border-radius: 6px; border: none;
    background: var(--team-a); color: #fff; cursor: pointer;
  }
  .scorewrap { display: flex; align-items: center; gap: 18px; margin-top: 14px; }
  .bignum { font-size: 46px; font-weight: 700; line-height: 1; }
  .bignum small { font-size: 16px; font-weight: 400; color: var(--muted); }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip { font-size: 12px; color: var(--ink2); background: var(--page);
          border: 1px solid var(--border); border-radius: 999px; padding: 3px 10px; }
  .chip b { color: var(--ink); }
  .legend { display: flex; gap: 14px; font-size: 12px; color: var(--ink2); margin-top: 8px; }
  .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 4px; }
  #scene { width: 100%; height: 480px; }
  #mom { width: 100%; height: 190px; }
  .momcard { margin-top: 16px; }
  .momtitle { font-size: 13px; color: var(--ink2); margin-bottom: 4px; }
  .momtitle b.a { color: var(--team-a); } .momtitle b.b { color: var(--team-b); }
</style>
</head>
<body>
<h1 id="title"></h1>
<div class="sub">Play the video — the 3D graph and momentum cursor follow along. Drag the 3D view to rotate.</div>

<div class="grid">
  <div class="card">
    <div id="videoslot"></div>
    <div class="scorewrap">
      <div class="bignum"><span id="score"></span><small> / 100</small></div>
      <div class="chips" id="chips"></div>
    </div>
    <div class="legend">
      <span><span class="dot" style="background:var(--team-a)"></span>Team A</span>
      <span><span class="dot" style="background:var(--team-b)"></span>Team B</span>
      <span><span class="dot" style="background:var(--ball)"></span>Ball</span>
      <span><span class="dot" style="background:var(--muted)"></span>Unassigned</span>
    </div>
  </div>
  <div class="card"><div id="scene"></div></div>
</div>

<div class="card momcard">
  <div class="momtitle">Momentum — above the line: <b class="a">Team A</b> in control · below: <b class="b">Team B</b></div>
  <div id="mom"></div>
</div>

<script>__PLOTLYJS__</script>
<script>
const D = __DATA__;
const TEAM = { "0": "#2a78d6", "1": "#e34948", "-1": "#898781" };
const BALL = "#eda100", INK = "#0b0b0b", MUTED = "#898781", GRID = "#e1e0d9", SURFACE = "#fcfcfb";
const MOMSCALE = [[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]];
const FONT = { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: INK };

document.getElementById("title").textContent = D.name + " — play dashboard";
document.getElementById("score").textContent = Math.round(D.score);
document.getElementById("chips").innerHTML = Object.entries(D.subscores)
  .map(([k, v]) => `<span class="chip">${k} <b>${Math.round(v)}</b></span>`).join("");

/* ---------- video / clock ---------- */
let video = null, simPlaying = false, simT = 0, lastTick = null;
const slot = document.getElementById("videoslot");
if (D.video) {
  slot.innerHTML = `<video id="vid" controls muted src="${D.video}"></video>`;
  video = document.getElementById("vid");
} else {
  slot.innerHTML = `<div class="novideo"><div>synthetic demo play — no video</div>
    <button id="playbtn">&#9654; Play</button></div>`;
  document.getElementById("playbtn").addEventListener("click", () => {
    simPlaying = !simPlaying;
    if (simT >= D.duration - 0.05) simT = 0;
    document.getElementById("playbtn").innerHTML = simPlaying ? "&#10074;&#10074; Pause" : "&#9654; Play";
  });
}
function currentTime() {
  if (video) return video.currentTime;
  return simT;
}

/* ---------- 3D scene ---------- */
const axis = { backgroundcolor: SURFACE, gridcolor: GRID, zerolinecolor: GRID,
               color: MUTED, titlefont: { color: INK, size: 12 } };
const traces = [];
for (const p of D.players) {
  traces.push({ type: "scatter3d", mode: "lines", x: p.x, y: p.y, z: D.t,
    line: { color: TEAM[String(p.team)], width: 3 }, opacity: 0.4,
    hoverinfo: "skip", showlegend: false });
}
traces.push({ type: "scatter3d", mode: "lines", x: D.bx, y: D.by, z: D.t,
  line: { color: D.momentum, colorscale: MOMSCALE, cmin: -1, cmax: 1, width: 7 },
  hovertemplate: "ball · t %{z:.1f}s<extra></extra>", showlegend: false });

const iPlayersNow = traces.length;
traces.push({ type: "scatter3d", mode: "markers+text", x: [], y: [], z: [],
  marker: { size: 6, color: [] }, text: [], textposition: "top center",
  textfont: { size: 10, color: INK }, hoverinfo: "skip", showlegend: false });
const iCarrierNow = traces.length;
traces.push({ type: "scatter3d", mode: "markers", x: [], y: [], z: [],
  marker: { size: 14, color: "rgba(237,161,0,0.35)", line: { color: BALL, width: 2 } },
  hoverinfo: "skip", showlegend: false });
const iBallNow = traces.length;
traces.push({ type: "scatter3d", mode: "markers", x: [], y: [], z: [],
  marker: { size: 6, color: BALL, symbol: "diamond" }, hoverinfo: "skip", showlegend: false });

Plotly.newPlot("scene", traces, {
  paper_bgcolor: SURFACE, font: FONT, showlegend: false,
  margin: { l: 0, r: 0, t: 0, b: 0 },
  scene: {
    xaxis: { ...axis, title: "field x", range: [0, 1] },
    yaxis: { ...axis, title: "field y", range: [0, 1] },
    zaxis: { ...axis, title: "time (s)" },
    aspectmode: "cube",
    camera: { eye: { x: 1.7, y: -1.7, z: 0.8 } },
  },
}, { displayModeBar: false, responsive: true });

/* ---------- momentum strip ---------- */
const momPos = D.momentum.map(v => v > 0 ? v : 0);
const momNeg = D.momentum.map(v => v < 0 ? v : 0);
const evX = D.events.map(e => e.t);
const evY = D.events.map((e, i) => (i % 2 ? -1.12 : 1.12));
const evT = D.events.map(e => e.label);
const iCursor = 4;
Plotly.newPlot("mom", [
  { x: D.t, y: momPos, mode: "lines", fill: "tozeroy",
    line: { color: "#2a78d6", width: 1.5 }, fillcolor: "rgba(42,120,214,0.25)", hoverinfo: "skip" },
  { x: D.t, y: momNeg, mode: "lines", fill: "tozeroy",
    line: { color: "#e34948", width: 1.5 }, fillcolor: "rgba(227,73,72,0.22)", hoverinfo: "skip" },
  { x: D.t, y: D.momentum, mode: "lines", line: { color: INK, width: 1 },
    hovertemplate: "t %{x:.1f}s · momentum %{y:.2f}<extra></extra>" },
  { x: evX, y: evY, mode: "markers+text", text: evT, textposition: "middle center",
    textfont: { size: 10, color: "#52514e" }, marker: { size: 1, color: MUTED }, hoverinfo: "skip" },
  { x: [0, 0], y: [-1.3, 1.3], mode: "lines",
    line: { color: INK, width: 1, dash: "dot" }, hoverinfo: "skip" },
], {
  paper_bgcolor: SURFACE, plot_bgcolor: SURFACE, font: FONT, showlegend: false,
  margin: { l: 40, r: 10, t: 6, b: 28 },
  xaxis: { title: "time (s)", color: MUTED, gridcolor: GRID, zeroline: false },
  yaxis: { range: [-1.35, 1.35], color: MUTED, gridcolor: GRID,
           zeroline: true, zerolinecolor: "#c3c2b7", tickvals: [-1, 0, 1] },
}, { displayModeBar: false, responsive: true });

/* ---------- sync loop ---------- */
let lastUpdate = 0;
function frameAt(t) { return Math.max(0, Math.min(D.n - 1, Math.round(t * D.fps))); }

function update(t) {
  const f = frameAt(t);
  const px = [], py = [], pz = [], pc = [], pt = [];
  for (const p of D.players) {
    if (p.x[f] == null) continue;
    px.push(p.x[f]); py.push(p.y[f]); pz.push(D.t[f]);
    pc.push(TEAM[String(p.team)]); pt.push(String(p.id));
  }
  Plotly.restyle("scene", { x: [px], y: [py], z: [pz],
    "marker.color": [pc], text: [pt] }, [iPlayersNow]);

  const owner = D.owners[f];
  if (owner !== -1) {
    const p = D.players.find(q => q.id === owner);
    if (p && p.x[f] != null) {
      Plotly.restyle("scene", { x: [[p.x[f]]], y: [[p.y[f]]], z: [[D.t[f]]] }, [iCarrierNow]);
    }
  } else {
    Plotly.restyle("scene", { x: [[]], y: [[]], z: [[]] }, [iCarrierNow]);
  }

  if (D.bx[f] != null) {
    Plotly.restyle("scene", { x: [[D.bx[f]]], y: [[D.by[f]]], z: [[D.t[f]]] }, [iBallNow]);
  }
  Plotly.restyle("mom", { x: [[t, t]] }, [iCursor]);
}

function tick(ts) {
  if (!video && simPlaying) {
    if (lastTick != null) simT = Math.min(simT + (ts - lastTick) / 1000, D.duration);
    if (simT >= D.duration) {
      simPlaying = false;
      document.getElementById("playbtn").innerHTML = "&#9654; Replay";
    }
  }
  lastTick = ts;
  const t = currentTime();
  if (ts - lastUpdate > 66) {   // ~15 updates/s
    update(t);
    lastUpdate = ts;
  }
  requestAnimationFrame(tick);
}
update(0);
requestAnimationFrame(tick);
</script>
</body>
</html>
"""
