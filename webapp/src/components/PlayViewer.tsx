import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import { api } from "../api";
import { FONT, MOMENTUM_SCALE, TEAM_COLOR, THEME } from "../theme";
import type { PlayData } from "../types";

/** Video side-by-side with the 3D play graph and the momentum timeline.
 *  During playback, live markers move through the 3D graph in sync with the
 *  video, and a cursor sweeps the momentum strip. */
export function PlayViewer({ play }: { play: PlayData }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const sceneRef = useRef<HTMLDivElement>(null);
  const momRef = useRef<HTMLDivElement>(null);
  // sim clock for plays without video (demo)
  const sim = useRef({ playing: false, t: 0 });
  const playBtnRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const scene = sceneRef.current!;
    const mom = momRef.current!;
    let disposed = false;
    let raf = 0;

    const axis = {
      backgroundcolor: THEME.surface,
      gridcolor: THEME.grid,
      zerolinecolor: THEME.grid,
      color: THEME.muted,
      titlefont: { color: THEME.ink, size: 12 },
    };

    // --- static traces: paths through (x, y, time) ---
    const traces: unknown[] = play.players.map((p) => ({
      type: "scatter3d",
      mode: "lines",
      x: p.x,
      y: p.y,
      z: play.t,
      line: { color: TEAM_COLOR[String(p.team)], width: 3 },
      opacity: 0.4,
      hoverinfo: "skip",
      showlegend: false,
    }));
    traces.push({
      type: "scatter3d",
      mode: "lines",
      x: play.bx,
      y: play.by,
      z: play.t,
      line: {
        color: play.momentum,
        colorscale: MOMENTUM_SCALE,
        cmin: -1,
        cmax: 1,
        width: 7,
      },
      hovertemplate: "ball · t %{z:.1f}s<extra></extra>",
      showlegend: false,
    });
    // --- live marker traces, updated during playback ---
    const iPlayersNow = traces.length;
    traces.push({
      type: "scatter3d",
      mode: "markers+text",
      x: [],
      y: [],
      z: [],
      marker: { size: 6, color: [] },
      text: [],
      textposition: "top center",
      textfont: { size: 10, color: THEME.ink },
      hoverinfo: "skip",
      showlegend: false,
    });
    const iCarrierNow = traces.length;
    traces.push({
      type: "scatter3d",
      mode: "markers",
      x: [],
      y: [],
      z: [],
      marker: {
        size: 14,
        color: "rgba(237,161,0,0.35)",
        line: { color: THEME.ball, width: 2 },
      },
      hoverinfo: "skip",
      showlegend: false,
    });
    const iBallNow = traces.length;
    traces.push({
      type: "scatter3d",
      mode: "markers",
      x: [],
      y: [],
      z: [],
      marker: { size: 6, color: THEME.ball, symbol: "diamond" },
      hoverinfo: "skip",
      showlegend: false,
    });

    void Plotly.newPlot(
      scene,
      traces,
      {
        paper_bgcolor: THEME.surface,
        font: FONT,
        showlegend: false,
        margin: { l: 0, r: 0, t: 0, b: 0 },
        scene: {
          xaxis: { ...axis, title: "field x", range: [0, 1] },
          yaxis: { ...axis, title: "field y", range: [0, 1] },
          zaxis: { ...axis, title: "time (s)" },
          aspectmode: "cube",
          camera: { eye: { x: 1.7, y: -1.7, z: 0.8 } },
        },
      },
      { displayModeBar: false, responsive: true },
    );

    // --- momentum strip ---
    const iCursor = 4;
    void Plotly.newPlot(
      mom,
      [
        {
          x: play.t,
          y: play.momentum.map((v) => (v > 0 ? v : 0)),
          mode: "lines",
          fill: "tozeroy",
          line: { color: THEME.teamA, width: 1.5 },
          fillcolor: "rgba(42,120,214,0.25)",
          hoverinfo: "skip",
        },
        {
          x: play.t,
          y: play.momentum.map((v) => (v < 0 ? v : 0)),
          mode: "lines",
          fill: "tozeroy",
          line: { color: THEME.teamB, width: 1.5 },
          fillcolor: "rgba(227,73,72,0.22)",
          hoverinfo: "skip",
        },
        {
          x: play.t,
          y: play.momentum,
          mode: "lines",
          line: { color: THEME.ink, width: 1 },
          hovertemplate: "t %{x:.1f}s · momentum %{y:.2f}<extra></extra>",
        },
        {
          x: play.events.map((e) => e.t),
          y: play.events.map((_, i) => (i % 2 ? -1.12 : 1.12)),
          mode: "markers+text",
          text: play.events.map((e) => e.label),
          textposition: "middle center",
          textfont: { size: 10, color: THEME.ink2 },
          marker: { size: 1, color: THEME.muted },
          hoverinfo: "skip",
        },
        {
          x: [0, 0],
          y: [-1.3, 1.3],
          mode: "lines",
          line: { color: THEME.ink, width: 1, dash: "dot" },
          hoverinfo: "skip",
        },
      ],
      {
        paper_bgcolor: THEME.surface,
        plot_bgcolor: THEME.surface,
        font: FONT,
        showlegend: false,
        margin: { l: 40, r: 10, t: 6, b: 28 },
        xaxis: { title: "time (s)", color: THEME.muted, gridcolor: THEME.grid, zeroline: false },
        yaxis: {
          range: [-1.35, 1.35],
          color: THEME.muted,
          gridcolor: THEME.grid,
          zeroline: true,
          zerolinecolor: "#c3c2b7",
          tickvals: [-1, 0, 1],
        },
      },
      { displayModeBar: false, responsive: true },
    );

    // --- sync loop ---
    const frameAt = (t: number) =>
      Math.max(0, Math.min(play.n - 1, Math.round(t * play.fps)));

    function update(t: number) {
      if (disposed) return;
      const f = frameAt(t);
      const px: number[] = [];
      const py: number[] = [];
      const pz: number[] = [];
      const pc: string[] = [];
      const pt: string[] = [];
      for (const p of play.players) {
        const x = p.x[f];
        const y = p.y[f];
        if (x == null || y == null) continue;
        px.push(x);
        py.push(y);
        pz.push(play.t[f]);
        pc.push(TEAM_COLOR[String(p.team)]);
        pt.push(String(p.id));
      }
      void Plotly.restyle(scene, { x: [px], y: [py], z: [pz], "marker.color": [pc], text: [pt] }, [iPlayersNow]);

      const owner = play.owners[f];
      const op = owner !== -1 ? play.players.find((q) => q.id === owner) : undefined;
      if (op && op.x[f] != null) {
        void Plotly.restyle(scene, { x: [[op.x[f]]], y: [[op.y[f]]], z: [[play.t[f]]] }, [iCarrierNow]);
      } else {
        void Plotly.restyle(scene, { x: [[]], y: [[]], z: [[]] }, [iCarrierNow]);
      }
      if (play.bx[f] != null) {
        void Plotly.restyle(scene, { x: [[play.bx[f]]], y: [[play.by[f]]], z: [[play.t[f]]] }, [iBallNow]);
      }
      void Plotly.restyle(mom, { x: [[t, t]] }, [iCursor]);
    }

    let lastUpdate = 0;
    let lastTs: number | null = null;
    function tick(ts: number) {
      if (disposed) return;
      if (!play.video && sim.current.playing && lastTs != null) {
        sim.current.t = Math.min(sim.current.t + (ts - lastTs) / 1000, play.duration);
        if (sim.current.t >= play.duration) {
          sim.current.playing = false;
          if (playBtnRef.current) playBtnRef.current.textContent = "▶ Replay";
        }
      }
      lastTs = ts;
      const t = videoRef.current ? videoRef.current.currentTime : sim.current.t;
      if (ts - lastUpdate > 66) {
        update(t);
        lastUpdate = ts;
      }
      raf = requestAnimationFrame(tick);
    }
    update(0);
    raf = requestAnimationFrame(tick);

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      Plotly.purge(scene);
      Plotly.purge(mom);
    };
  }, [play]);

  const togglePlay = () => {
    sim.current.playing = !sim.current.playing;
    if (sim.current.t >= play.duration - 0.05) sim.current.t = 0;
    if (playBtnRef.current) {
      playBtnRef.current.textContent = sim.current.playing ? "⏸ Pause" : "▶ Play";
    }
  };

  return (
    <div className="viewer">
      <div className="grid">
        <div className="card">
          {play.video ? (
            <video ref={videoRef} controls muted src={api.videoUrl(play.name)} />
          ) : (
            <div className="novideo">
              <div>synthetic demo play — no video</div>
              <button ref={playBtnRef} onClick={togglePlay}>
                ▶ Play
              </button>
            </div>
          )}
          <div className="legend">
            <span><span className="dot" style={{ background: THEME.teamA }} />Team A</span>
            <span><span className="dot" style={{ background: THEME.teamB }} />Team B</span>
            <span><span className="dot" style={{ background: THEME.ball }} />Ball</span>
            <span><span className="dot" style={{ background: THEME.muted }} />Official</span>
          </div>
        </div>
        <div className="card">
          <div ref={sceneRef} className="scene" />
        </div>
      </div>
      <div className="card momcard">
        <div className="momtitle">
          Momentum — above the line: <b style={{ color: THEME.teamA }}>Team A</b> in
          control · below: <b style={{ color: THEME.teamB }}>Team B</b>
        </div>
        <div ref={momRef} className="mom" />
      </div>
    </div>
  );
}
