import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { Upload } from "./components/Upload";
import { PlayViewer } from "./components/PlayViewer";
import { ScoreCard } from "./components/ScoreCard";
import type { PlayData, PlaySummary } from "./types";

export default function App() {
  const [plays, setPlays] = useState<PlaySummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [play, setPlay] = useState<PlayData | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const refresh = useCallback(async (selectName?: string) => {
    const list = await api.plays();
    setPlays(list);
    if (selectName) setSelected(selectName);
    else if (list.length > 0 && !selected) setSelected(list[list.length - 1].name);
  }, [selected]);

  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!selected) return;
    setPlay(null);
    setLoadError(null);
    api
      .play(selected)
      .then(setPlay)
      .catch((e) => setLoadError(e instanceof Error ? e.message : String(e)));
  }, [selected]);

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>soccer-analyze</h1>
        <div className="sub">play difficulty from video</div>
        <Upload onDone={(name) => void refresh(name)} />
        <div className="playlist">
          {plays.map((p) => (
            <button
              key={p.name}
              className={`playitem ${p.name === selected ? "active" : ""}`}
              onClick={() => setSelected(p.name)}
            >
              <span className="playname">{p.name}</span>
              <span className="playscore">{Math.round(p.score)}</span>
            </button>
          ))}
          {plays.length === 0 && <div className="empty">no plays yet — upload a clip</div>}
        </div>
      </aside>
      <main className="main">
        {play ? (
          <>
            <ScoreCard play={play} />
            <PlayViewer play={play} />
          </>
        ) : loadError ? (
          <div className="empty">
            couldn't load this play: {loadError}
            <br />
            (older plays may need re-analysis to generate web data)
          </div>
        ) : selected ? (
          <div className="empty">loading {selected}…</div>
        ) : (
          <div className="empty">select or upload a play</div>
        )}
      </main>
    </div>
  );
}
