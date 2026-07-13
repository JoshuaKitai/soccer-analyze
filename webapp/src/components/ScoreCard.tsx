import type { PlayData } from "../types";

export function ScoreCard({ play }: { play: PlayData }) {
  return (
    <div className="card scorecard">
      <div className="bignum">
        {Math.round(play.score)}
        <small> / 100</small>
      </div>
      <div className="chips">
        {Object.entries(play.subscores).map(([k, v]) => (
          <span key={k} className="chip">
            {k} <b>{Math.round(v)}</b>
          </span>
        ))}
      </div>
      <div className="chips detail">
        <span className="chip">passes <b>{play.detail.n_passes ?? 0}</b></span>
        <span className="chip">dribbles <b>{play.detail.n_dribbles ?? 0}</b></span>
        <span className="chip">shots <b>{play.detail.n_shots ?? 0}</b></span>
        <span className="chip">players involved <b>{play.detail.players_involved ?? 0}</b></span>
      </div>
    </div>
  );
}
