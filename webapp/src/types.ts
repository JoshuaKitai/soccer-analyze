export interface PlayerTrack {
  id: number;
  team: number; // 0, 1, or -1 (official/unknown)
  x: (number | null)[];
  y: (number | null)[];
}

export interface PlayEvent {
  t: number;
  label: string;
}

export interface PlayData {
  name: string;
  fps: number;
  n: number;
  duration: number;
  t: number[];
  bx: (number | null)[];
  by: (number | null)[];
  players: PlayerTrack[];
  owners: number[];
  momentum: number[];
  score: number;
  subscores: Record<string, number>;
  detail: Record<string, number>;
  events: PlayEvent[];
  video: string | null;
}

export interface PlaySummary {
  name: string;
  score: number;
  subscores: Record<string, number>;
  has_video: boolean;
}

export interface Job {
  status: "queued" | "running" | "done" | "error";
  log: string[];
  name: string;
  error: string | null;
  score?: number;
}
