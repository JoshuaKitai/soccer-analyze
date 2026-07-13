// Palette shared with the Python-side visuals (validated categorical set)
export const THEME = {
  surface: "#fcfcfb",
  page: "#f9f9f7",
  ink: "#0b0b0b",
  ink2: "#52514e",
  muted: "#898781",
  grid: "#e1e0d9",
  border: "rgba(11,11,11,0.10)",
  teamA: "#2a78d6",
  teamB: "#e34948",
  ball: "#eda100",
};

export const TEAM_COLOR: Record<string, string> = {
  "0": THEME.teamA,
  "1": THEME.teamB,
  "-1": THEME.muted,
};

// diverging momentum scale: Team B (red) <- neutral -> Team A (blue)
export const MOMENTUM_SCALE: [number, string][] = [
  [0, THEME.teamB],
  [0.5, "#f0efec"],
  [1, THEME.teamA],
];

export const FONT = {
  family: 'system-ui, -apple-system, "Segoe UI", sans-serif',
  color: THEME.ink,
};
