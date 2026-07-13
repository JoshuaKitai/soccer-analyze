import type { Job, PlayData, PlaySummary } from "./types";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const api = {
  plays: () => fetch("/api/plays").then((r) => json<PlaySummary[]>(r)),

  play: (name: string) =>
    fetch(`/api/plays/${encodeURIComponent(name)}`).then((r) => json<PlayData>(r)),

  videoUrl: (name: string) => `/api/plays/${encodeURIComponent(name)}/video`,

  analyze: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return fetch("/api/analyze", { method: "POST", body }).then((r) =>
      json<{ job_id: string; name: string }>(r),
    );
  },

  job: (id: string) => fetch(`/api/jobs/${id}`).then((r) => json<Job>(r)),
};
