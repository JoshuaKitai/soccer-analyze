import { useRef, useState } from "react";
import { api } from "../api";

/** Upload a clip, watch the analysis progress, notify when it's done. */
export function Upload({ onDone }: { onDone: (name: string) => void }) {
  const [log, setLog] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFile(file: File) {
    setBusy(true);
    setError(null);
    setLog([`uploading ${file.name}...`]);
    try {
      const { job_id, name } = await api.analyze(file);
      // poll until the pipeline finishes (analysis takes a few minutes on CPU)
      for (;;) {
        await new Promise((r) => setTimeout(r, 2000));
        const job = await api.job(job_id);
        setLog([`analyzing as "${name}"...`, ...job.log.slice(-6)]);
        if (job.status === "done") {
          setLog([`done — scored ${Math.round(job.score ?? 0)}/100`]);
          onDone(name);
          break;
        }
        if (job.status === "error") throw new Error(job.error ?? "analysis failed");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <div className="upload">
      <label className={`uploadbtn ${busy ? "busy" : ""}`}>
        {busy ? "analyzing…" : "+ analyze a clip"}
        <input
          ref={inputRef}
          type="file"
          accept="video/mp4,video/quicktime,video/x-matroska,video/webm,video/avi"
          disabled={busy}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void handleFile(f);
          }}
        />
      </label>
      {log.length > 0 && (
        <pre className="joblog">{log.join("\n")}</pre>
      )}
      {error && <div className="error">{error}</div>}
    </div>
  );
}
