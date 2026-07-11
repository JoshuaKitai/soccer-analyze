"""Optional VLM layer: Claude watches sampled frames and produces a semantic
read of the play - what happened, what made it hard, and its own 0-100
difficulty estimate. This complements the geometry pipeline, which measures
precisely but has no idea whether a touch was a rabona or a shin-bounce.

Requires ANTHROPIC_API_KEY (or an `ant auth login` profile) in the environment.
"""

from __future__ import annotations

import base64

from pydantic import BaseModel, Field


class PlayAnalysis(BaseModel):
    description: str = Field(description="2-3 sentence account of what happens in the play")
    play_type: str = Field(description="e.g. team goal, individual goal, tap-in, counter-attack, build-up, set piece")
    key_actions: list[str] = Field(description="ordered list of the notable actions (passes, dribbles, skills, shot)")
    skill_elements: list[str] = Field(description="specific technical elements that raise difficulty (first-time finish, weak foot, volley, nutmeg, ...)")
    difficulty_estimate: int = Field(ge=0, le=100, description="holistic difficulty of the play, 0-100")
    reasoning: str = Field(description="1-2 sentences on why that difficulty rating")


def sample_frames(video_path: str, n_frames: int = 8, max_width: int = 960) -> list[bytes]:
    """Evenly sample n_frames from the clip as JPEG bytes."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Cannot read frames from {video_path}")
    idxs = [int(i * (total - 1) / (n_frames - 1)) for i in range(n_frames)]
    frames: list[bytes] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if w > max_width:
            frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            frames.append(buf.tobytes())
    cap.release()
    return frames


def analyze_play(video_path: str, model: str = "claude-opus-4-8") -> PlayAnalysis:
    """Send sampled frames to Claude and get a structured play analysis."""
    import anthropic

    client = anthropic.Anthropic()
    frames = sample_frames(video_path)

    content: list[dict] = [{
        "type": "text",
        "text": (
            "These are frames sampled in order from a ~15 second soccer clip. "
            "Analyze the play: reconstruct what happens across the frames, "
            "identify the key actions and technical skill elements, and rate "
            "how difficult the play was to execute on a 0-100 scale "
            "(0 = trivial tap-in with no pressure, 50 = solid contested play, "
            "100 = generational skill under maximum pressure). Judge execution "
            "difficulty, not outcome - a missed wonder-strike still rates high."
        ),
    }]
    for i, jpg in enumerate(frames):
        content.append({"type": "text", "text": f"Frame {i + 1}:"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(jpg).decode(),
            },
        })

    response = client.messages.parse(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": content}],
        output_format=PlayAnalysis,
    )
    return response.parsed_output
