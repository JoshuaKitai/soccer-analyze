# soccer_analyze — play difficulty rating from video

Feed it a ~15 second soccer clip. It tracks every player and the ball,
figures out what happened (passes, dribbles, shots, pressure), and rates the
play's execution difficulty **0–100**, with 3D visualizations of both the
play itself and where it sits in "difficulty space" relative to other plays.

## How it works

```
clip.mp4
   │
   ▼
[1] tracking.py     YOLOv8 + ByteTrack → per-frame player/ball positions,
                    teams assigned by jersey-color clustering
   │
   ▼
[2] events.py       possession spells → passes / dribbles / shot attempts
   │
   ▼
[3] features.py     five 0–1 difficulty dimensions:
                    technical · pressure · speed · complexity · finish
   │
   ▼
[4] scoring.py      score = 100 · σ(k·(w·x − b))   → 0–100
   │                (optionally fused with Claude's holistic estimate)
   ▼
[5] visualize.py    play_3d.html (trajectories in x,y,time)
                    difficulty_space.html (all plays in 3D metric space)
```

The optional `--vlm` step sends sampled frames to Claude (vision), which
reconstructs the play semantically — play type, key actions, skill elements
(first-time finish, weak foot, nutmeg…) — and produces its own 0–100
estimate. Final score = 0.65 · metric + 0.35 · VLM. The geometry pipeline
measures precisely but can't tell a rabona from a toe-poke; the VLM sees
context but can't measure. Fusing keeps both honest.

## The math (short version)

Each dimension is built from measured quantities, normalized with a
saturating map `soft(x, s) = x / (x + s)` so values scale smoothly 0→1:

| Dimension  | Built from |
|---|---|
| technical  | pass length/flight-speed/tightness at reception, dribble travel + direction changes under pressure, one-touch actions |
| pressure   | nearest-opponent distance to the ball carrier, defenders inside a pressure radius |
| speed      | 75th-percentile ball speed, carrier sprint speed (p90) |
| complexity | chained events per 10s, distinct players involved, possession spells |
| finish     | shot release speed, defenders between shooter and target |

Weighted sum (`technical .28, pressure .26, complexity .18, speed .16,
finish .12`) → logistic squash calibrated so an uncontested tap-in ≈ 20 and
an elite chained sequence under pressure ≈ 85+. Weights live in
`src/scoring.py` — retune them to your taste.

**Coordinate caveat:** v1 works in camera-normalized coordinates, not true
pitch meters. That's fine for *comparing* plays shot from similar angles;
pitch homography is the first roadmap item below.

## Setup

```powershell
pip install -r requirements.txt
```

CPU is fine — YOLOv8-nano processes a 15s clip in ~1–3 minutes. First run
downloads the model weights (~6 MB) automatically.

## Usage

```powershell
python main.py --demo                      # synthetic play, verifies the pipeline
python main.py clips\my_goal.mp4           # analyze a real clip
python main.py clips\my_goal.mp4 --vlm     # + Claude semantic analysis
python main.py clips\my_goal.mp4 --annotate  # + overlay video with tracks
```

Outputs:

- `results/<name>/report.json` — score, subscores, event counts, VLM analysis
- `results/<name>/play_3d.html` — interactive 3D trajectories (open in browser)
- `results/difficulty_space.html` — every play you've analyzed, plotted in 3D
  (technical × pressure × speed), colored by difficulty
- `results/plays.csv` — the accumulated gallery feeding that chart

For `--vlm`, set `ANTHROPIC_API_KEY` in your environment (or `ant auth login`).

## What to expect / limitations (v1)

- **Broadcast or elevated side-view clips work best.** Ground-level phone
  footage behind the goal confuses the possession model.
- The COCO "sports ball" detector misses small/fast balls sometimes; gaps up
  to ~0.4 s are interpolated, longer gaps degrade event detection.
- Shot detection is a heuristic (fast release toward frame edge) — it will
  miss shots toward the camera.
- Scores are *relative*, calibrated by judgment, not trained on labels yet.

## Roadmap (in rough order of impact)

1. **Pitch homography** — detect field lines/keypoints, map tracks to real
   pitch coordinates. Unlocks true distances, real xG geometry, speed in m/s.
   (Roboflow's `sports` repo has a ready keypoint model.)
2. **Fine-tuned detector** — train YOLO on the SoccerNet tracking dataset for
   far better ball/player recall than COCO classes.
3. **Learned scoring** — replace hand-set weights: collect plays + human
   difficulty labels (even ~200 ranked pairs), fit a Bradley-Terry or
   gradient-boosted model on the feature vector. The pipeline already emits
   the features, so this is purely additive.
4. **xT / VAEP integration** — value each action by how much it advances
   scoring probability, using public event-value models.
5. **Action recognition** — a temporal model (VideoMAE / X-CLIP fine-tune)
   to classify skill moves the geometry can't see (rabona, elastico, volley).
