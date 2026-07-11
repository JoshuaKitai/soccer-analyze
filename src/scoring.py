"""Difficulty model: five subscores -> one 0-100 rating.

score = 100 * sigma(k * (w . x - b))

where x is the feature vector, w the dimension weights, and sigma a logistic
squash calibrated so that:
  - a quiet, uncontested play (all features ~0.15) lands near 20
  - a typical contested build-up (~0.45) lands near 50
  - an elite chained sequence under pressure (~0.8) lands near 85+

The weights encode a judgment call you can retune: pressure and technical
skill dominate, tempo and structure matter, the finish is a bonus multiplier
rather than a requirement (a brilliant build-up without a shot still rates).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import DifficultyFeatures

WEIGHTS = {
    "technical": 0.28,
    "pressure": 0.26,
    "speed": 0.16,
    "complexity": 0.18,
    "finish": 0.12,
}

# logistic calibration
_K = 6.0     # steepness
_B = 0.45    # feature level that maps to 50


@dataclass
class DifficultyScore:
    score: float                 # 0-100
    subscores: dict[str, float]  # each 0-100 for display
    weights: dict[str, float]
    detail: dict

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 1),
            "subscores": {k: round(v, 1) for k, v in self.subscores.items()},
            "weights": self.weights,
            "detail": self.detail,
        }


def score_play(features: DifficultyFeatures) -> DifficultyScore:
    x = {
        "technical": features.technical,
        "pressure": features.pressure,
        "speed": features.speed,
        "complexity": features.complexity,
        "finish": features.finish,
    }
    weighted = sum(WEIGHTS[k] * v for k, v in x.items())
    score = 100.0 / (1.0 + np.exp(-_K * (weighted - _B)))
    return DifficultyScore(
        score=float(score),
        subscores={k: 100.0 * v for k, v in x.items()},
        weights=WEIGHTS,
        detail=features.detail,
    )


def fuse_with_vlm(metric_score: float, vlm_score: float,
                  metric_weight: float = 0.65) -> float:
    """Blend the geometry-based score with the VLM's holistic estimate.

    The metric score is grounded in measured positions but blind to context
    (a rabona and a toe-poke can look identical in track space); the VLM sees
    context but can't measure. The blend keeps both honest.
    """
    return float(metric_weight * metric_score + (1 - metric_weight) * vlm_score)
