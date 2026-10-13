"""Candidate score of the deterministic policy (paper Eq. 3).

    s(a) = w_d / (d + eps) + (w_n + w_v) * c

``d`` is the distance to the candidate's stored 3D position and ``c`` its
current or temporally smoothed CLIP relevance. Every ranking path in the agent
stack calls this function, so the published weights live in exactly one place.
"""

from __future__ import annotations

from typing import Mapping, Optional

EPS = 1e-3
# Distance assumed for a candidate whose distance could not be estimated.
UNKNOWN_DISTANCE = 1.0


def candidate_score(
    distance: Optional[float],
    relevance: Optional[float],
    weights: Mapping[str, float],
    use_relevance: bool = True,
) -> float:
    """Return s(a); with ``use_relevance`` False only the distance term remains."""
    try:
        d = max(float(distance), 0.0)
    except (TypeError, ValueError):
        d = UNKNOWN_DISTANCE
    score = weights["distance"] / (d + EPS)
    if use_relevance and relevance is not None:
        score += (weights["novelty"] + weights["visibility"]) * float(relevance)
    return score


__all__ = ["EPS", "candidate_score"]
