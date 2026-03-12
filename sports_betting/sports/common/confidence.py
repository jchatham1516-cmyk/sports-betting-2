from __future__ import annotations


def confidence_score(edge: float, data_quality: float, calibration_quality: float) -> float:
    edge_score = min(1.0, max(0.0, abs(edge) / 0.08))
    score = (0.5 * edge_score) + (0.3 * data_quality) + (0.2 * calibration_quality)
    return round(min(1.0, max(0.0, score)), 3)


def confidence_label(score: float) -> str:
    if score >= 0.82:
        return "A"
    if score >= 0.68:
        return "B"
    if score >= 0.55:
        return "C"
    return "D"
