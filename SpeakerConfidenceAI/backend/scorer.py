"""
Hybrid confidence scorer.

    final_score  = 0.55 * ML_score + 0.45 * rule_based_score
    final_level  = "Low"   for final_score < 0.40
                   "Medium" for 0.40 <= final_score < 0.70
                   "High"   otherwise

If the ML model (saved by model/train.py) is not present on disk, the
scorer falls back gracefully to a pure rule-based score so the backend
still works out of the box.

The module is deliberately self-contained — the FastAPI layer only sees
`score_text()` and the returned dict.
"""

from __future__ import annotations

import json
import math
import pickle
import re
from pathlib import Path
from typing import Any

from lexicon import HIGH_CONFIDENCE_WORDS, LOW_CONFIDENCE_WORDS, format_markers

# --------------------------------------------------------------------
# Paths / lazy-loaded artefacts
# --------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BACKEND_DIR.parent / "model" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "best_model.pkl"
META_PATH = ARTIFACT_DIR / "label_meta.json"

_LABELS = [1, 2, 3, 4, 5]
_LABEL_NAMES = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}
_BEST_MODEL_NAME = "Rule-based only"

_ml_pipeline: Any | None = None
_ml_ready: bool = False


def _load_ml_artifacts() -> None:
    """Attempt to load the pickled pipeline. Silent fallback on failure."""
    global _ml_pipeline, _ml_ready, _BEST_MODEL_NAME, _LABELS

    if _ml_ready:
        return

    try:
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as fh:
                _ml_pipeline = pickle.load(fh)
            _ml_ready = True
        if META_PATH.exists():
            meta = json.loads(META_PATH.read_text())
            _LABELS = meta.get("labels", _LABELS)
            _BEST_MODEL_NAME = meta.get("best_model", _BEST_MODEL_NAME)
    except Exception as exc:   # pragma: no cover
        print(f"[scorer] Unable to load ML model: {exc}. Rule-based mode only.")
        _ml_pipeline = None
        _ml_ready = False


# --------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------
def _clean(text: str) -> str:
    t = text.lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-zA-Z\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _rule_score(markers: dict) -> float:
    """
    Map hedge/booster counts to a [0,1] score.
    Boosters push score up, hedges push it down, mild base at 0.5.
    """
    low = markers["low_count"]
    high = markers["high_count"]
    net = high - low
    words = max(markers["lowered"].count(" ") + 1, 1)
    density = net / math.sqrt(words)
    # squash with logistic
    rule = 1.0 / (1.0 + math.exp(-1.6 * density))
    return float(rule)


def _ml_score(text: str) -> tuple[float, list[float]] | tuple[None, None]:
    """
    Produce a [0,1] ML-based confidence and the per-class probabilities.
    Uses the saved best pipeline if available.
    """
    if not _ml_ready or _ml_pipeline is None:
        return None, None

    x = _clean(text)
    if not x:
        return None, None

    try:
        if hasattr(_ml_pipeline, "predict_proba"):
            probs = _ml_pipeline.predict_proba([x])[0]
            classes = list(_ml_pipeline.classes_)
        else:
            # LinearSVC fallback – decision function → softmax
            scores = _ml_pipeline.decision_function([x])[0]
            exp = [math.exp(s - max(scores)) for s in scores]
            total = sum(exp)
            probs = [e / total for e in exp]
            classes = list(_ml_pipeline.classes_)
    except Exception:
        return None, None

    # map labels 1..5 → weights 0, 0.25, 0.5, 0.75, 1.0
    weight = {1: 0.00, 2: 0.25, 3: 0.50, 4: 0.75, 5: 1.00}
    score = sum(weight.get(int(c), 0.5) * float(p) for c, p in zip(classes, probs))

    # return probabilities aligned to LABEL order 1..5 (zeros if missing)
    prob_map = {int(c): float(p) for c, p in zip(classes, probs)}
    ordered = [prob_map.get(lbl, 0.0) for lbl in _LABELS]
    return float(score), ordered


def _combined_level(score01: float) -> str:
    if score01 < 0.40:
        return "Low"
    if score01 < 0.70:
        return "Medium"
    return "High"


def _explanation(text: str, final: float, rule: float,
                 ml: float | None, markers: dict) -> str:
    parts: list[str] = []
    level = _combined_level(final)
    parts.append(f"Overall the speaker sounds **{level.lower()}** in confidence.")
    if markers["high_count"] and markers["low_count"] == 0:
        parts.append(
            f"The text contains {markers['high_count']} assertive marker"
            f"{'s' if markers['high_count']!=1 else ''} "
            f"and no hedging words, which pushes the score up."
        )
    elif markers["low_count"] and markers["high_count"] == 0:
        parts.append(
            f"The text contains {markers['low_count']} hedging word"
            f"{'s' if markers['low_count']!=1 else ''} "
            f"(e.g. 'maybe', 'I think', 'not sure'), which lowers the score."
        )
    elif markers["high_count"] and markers["low_count"]:
        parts.append(
            f"The speaker mixes {markers['high_count']} booster"
            f"{'s' if markers['high_count']!=1 else ''} "
            f"with {markers['low_count']} hedge"
            f"{'s' if markers['low_count']!=1 else ''}, "
            f"so the classifier produced a moderate reading."
        )
    else:
        parts.append(
            "No strong lexical confidence markers were detected, "
            "so the ML model drove the final score."
        )

    if ml is not None:
        parts.append(
            f"The TF-IDF classifier predicted a score of "
            f"{ml*100:.1f}% and the rule-based lexicon gave {rule*100:.1f}%. "
            f"The final score is a 55/45 blend of the two."
        )
    else:
        parts.append(
            f"Running in rule-based mode (train the model to enable ML scoring). "
            f"Lexicon score = {rule*100:.1f}%."
        )
    return " ".join(parts)


# --------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------
def score_text(text: str) -> dict:
    _load_ml_artifacts()

    text = (text or "").strip()
    if not text:
        return {
            "error": "Empty text. Please paste at least one sentence.",
        }

    markers = format_markers(text)
    rule = _rule_score(markers)
    ml, probs = _ml_score(text)

    if ml is None:
        final = rule
    else:
        final = 0.55 * ml + 0.45 * rule

    # clamp & round
    final = max(0.0, min(1.0, final))

    return {
        "text": text,
        "score_percent": round(final * 100, 2),
        "score_normalised": round(final, 4),
        "rule_score": round(rule, 4),
        "ml_score": None if ml is None else round(ml, 4),
        "level": _combined_level(final),
        "best_model": _BEST_MODEL_NAME if _ml_ready else "Rule-based only",
        "explanation": _explanation(text, final, rule, ml, markers),
        "probabilities": {
            str(lbl): round(p, 4) for lbl, p in zip(_LABELS, probs or [0]*len(_LABELS))
        } if probs else None,
        "labels": {str(k): v for k, v in _LABEL_NAMES.items()},
        "markers": {
            "low": [{"word": w, "start": s, "end": e} for w, s, e in markers["low"]],
            "high": [{"word": w, "start": s, "end": e} for w, s, e in markers["high"]],
        },
        "stats": {
            "word_count": len(markers["lowered"].split()),
            "low_count": markers["low_count"],
            "high_count": markers["high_count"],
        },
    }


def available_lexicon():
    return {
        "low_confidence_words": LOW_CONFIDENCE_WORDS,
        "high_confidence_words": HIGH_CONFIDENCE_WORDS,
    }


def model_info() -> dict:
    _load_ml_artifacts()
    metrics = None
    metrics_path = ARTIFACT_DIR / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            metrics = None
    return {
        "ml_ready": _ml_ready,
        "best_model": _BEST_MODEL_NAME,
        "labels": _LABELS,
        "label_names": _LABEL_NAMES,
        "metrics": metrics,
    }
