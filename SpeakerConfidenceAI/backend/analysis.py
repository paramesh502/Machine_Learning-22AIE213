"""
analysis.py — Feature extraction + three ML analysis utilities:
  1. Feature-Feature Correlation Heatmap
  2. PCA (Principal Component Analysis)
  3. Feature-Target Correlation (raw features and PCA components)

Run standalone:
    python analysis.py
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── paths ────────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent
DATASET_PATH = BACKEND_DIR.parent / "dataset" / "Conf_Text_Labels.xlsx"

# ── lexicons (reuse from lexicon.py) ─────────────────────────────────────────
sys.path.insert(0, str(BACKEND_DIR))
from lexicon import LOW_CONFIDENCE_WORDS, HIGH_CONFIDENCE_WORDS, _scan_non_overlapping

# ── in-memory cache ───────────────────────────────────────────────────────────
_cache: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_features(text: str) -> dict:
    """Extract numeric features from a single text string."""
    lowered = text.lower()

    low_hits, low_spans = _scan_non_overlapping(LOW_CONFIDENCE_WORDS, lowered)
    high_raw, _ = _scan_non_overlapping(HIGH_CONFIDENCE_WORDS, lowered)
    high_hits = [
        (w, s, e) for (w, s, e) in high_raw
        if not any(not (e <= ls or s >= le) for ls, le in low_spans)
    ]

    words = lowered.split()
    word_count = max(len(words), 1)
    low_count = len(low_hits)
    high_count = len(high_hits)
    net = high_count - low_count
    text_length = len(text)
    avg_word_len = sum(len(w) for w in words) / word_count

    # sentence count (split on . ! ?)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = max(len(sentences), 1)
    avg_sentence_len = word_count / sentence_count

    return {
        "word_count": word_count,
        "text_length": text_length,
        "avg_word_length": round(avg_word_len, 4),
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_len, 4),
        "hedge_count": low_count,
        "booster_count": high_count,
        "net_confidence": net,
        "hedge_density": round(low_count / word_count, 4),
        "booster_density": round(high_count / word_count, 4),
    }


def _load_feature_matrix() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset, extract features from Text column, return:
        X  — DataFrame of numeric features
        y  — Series of Conf Label (target)
    Uses in-memory cache after first call.
    """
    if "X" in _cache and "y" in _cache:
        return _cache["X"], _cache["y"]

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_excel(DATASET_PATH)
    records = df["Text"].fillna("").astype(str).apply(_extract_features).tolist()
    X = pd.DataFrame(records)
    y = df["Conf Label"].fillna(0)

    _cache["X"] = X
    _cache["y"] = y
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature-Feature Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def correlation_heatmap() -> dict:
    """
    Returns:
        features  — list of feature name strings
        matrix    — 2-D list of Pearson correlation values (rounded to 4 dp)
    """
    X, _ = _load_feature_matrix()
    corr = X.corr(method="pearson").round(4)
    return {
        "features": corr.columns.tolist(),
        "matrix": corr.values.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. PCA
# ─────────────────────────────────────────────────────────────────────────────

def pca_analysis(n_components: int = 5) -> dict:
    """
    Standardises features, fits PCA, returns explained variance and loadings.

    Returns:
        n_components           — int
        feature_names          — list of str
        explained_variance_ratio — list of float (per component)
        cumulative_variance    — list of float (cumulative)
        loadings               — 2-D list [feature_i][component_j]
    """
    X, _ = _load_feature_matrix()
    n_features = X.shape[1]

    if n_components < 2 or n_components > n_features:
        raise ValueError(
            f"n_components must be between 2 and {n_features}, got {n_components}"
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    evr = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    cumulative = [round(float(v), 4) for v in np.cumsum(pca.explained_variance_ratio_)]

    # loadings[i][j] = correlation of feature i with component j
    loadings = np.round(pca.components_.T, 4).tolist()  # shape: (n_features, n_components)

    return {
        "n_components": n_components,
        "feature_names": X.columns.tolist(),
        "explained_variance_ratio": evr,
        "cumulative_variance": cumulative,
        "loadings": loadings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature-Target Correlation
# ─────────────────────────────────────────────────────────────────────────────

def feature_target_correlation(use_pca: bool = False, n_components: int = 5) -> dict:
    """
    Pearson correlation of each feature (or PCA component) with the target.

    Returns:
        target       — name of the target column
        correlations — list of {name, correlation, abs_correlation} sorted by
                       abs_correlation descending
    """
    X, y = _load_feature_matrix()

    if use_pca:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=min(n_components, X.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        matrix = pd.DataFrame(X_pca, columns=names)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        matrix = pd.DataFrame(X_scaled, columns=X.columns)

    results = []
    for col in matrix.columns:
        r = float(np.corrcoef(matrix[col], y)[0, 1])
        if math.isnan(r):
            r = 0.0
        results.append({
            "name": col,
            "correlation": round(r, 4),
            "abs_correlation": round(abs(r), 4),
        })

    results.sort(key=lambda x: x["abs_correlation"], reverse=True)

    return {
        "target": "Conf Label",
        "correlations": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner — prints all three analyses
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("1. FEATURE-FEATURE CORRELATION HEATMAP")
    print("=" * 60)
    heatmap = correlation_heatmap()
    features = heatmap["features"]
    matrix = heatmap["matrix"]
    print(f"Features ({len(features)}): {features}")
    print("\nCorrelation Matrix:")
    header = f"{'':20s}" + "".join(f"{f[:8]:>10s}" for f in features)
    print(header)
    for i, row in enumerate(matrix):
        line = f"{features[i]:20s}" + "".join(f"{v:>10.4f}" for v in row)
        print(line)

    print("\n" + "=" * 60)
    print("2. PCA (n_components=5)")
    print("=" * 60)
    pca_result = pca_analysis(n_components=5)
    print(f"Features used: {pca_result['feature_names']}")
    print(f"\nExplained Variance Ratio: {pca_result['explained_variance_ratio']}")
    print(f"Cumulative Variance:      {pca_result['cumulative_variance']}")
    print(f"\nLoadings (feature x component):")
    comp_header = f"{'Feature':22s}" + "".join(f"{'PC'+str(j+1):>10s}" for j in range(pca_result['n_components']))
    print(comp_header)
    for i, fname in enumerate(pca_result['feature_names']):
        row_vals = [pca_result['loadings'][i][j] for j in range(pca_result['n_components'])]
        print(f"{fname:22s}" + "".join(f"{v:>10.4f}" for v in row_vals))

    print("\n" + "=" * 60)
    print("3. FEATURE-TARGET CORRELATION (raw features)")
    print("=" * 60)
    ft_raw = feature_target_correlation(use_pca=False)
    print(f"Target: {ft_raw['target']}")
    print(f"\n{'Feature':22s} {'Correlation':>12s} {'|Correlation|':>14s}")
    print("-" * 50)
    for item in ft_raw["correlations"]:
        print(f"{item['name']:22s} {item['correlation']:>12.4f} {item['abs_correlation']:>14.4f}")

    print("\n" + "=" * 60)
    print("3b. FEATURE-TARGET CORRELATION (PCA components)")
    print("=" * 60)
    ft_pca = feature_target_correlation(use_pca=True)
    print(f"Target: {ft_pca['target']}")
    print(f"\n{'Component':22s} {'Correlation':>12s} {'|Correlation|':>14s}")
    print("-" * 50)
    for item in ft_pca["correlations"]:
        print(f"{item['name']:22s} {item['correlation']:>12.4f} {item['abs_correlation']:>14.4f}")
