"""
============================================================
  train.py  —  Speaker Confidence Assessment from Text
============================================================

Trains, evaluates and persists several classical ML pipelines
(TF-IDF + Logistic Regression / Random Forest / Gradient Boosting /
Linear SVM / Stacking) on the Conf_Text_Labels dataset.

Outputs (written next to this script):
    artifacts/best_model.pkl        – sklearn Pipeline (vectorizer + clf)
    artifacts/label_meta.json       – label list + human readable names
    artifacts/metrics.json          – per-model metrics + best selection
    artifacts/confusion_matrix.png  – best-model confusion matrix
    artifacts/model_comparison.png  – F1-score bar chart
    artifacts/class_distribution.png

Labels:
    1 – Very Low, 2 – Low, 3 – Medium, 4 – High, 5 – Very High
"""

from __future__ import annotations

import json
import os
import pickle
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_PATH = ROOT / "dataset" / "Conf_Text_Labels.xlsx"
ART_DIR = HERE / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = {
    1: "Very Low",
    2: "Low",
    3: "Medium",
    4: "High",
    5: "Very High",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def clean_text(t: str) -> str:
    """Light normalisation — lowercase, strip URLs, keep apostrophes."""
    t = str(t).lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-zA-Z\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df[["Text", "Conf Label"]].dropna()
    df["Conf Label"] = pd.to_numeric(df["Conf Label"], errors="coerce")
    df = df.dropna(subset=["Conf Label"])
    df["label"] = df["Conf Label"].astype(int)
    df["clean"] = df["Text"].astype(str).apply(clean_text)
    df = df[df["clean"].str.len() > 1].reset_index(drop=True)
    return df[["Text", "clean", "label"]]


def build_pipelines() -> dict[str, Pipeline]:
    """All pipelines share the same TF-IDF so comparison is apples-to-apples."""
    def tfidf() -> TfidfVectorizer:
        return TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words="english",
            sublinear_tf=True,
        )

    base = [
        ("logreg", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
        ("svc", LinearSVC(C=0.5, class_weight="balanced")),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
        ("nb", MultinomialNB()),
    ]

    return {
        "Logistic Regression": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LinearSVC(C=0.5, class_weight="balanced")),
        ]),
        "Multinomial NB": Pipeline([
            ("tfidf", tfidf()),
            ("clf", MultinomialNB()),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", tfidf()),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42,
                                            class_weight="balanced")),
        ]),
        "Gradient Boosting": Pipeline([
            ("tfidf", tfidf()),
            ("clf", GradientBoostingClassifier(n_estimators=200, random_state=42)),
        ]),
        "Stacking (LR+SVC+RF+NB → LR)": Pipeline([
            ("tfidf", tfidf()),
            ("clf", StackingClassifier(
                estimators=base,
                final_estimator=LogisticRegression(max_iter=1000),
                cv=5,
                n_jobs=-1,
                passthrough=False,
            )),
        ]),
    }


def score_model(model, X_tr, y_tr, X_te, y_te) -> dict:
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_te, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_te, y_pred, average="weighted", zero_division=0)),
        "y_pred": y_pred.tolist(),
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("[1/5] Loading dataset …")
    df = load_dataset(DATA_PATH)
    print(f"       samples={len(df)}   classes={sorted(df['label'].unique())}")

    X = df["clean"].values
    y = df["label"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[2/5] Split: train={len(X_tr)}  test={len(X_te)}")

    # -- class distribution plot ----------------------------------------
    plt.figure(figsize=(6, 3.5))
    counts = pd.Series(y).value_counts().sort_index()
    sns.barplot(x=[LABEL_NAMES[i] for i in counts.index],
                y=counts.values, palette="Blues_d")
    plt.title("Class distribution")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(ART_DIR / "class_distribution.png", dpi=150)
    plt.close()

    print("[3/5] Training models …")
    models = build_pipelines()
    results: dict[str, dict] = {}
    for name, pipe in models.items():
        print(f"  • {name:32s} ", end="", flush=True)
        res = score_model(pipe, X_tr, y_tr, X_te, y_te)
        results[name] = res
        print(f"acc={res['accuracy']:.4f}  f1={res['f1']:.4f}")

    # -- pick best by F1 -------------------------------------------------
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_pipe = models[best_name]
    print(f"[4/5] Best model: {best_name}  (F1={results[best_name]['f1']:.4f})")

    # -- save best pipeline -------------------------------------------------
    with open(ART_DIR / "best_model.pkl", "wb") as fh:
        pickle.dump(best_pipe, fh)

    # -- save label meta -------------------------------------------------
    with open(ART_DIR / "label_meta.json", "w") as fh:
        json.dump({
            "labels": [1, 2, 3, 4, 5],
            "names": LABEL_NAMES,
            "best_model": best_name,
        }, fh, indent=2)

    # -- save metrics table -----------------------------------------------
    summary = {
        name: {k: v for k, v in r.items() if k != "y_pred"}
        for name, r in results.items()
    }
    with open(ART_DIR / "metrics.json", "w") as fh:
        json.dump({"best_model": best_name, "results": summary,
                   "classification_report": classification_report(
                       y_te, results[best_name]["y_pred"], zero_division=0,
                       output_dict=True)},
                  fh, indent=2)

    # -- confusion matrix plot --------------------------------------------
    cm = confusion_matrix(y_te, results[best_name]["y_pred"], labels=[1, 2, 3, 4, 5])
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[LABEL_NAMES[i] for i in [1, 2, 3, 4, 5]],
                yticklabels=[LABEL_NAMES[i] for i in [1, 2, 3, 4, 5]])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(ART_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # -- model comparison bar chart ---------------------------------------
    comp_df = (
        pd.DataFrame(summary).T[["accuracy", "precision", "recall", "f1"]]
        .sort_values("f1", ascending=False)
    )
    comp_df.to_csv(ART_DIR / "model_comparison.csv")
    ax = comp_df.plot(kind="bar", figsize=(9, 4.5), rot=20,
                      color=["#0071e3", "#6e6e73", "#86868b", "#1d1d1f"])
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison (weighted metrics on test split)")
    ax.set_ylabel("score")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(ART_DIR / "model_comparison.png", dpi=150)
    plt.close()

    print("[5/5] Saved artifacts to", ART_DIR)
    for p in sorted(ART_DIR.iterdir()):
        print("      -", p.name)


if __name__ == "__main__":
    main()
