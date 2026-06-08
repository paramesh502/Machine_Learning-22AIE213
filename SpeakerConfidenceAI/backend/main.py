"""
FastAPI application for Speaker Confidence Assessment.

Endpoints
---------
GET  /                                    — liveness probe & basic info
GET  /api/info                            — backend + model info
GET  /api/lexicon                         — the low/high confidence word lists
POST /api/predict                         — hybrid ML + rule-based confidence score
GET  /api/analysis/correlation-heatmap   — feature-feature Pearson correlation matrix
GET  /api/analysis/pca                   — PCA explained variance + loadings
GET  /api/analysis/feature-target-correlation — feature/PCA vs target correlations
POST /api/analysis/lime                  — LIME explanation for a text sample

Run locally:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from scorer import available_lexicon, model_info, score_text

app = FastAPI(
    title="Speaker Confidence Assessment API",
    description=(
        "Hybrid NLP backend that predicts how confident a speaker sounds "
        "based on text. Combines a TF-IDF classifier trained on the "
        "Conf_Text_Labels corpus with a curated hedging / booster lexicon."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000,
                      description="Raw text whose speaker confidence is to be assessed.")


# --------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "name": "Speaker Confidence Assessment API",
        "version": "1.0.0",
        "endpoints": ["/api/info", "/api/lexicon", "/api/predict"],
    }


@app.get("/api/info")
def info():
    return model_info()


@app.get("/api/lexicon")
def lexicon():
    return available_lexicon()


@app.post("/api/predict")
def predict(payload: PredictRequest):
    return score_text(payload.text)


# ── Analysis endpoints ────────────────────────────────────────────────────────

@app.get("/api/analysis/correlation-heatmap")
def analysis_heatmap():
    """Return pairwise Pearson correlation matrix for extracted text features."""
    try:
        from analysis import correlation_heatmap
        return correlation_heatmap()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail={"features": [], "matrix": [], "error": str(e)})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"features": [], "matrix": [], "error": str(e)})


@app.get("/api/analysis/pca")
def analysis_pca(n_components: int = Query(default=5, ge=2)):
    """Run PCA on extracted features and return explained variance + loadings."""
    try:
        from analysis import pca_analysis
        return pca_analysis(n_components=n_components)
    except ValueError as e:
        raise HTTPException(status_code=422, detail={"error": str(e)})
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail={"error": str(e)})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


@app.get("/api/analysis/feature-target-correlation")
def analysis_feature_target(use_pca: bool = Query(default=False)):
    """Return Pearson correlation of each feature (or PCA component) with the target."""
    try:
        from analysis import feature_target_correlation
        return feature_target_correlation(use_pca=use_pca)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail={"error": str(e)})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


# ── LIME endpoint ─────────────────────────────────────────────────────────────

class LimeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000,
                      description="Text to explain with LIME.")
    num_features: int = Field(default=10, ge=1, le=30,
                              description="Number of top features to return.")


@app.post("/api/analysis/lime")
def analysis_lime(payload: LimeRequest):
    """
    Run LIME (Local Interpretable Model-agnostic Explanations) on the given text.

    Uses the same TF-IDF + Logistic Regression pipeline as Lab Session 10.
    Falls back to a rule-based explanation if the ML model is not trained yet.
    """
    try:
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        import lime.lime_text

        DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "Conf_Text_Labels.xlsx"

        if not DATASET_PATH.exists():
            raise HTTPException(status_code=503, detail={"error": f"Dataset not found: {DATASET_PATH}"})

        # Load & prepare data (cached in module-level dict for speed)
        if not hasattr(analysis_lime, "_cache"):
            df = pd.read_excel(DATASET_PATH)
            df = df[["Text", "Conf Label"]].dropna()
            df["Text"] = df["Text"].astype(str)
            df["Conf Label"] = df["Conf Label"].astype(int)

            vectorizer = TfidfVectorizer(stop_words="english", max_features=300, lowercase=True)
            X = vectorizer.fit_transform(df["Text"])
            y = df["Conf Label"]

            model = LogisticRegression(max_iter=3000)
            model.fit(X, y)

            analysis_lime._cache = {"vectorizer": vectorizer, "model": model}

        vectorizer = analysis_lime._cache["vectorizer"]
        model = analysis_lime._cache["model"]

        class_names = [str(c) for c in model.classes_]

        def predictor(texts):
            X = vectorizer.transform(texts)
            return model.predict_proba(X)

        explainer = lime.lime_text.LimeTextExplainer(class_names=class_names)
        explanation = explainer.explain_instance(
            payload.text,
            predictor,
            num_features=payload.num_features,
            top_labels=1,          # only compute for the top predicted class
        )

        # Predicted class
        X_input = vectorizer.transform([payload.text])
        probs = model.predict_proba(X_input)[0]

        # available_labels()[0] is the LIME-internal index for the top class
        lime_label = explanation.available_labels()[0]
        predicted_class = class_names[int(lime_label)]  # consistent with LIME's top label
        exp_list = explanation.as_list(label=lime_label)

        return {
            "text": payload.text,
            "predicted_class": predicted_class,
            "class_names": class_names,
            "probabilities": {
                str(cls): round(float(p), 4)
                for cls, p in zip(model.classes_, probs)
            },
            "explanation": [
                {"word": word, "weight": round(float(weight), 4)}
                for word, weight in exp_list
            ],
            "num_features": payload.num_features,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})
