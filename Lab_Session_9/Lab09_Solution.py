# Lab Session 09 - 22AIE213
# Stacking classifier, Pipeline and LIME explainer on our text dataset
# Dataset: Conf_Text_Labels.xlsx  (student text -> Confidence label 1..5)

import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# base learners (these are the ones we already built in the previous labs)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier

from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report,
                             confusion_matrix)

# --------- helper / reusable functions ---------

def load_dataset(path):
    """Read the excel file and return dataframe of text + label only."""
    df = pd.read_excel(path)
    # keep only what we need; drop rows that don't have a label
    df = df[['Text', 'Conf Label']].dropna()
    df = df.rename(columns={'Conf Label': 'label'})
    df['label'] = df['label'].astype(int)
    # very short / empty text rows are noise, remove them
    df = df[df['Text'].astype(str).str.strip().str.len() > 1].reset_index(drop=True)
    return df


def clean_text(t):
    """quick text cleaning - lowercase, remove weird chars, collapse spaces"""
    t = str(t).lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-zA-Z\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def make_base_learners():
    """list of already-implemented base classifiers (same ones from prev labs)"""
    base = [
        ('logreg', LogisticRegression(max_iter=1000, C=1.0)),
        ('dtree',  DecisionTreeClassifier(max_depth=15, random_state=42)),
        ('knn',    KNeighborsClassifier(n_neighbors=5)),
        ('nb',     MultinomialNB()),
        ('svc',    LinearSVC(C=1.0)),
        ('rf',     RandomForestClassifier(n_estimators=150, random_state=42))
    ]
    return base


def build_stacking(base_models, meta_model):
    """Build a StackingClassifier given the base learners + a final estimator."""
    clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1,
        passthrough=False
    )
    return clf


def evaluate_model(model, X_tr, y_tr, X_te, y_te):
    """fit model and return a dict of common metrics on the test set"""
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    res = {
        'accuracy':  accuracy_score(y_te, preds),
        'precision': precision_score(y_te, preds, average='weighted', zero_division=0),
        'recall':    recall_score(y_te, preds, average='weighted', zero_division=0),
        'f1':        f1_score(y_te, preds, average='weighted', zero_division=0),
    }
    return res, preds


def build_pipeline(final_estimator):
    """A2: TF-IDF vectorizer followed by a classifier, wrapped as a Pipeline."""
    pipe = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2,
                                  max_df=0.95, stop_words='english')),
        ('clf',   final_estimator)
    ])
    return pipe


def lime_explain(pipe, texts_to_explain, class_names, num_features=10):
    """
    A3: use LIME to explain a few predictions coming from our pipeline.
    Returns a list of (text, pred_label, explanation_as_list).
    """
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=class_names)
    out = []
    for t in texts_to_explain:
        pred_label = pipe.predict([t])[0]
        # LIME needs a callable producing class probabilities
        exp = explainer.explain_instance(
            t,
            pipe.predict_proba,
            num_features=num_features,
            labels=(list(pipe.classes_).index(pred_label),)
        )
        label_idx = list(pipe.classes_).index(pred_label)
        explanation = exp.as_list(label=label_idx)
        out.append((t, pred_label, explanation))
    return out


# ---------------- main program ----------------

if __name__ == "__main__":

    DATA_PATH = "/sessions/beautiful-great-dijkstra/mnt/Lab_Session_9/Conf_Text_Labels.xlsx"

    # 1) load + clean
    df = load_dataset(DATA_PATH)
    df['clean'] = df['Text'].apply(clean_text)
    df = df[df['clean'].str.len() > 1].reset_index(drop=True)
    print("Dataset size after cleaning:", df.shape)
    print("Class distribution:\n", df['label'].value_counts().sort_index())

    # 2) train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean'].values, df['label'].values,
        test_size=0.2, random_state=42, stratify=df['label'].values
    )

    # Vectorize once for the A1 stacking experiments (keep it separate from the pipeline)
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95, stop_words='english')
    Xtr_vec = vec.fit_transform(X_train)
    Xte_vec = vec.transform(X_test)

    # -------- A1: Stacking classifier with different metamodels --------
    print("\n================  A1 : Stacking Classifier  ================")
    base_learners = make_base_learners()

    meta_models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest':       RandomForestClassifier(n_estimators=200, random_state=42),
        'GradientBoosting':   GradientBoostingClassifier(random_state=42),
        'DecisionTree':       DecisionTreeClassifier(max_depth=10, random_state=42),
    }

    stacking_results = {}
    for name, mm in meta_models.items():
        print(f"\n>> training stacking with final_estimator = {name}")
        stk = build_stacking(base_learners, mm)
        metrics, _ = evaluate_model(stk, Xtr_vec, y_train, Xte_vec, y_test)
        stacking_results[name] = metrics
        print(f"   accuracy : {metrics['accuracy']:.4f}")
        print(f"   precision: {metrics['precision']:.4f}")
        print(f"   recall   : {metrics['recall']:.4f}")
        print(f"   f1       : {metrics['f1']:.4f}")

    # quick results table
    results_df = pd.DataFrame(stacking_results).T
    results_df = results_df.round(4)
    print("\n--- Stacking classifier comparison (A1) ---")
    print(results_df.to_string())

    # -------- A2: Pipeline of processing + classification --------
    print("\n================  A2 : Pipeline  ================")
    # we use the best meta model found above as the final estimator in the pipeline,
    # but the whole thing is wrapped so fit/predict happen end-to-end on raw text.
    best_meta_name = max(stacking_results, key=lambda k: stacking_results[k]['f1'])
    print("Best meta model from A1:", best_meta_name)

    # pipeline final estimator = stacking classifier (with best meta)
    stacked_for_pipe = build_stacking(make_base_learners(),
                                      meta_models[best_meta_name])
    pipe = build_pipeline(stacked_for_pipe)
    pipe.fit(X_train, y_train)
    pipe_preds = pipe.predict(X_test)
    pipe_metrics = {
        'accuracy':  accuracy_score(y_test, pipe_preds),
        'precision': precision_score(y_test, pipe_preds, average='weighted', zero_division=0),
        'recall':    recall_score(y_test, pipe_preds, average='weighted', zero_division=0),
        'f1':        f1_score(y_test, pipe_preds, average='weighted', zero_division=0),
    }
    print("Pipeline (TF-IDF -> Stacking) test metrics:")
    for k, v in pipe_metrics.items():
        print(f"   {k:<10}: {v:.4f}")

    print("\nClassification report for pipeline:")
    print(classification_report(y_test, pipe_preds, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pipe_preds))

    # -------- A3: LIME explainer on pipeline predictions --------
    # NOTE: LIME needs predict_proba. Our stacking uses LinearSVC as a base learner
    # which does not give probabilities, so for the LIME step we rebuild a lighter
    # pipeline whose final estimator supports predict_proba (LogReg works well).
    print("\n================  A3 : LIME explanations  ================")

    lime_pipe = build_pipeline(
        StackingClassifier(
            estimators=[
                ('logreg', LogisticRegression(max_iter=1000)),
                ('dtree',  DecisionTreeClassifier(max_depth=15, random_state=42)),
                ('nb',     MultinomialNB()),
                ('rf',     RandomForestClassifier(n_estimators=150, random_state=42))
            ],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5, n_jobs=-1
        )
    )
    lime_pipe.fit(X_train, y_train)

    # pick a few test samples to explain
    samples_idx = [0, 10, 25, 50, 100]
    sample_texts = [X_test[i] for i in samples_idx if i < len(X_test)]
    class_names = [str(c) for c in sorted(np.unique(y_train))]

    lime_results = lime_explain(lime_pipe, sample_texts, class_names, num_features=8)

    for i, (txt, pred, exp_list) in enumerate(lime_results):
        print(f"\n[Sample {i+1}]")
        print("Text     :", txt[:120], ("..." if len(txt) > 120 else ""))
        print("Predicted:", pred)
        print("Top words contributing to this prediction:")
        for w, score in exp_list:
            print(f"   {w:<25} weight = {score:+.4f}")
