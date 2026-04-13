# ============================================================
# lab7_experiments.py
# Lab 7: Text Classification — Embeddings × Classifiers
#
# Dataset  : Conf_Text_Labels.xlsx  ("Conf Data" sheet)
# Embeddings: TF-IDF | BERT (bert-base-uncased) | RoBERTa (roberta-base)
# Classifiers: Logistic Regression, KNN, SVM Linear, SVM RBF,
#              Decision Tree, Random Forest, Naive Bayes,
#              Gradient Boosting, MLP, AdaBoost
#
# Output CSVs:
#   embedding_classifier_summary_all_listed.csv  — metrics per combination
#   embedding_classifier_predictions.csv         — test-set predictions
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

# Embedding
from sklearn.feature_extraction.text import TfidfVectorizer

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


# ============================================================
# STEP 1 — Load and clean the dataset
# ============================================================

def load_data(filepath="Conf_Text_Labels.xlsx"):
    """
    Read the Excel file, drop rows with missing text or label,
    and encode the class labels as integers (0-based).

    Returns a cleaned DataFrame with columns:
        Text  — raw utterance text
        Label — integer class label
    """
    df = pd.read_excel(filepath, sheet_name="Conf Data")
    df = df[["Text", "Conf Label"]].dropna()
    df = df[df["Text"].astype(str).str.strip() != ""].copy()
    df["Text"] = df["Text"].astype(str).str.strip()

    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Conf Label"])

    print(f"[Data] Loaded {len(df)} samples | {df['Label'].nunique()} classes")
    return df


# ============================================================
# STEP 2 — TF-IDF embeddings
# ============================================================

def get_tfidf_embeddings(X_train_text, X_test_text, max_features=5000):
    """
    Fit a TF-IDF vectorizer on training text and transform both splits.

    Returns:
        X_train (ndarray), X_test (ndarray)
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text).toarray()
    X_test  = vectorizer.transform(X_test_text).toarray()
    return X_train, X_test


# ============================================================
# STEP 3 — Transformer embeddings (BERT or RoBERTa)
# ============================================================

def get_transformer_embeddings(texts, model_name, batch_size=32):
    """
    Generate fixed-size sentence embeddings by extracting the [CLS]
    token from the last hidden state of a HuggingFace model.

    Args:
        texts      : list / array of raw text strings
        model_name : HuggingFace model identifier
                     e.g. "bert-base-uncased" or "roberta-base"
        batch_size : number of sentences per forward pass

    Returns:
        embeddings (ndarray) of shape (len(texts), hidden_size)
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"  Loading tokenizer & model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  Using device: {device}")

    all_embeddings = []
    total = len(texts)

    for start in range(0, total, batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model(**encoded)

        # CLS token = first token of last hidden state
        cls_vec = output.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_vec)

        done = min(start + batch_size, total)
        print(f"  {model_name}: {done}/{total} samples embedded", end="\r")

    print()  # newline after progress output
    return np.vstack(all_embeddings)


# ============================================================
# STEP 4 — Classifier definitions
# ============================================================

def get_classifiers():
    """
    Return an ordered dict of  classifier_name -> sklearn estimator.
    All classifiers use random_state=42 where applicable for reproducibility.
    GaussianNB is used for Naive Bayes so it works with dense (possibly
    negative) embeddings produced by BERT / RoBERTa.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000,
                                                   random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
        "SVM Linear":          SVC(kernel="linear", random_state=42),
        "SVM RBF":             SVC(kernel="rbf",    random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100,
                                                      random_state=42),
        "Naive Bayes":         GaussianNB(),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100,
                                                           random_state=42),
        "MLP":                 MLPClassifier(hidden_layer_sizes=(128,),
                                             max_iter=300,
                                             random_state=42),
        "AdaBoost":            AdaBoostClassifier(n_estimators=100,
                                                  random_state=42),
    }


# ============================================================
# STEP 5 — Train one classifier and compute metrics
# ============================================================

def evaluate_classifier(clf, X_train, X_test, y_train, y_test):
    """
    Fit the classifier, predict on both splits, and return a dict of
    performance metrics plus the test-set predictions.

    Metrics returned (all weighted-average for multi-class):
        Train Accuracy, Test Accuracy, Precision, Recall, F1 Score
    """
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred  = clf.predict(X_test)

    metrics = {
        "Train Accuracy": round(accuracy_score(y_train, y_train_pred), 4),
        "Test Accuracy":  round(accuracy_score(y_test,  y_test_pred),  4),
        "Precision":      round(precision_score(y_test, y_test_pred,
                                               average="weighted",
                                               zero_division=0),        4),
        "Recall":         round(recall_score(y_test,    y_test_pred,
                                             average="weighted",
                                             zero_division=0),          4),
        "F1 Score":       round(f1_score(y_test,        y_test_pred,
                                         average="weighted",
                                         zero_division=0),              4),
    }
    return metrics, y_test_pred


# ============================================================
# STEP 6 — Master experiment runner
# ============================================================

def run_experiments(filepath="Conf_Text_Labels.xlsx",
                    test_size=0.2,
                    random_state=42):
    """
    Run every Embedding × Classifier combination and collect results.

    Returns
    -------
    dataset        : cleaned DataFrame (Text, Label)
    results_df     : one row per combination with all 5 metrics
    predictions_df : test-set true label + predicted label per combination
    top_results    : top 10 rows sorted by F1 Score (descending)
    best_by_emb    : best F1 combination for each embedding type
    best_result    : dict of the single best combination overall
    """

    # ---------- 1. Load data ----------
    dataset = load_data(filepath)
    X_text  = dataset["Text"].values
    y       = dataset["Label"].values

    # ---------- 2. Stratified train/test split ----------
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"[Split] Train={len(y_train)}  Test={len(y_test)}")

    # ---------- 3. Compute embeddings for all three methods ----------
    print("\n[Embedding 1/3] TF-IDF")
    tfidf_train, tfidf_test = get_tfidf_embeddings(X_train_text, X_test_text)

    print("\n[Embedding 2/3] BERT")
    all_texts   = np.concatenate([X_train_text, X_test_text])
    bert_all    = get_transformer_embeddings(all_texts, "bert-base-uncased")
    bert_train  = bert_all[: len(X_train_text)]
    bert_test   = bert_all[len(X_train_text) :]

    print("\n[Embedding 3/3] RoBERTa")
    roberta_all   = get_transformer_embeddings(all_texts, "roberta-base")
    roberta_train = roberta_all[: len(X_train_text)]
    roberta_test  = roberta_all[len(X_train_text) :]

    # ---------- 4. Map embedding names to feature matrices ----------
    embeddings = {
        "TF-IDF":  (tfidf_train,   tfidf_test),
        "BERT":    (bert_train,    bert_test),
        "RoBERTa": (roberta_train, roberta_test),
    }

    # ---------- 5. Run all combinations ----------
    classifiers = get_classifiers()
    rows        = []                         # one dict per combination
    preds_dict  = {"True Label": y_test}     # columns for predictions CSV

    for emb_name, (X_tr, X_te) in embeddings.items():
        print(f"\n{'='*55}")
        print(f"  Embedding: {emb_name}")
        print(f"{'='*55}")

        for clf_name, clf in classifiers.items():
            print(f"  ▸ {clf_name:<22}", end=" ")
            try:
                metrics, y_pred = evaluate_classifier(
                    clf, X_tr, X_te, y_train, y_test
                )
                print(f"  F1={metrics['F1 Score']:.4f}  "
                      f"Acc(test)={metrics['Test Accuracy']:.4f}")

                rows.append({"Embedding": emb_name,
                             "Classifier": clf_name,
                             **metrics})
                preds_dict[f"{emb_name}_{clf_name}"] = y_pred

            except Exception as exc:
                print(f"  FAILED — {exc}")
                rows.append({
                    "Embedding":      emb_name,
                    "Classifier":     clf_name,
                    "Train Accuracy": None,
                    "Test Accuracy":  None,
                    "Precision":      None,
                    "Recall":         None,
                    "F1 Score":       None,
                })

    # ---------- 6. Assemble output DataFrames ----------
    results_df     = pd.DataFrame(rows)
    predictions_df = pd.DataFrame(preds_dict)

    top_results = (results_df
                   .dropna(subset=["F1 Score"])
                   .sort_values("F1 Score", ascending=False)
                   .head(10))

    best_by_emb = (results_df
                   .dropna(subset=["F1 Score"])
                   .loc[results_df.dropna(subset=["F1 Score"])
                        .groupby("Embedding")["F1 Score"].idxmax()])

    best_idx    = results_df["F1 Score"].idxmax()
    best_result = results_df.loc[best_idx].to_dict()

    return dataset, results_df, predictions_df, top_results, best_by_emb, best_result


# ============================================================
# Entry point — run directly with: python lab7_experiments.py
# ============================================================

if __name__ == "__main__":
    (dataset,
     results_df,
     predictions_df,
     top_results,
     best_by_emb,
     best_result) = run_experiments()

    # Print summary
    print(f"\n{'='*55}")
    print(f"  Dataset size : {len(dataset)}")
    print(f"{'='*55}")

    print("\nTop 10 combinations by F1 Score:")
    print(top_results.to_string(index=False))

    print("\nBest model per embedding type:")
    print(best_by_emb[["Embedding","Classifier","F1 Score",
                        "Test Accuracy"]].to_string(index=False))

    print("\nBest overall combination:")
    for k, v in best_result.items():
        print(f"  {k}: {v}")

    # Save CSVs
    results_df.to_csv("embedding_classifier_summary_all_listed.csv", index=False)
    predictions_df.to_csv("embedding_classifier_predictions.csv",    index=False)

    print("\n[Saved] embedding_classifier_summary_all_listed.csv")
    print("[Saved] embedding_classifier_predictions.csv")
