# LAB 10 - Speaker Confidence Assessment (TEXT DATASET)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SequentialFeatureSelector

try:
    import lime
    import lime.lime_text
except:
    lime = None

try:
    import shap
except:
    shap = None

def load_dataset(file_path):
    df = pd.read_excel(file_path)

    df = df[['Text', 'Conf Label']]
    df.dropna(inplace=True)

    df['Text'] = df['Text'].astype(str)
    df['Conf Label'] = df['Conf Label'].astype(int)

    return df

# FUNCTION 2 - TFIDF FEATURES
def create_features(text_data):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=300,
        lowercase=True
    )

    X = vectorizer.fit_transform(text_data).toarray()
    feature_names = vectorizer.get_feature_names_out()

    return X, feature_names, vectorizer


# FUNCTION 3 - SPLIT DATA
def split_data(X, y):
    return train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


# FUNCTION 4 - MODELS
def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    }

    return models


# FUNCTION 5 - EVALUATE MODELS
def evaluate_models(models, X_train, X_test, y_train, y_test):

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4)
        })

    return pd.DataFrame(results)

# FUNCTION 6 - HEATMAP
def plot_heatmap(X, feature_names):

    df_features = pd.DataFrame(X, columns=feature_names)

    selected_cols = df_features.var().sort_values(
        ascending=False
    ).head(25).index

    corr = df_features[selected_cols].corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap='coolwarm')
    plt.title("A1 Correlation Heatmap")
    plt.show()


# FUNCTION 7 - PCA
def apply_pca(X_train, X_test, variance):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=variance)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    retained = X_train_pca.shape[1]

    return X_train_pca, X_test_pca, retained


# FUNCTION 8 - SEQUENTIAL FEATURE SELECTION
def apply_sfs(X_train, X_test, y_train):

    base_model = LogisticRegression(max_iter=3000)

    sfs = SequentialFeatureSelector(
        estimator=base_model,
        n_features_to_select=20,
        direction='forward',
        scoring='accuracy',
        cv=3,
        n_jobs=1
    )

    sfs.fit(X_train, y_train)

    X_train_sfs = sfs.transform(X_train)
    X_test_sfs = sfs.transform(X_test)

    return X_train_sfs, X_test_sfs

# FUNCTION 9 - LIME
def run_lime(model, vectorizer, sample_text):

    if lime is None:
        return None

    class_names = ['1', '2', '3', '4', '5']

    explainer = lime.lime_text.LimeTextExplainer(
        class_names=class_names
    )

    def predictor(texts):
        X = vectorizer.transform(texts)
        return model.predict_proba(X)

    explanation = explainer.explain_instance(
        sample_text,
        predictor,
        num_features=10
    )

    return explanation

# FUNCTION 10 - SHAP
def run_shap(model, X_train, X_test):

    if shap is None:
        return None

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test[:20])

    return shap_values


# MAIN PROGRAM
if __name__ == "__main__":

    file_path = r"C:\Users\hema3\Downloads\machine learning\Conf_Text_Labels.xlsx"

    df = load_dataset(file_path)

    print("Dataset Shape:", df.shape)
    print(df.head())

    X, feature_names, vectorizer = create_features(df['Text'])

    y = df['Conf Label']

    print("\nTotal Features:", X.shape[1]
          
    X_train, X_test, y_train, y_test = split_data(X, y)


    print("\nA1: Correlation Heatmap")
    plot_heatmap(X, feature_names)


    print("\nOriginal Feature Results")

    models = get_models()

    original_results = evaluate_models(
        models,
        X_train,
        X_test,
        y_train,
        y_test
    )

    print(original_results)

    print("\nA2: PCA with 99% Variance")

    X_train_pca99, X_test_pca99, n99 = apply_pca(
        X_train,
        X_test,
        0.99
    )

    print("Features Retained:", n99)

    models = get_models()

    pca99_results = evaluate_models(
        models,
        X_train_pca99,
        X_test_pca99,
        y_train,
        y_test
    )

    print(pca99_results)

    print("\nA3: PCA with 95% Variance")

    X_train_pca95, X_test_pca95, n95 = apply_pca(
        X_train,
        X_test,
        0.95
    )

    print("Features Retained:", n95)

    models = get_models()

    pca95_results = evaluate_models(
        models,
        X_train_pca95,
        X_test_pca95,
        y_train,
        y_test
    )

    print(pca95_results)


    print("\nA4: Sequential Feature Selection")

    X_train_sfs, X_test_sfs = apply_sfs(
        X_train,
        X_test,
        y_train
    )

    print("Selected Features:", X_train_sfs.shape[1])

    models = get_models()

    sfs_results = evaluate_models(
        models,
        X_train_sfs,
        X_test_sfs,
        y_train,
        y_test
    )

    print(sfs_results)


    print("\nA5: LIME + SHAP")

    final_model = LogisticRegression(max_iter=3000)
    final_model.fit(X_train, y_train)


    if lime is not None:

        sample_text = df['Text'].iloc[0]

        lime_exp = run_lime(
            final_model,
            vectorizer,
            sample_text
        )

        print("\nLIME Explanation:")
        print(lime_exp.as_list())

    else:
        print("LIME not installed.")

    if shap is not None:

        shap_values = run_shap(
            final_model,
            X_train,
            X_test
        )

        print("\nSHAP Summary Plot")

        shap.summary_plot(
            shap_values.values,
            X_test[:20],
            show=False
        )

        plt.show()

    else:
        print("SHAP not installed.")


    print("\n=========== FINAL COMPARISON ===========")

    print("\nOriginal Features")
    print(original_results)

    print("\nPCA 99%")
    print(pca99_results)

    print("\nPCA 95%")
    print(pca95_results)

    print("\nSequential Feature Selection")
    print(sfs_results)
