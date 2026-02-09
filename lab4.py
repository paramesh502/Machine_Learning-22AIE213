import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, mean_squared_error, r2_score

# A1
def evaluate_classifier(true_labels, predictions):
    matrix = confusion_matrix(true_labels, predictions)
    acc = np.sum(true_labels == predictions) / len(true_labels)
    prec = precision_score(true_labels, predictions)
    rec = recall_score(true_labels, predictions)
    f1_val = f1_score(true_labels, predictions)
    return matrix, acc, prec, rec, f1_val


# A2
def calculate_regression_stats(actual, predicted):
    mse_val = mean_squared_error(actual, predicted)
    rmse_val = np.sqrt(mse_val)
    mape_val = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2_val = r2_score(actual, predicted)
    return mse_val, rmse_val, mape_val, r2_val


# A3
def create_points(samples=20):
    pts = np.random.uniform(1, 10, (samples, 2))
    labels = (pts.sum(axis=1) >= 11).astype(int)
    return pts, labels


def visualize_points(points, labels):
    colors = np.where(labels == 0, 'blue', 'red')
    plt.figure(figsize=(5,5))
    plt.scatter(points[:, 0], points[:, 1], c=colors, edgecolor='k')
    plt.title("Random Training Points")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.show()


#A4 & A5
def make_mesh(step_size=0.1):
    a = np.arange(0, 10, step_size)
    b = np.arange(0, 10, step_size)
    grid_x, grid_y = np.meshgrid(a, b)
    coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    return grid_x, grid_y, coords


def display_knn_regions(train_X, train_y, neighbors):
    model = KNeighborsClassifier(n_neighbors=neighbors)
    model.fit(train_X, train_y)

    gx, gy, coords = make_mesh()
    labels = model.predict(coords)

    plt.figure(figsize=(6,6))
    plt.scatter(coords[:, 0], coords[:, 1], c=labels, alpha=0.2, cmap='coolwarm')
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, edgecolor='black', cmap='coolwarm')
    plt.title(f"kNN Boundary (k={neighbors})")
    plt.show()


# A7
def optimize_k(train_X, train_y):
    params = {"n_neighbors": np.arange(1, 21)}
    search = GridSearchCV(KNeighborsClassifier(), params, cv=5)
    search.fit(train_X, train_y)
    return search.best_params_["n_neighbors"], search.best_score_


data = pd.read_excel("Conf_Text_Labels.xlsx", sheet_name="Conf Data", engine="openpyxl")
data = pd.read_excel(file_path, sheet_name="Conf Data", engine="openpyxl")

X_vals = data.select_dtypes(include=[np.number]).iloc[:, :-1].values
y_vals = data.iloc[:, -1].values

unique_classes = np.unique(y_vals)[:2]
filter_mask = np.isin(y_vals, unique_classes)
X_vals, y_vals = X_vals[filter_mask], y_vals[filter_mask]

mapping = {unique_classes[0]: 0, unique_classes[1]: 1}
y_vals = np.vectorize(mapping.get)(y_vals)

Xtr, Xte, ytr, yte = train_test_split(X_vals, y_vals, test_size=0.3, random_state=42)

model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(Xtr, ytr)

train_out = evaluate_classifier(ytr, model_knn.predict(Xtr))
test_out = evaluate_classifier(yte, model_knn.predict(Xte))

print("Train Metrics:", train_out)
print("Test Metrics:", test_out)

# Synthetic visualization
pts, lbls = create_points()
visualize_points(pts, lbls)

for val in [1, 3, 7]:
    display_knn_regions(pts, lbls, val)

best_k_val, best_cv = optimize_k(Xtr, ytr)
print("Optimal k:", best_k_val)
print("Cross-val Accuracy:", best_cv)