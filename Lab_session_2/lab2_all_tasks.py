import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

def load_purchase_data(file_path):
    return pd.read_excel(file_path, sheet_name=0)

def load_irctc_data(file_path):
    return pd.read_excel(file_path, sheet_name=1)

def load_thyroid_data(file_path):
    return pd.read_excel(file_path, sheet_name=2)


def create_X_y(data):
    X = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = data["Payment (Rs)"].values
    return X, y

def matrix_rank(X):
    return np.linalg.matrix_rank(X)

def product_cost_pinv(X, y):
    return np.linalg.pinv(X).dot(y)


def label_customers(payments):
    labels = []
    for p in payments:
        if p > 200:
            labels.append("RICH")
        else:
            labels.append("POOR")
    return labels


def mean_numpy(values):
    return np.mean(values)

def var_numpy(values):
    return np.var(values)

def mean_manual(values):
    total = 0
    for v in values:
        total += v
    return total / len(values)

def var_manual(values):
    m = mean_manual(values)
    total = 0
    for v in values:
        total += (v - m) ** 2
    return total / len(values)

def avg_time(func, values):
    t = 0
    for _ in range(10):
        start = time.time()
        func(values)
        t += time.time() - start
    return t / 10

def wednesday_mean(data):
    wed = data[data["Day"].str.strip().str.lower() == "wednesday"]
    if len(wed) == 0:
        return 0
    return mean_manual(wed["Price"].values)

def april_mean(data):
    apr = data[data["Month"].str.strip().str.lower() == "apr"]
    if len(apr) == 0:
        return 0
    return mean_manual(apr["Price"].values)

def loss_probability(chg):
    loss = list(filter(lambda x: x < 0, chg))
    return len(loss) / len(chg)

def profit_on_wednesday(data):
    wed = data[data["Day"].str.strip().str.lower() == "wednesday"]
    if len(wed) == 0:
        return 0
    profit = wed[wed["Chg%"] > 0]
    return len(profit) / len(wed)

def scatter_plot(data):
    plt.scatter(data["Day"], data["Chg%"])
    plt.xlabel("Day")
    plt.ylabel("Change %")
    plt.show()


def show_basic_info(data):
    print(data.dtypes)

def missing_values(data):
    return data.isna().sum()

def numeric_stats(data):
    return data.describe()

def detect_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).any()



def binary_columns(data):
    cols = []
    for c in data.columns:
        vals = data[c].dropna().unique()
        if set(vals).issubset({0, 1}):
            cols.append(c)
    return cols

def frequencies(v1, v2):
    f11 = f10 = f01 = f00 = 0
    for i in range(len(v1)):
        if v1[i] == 1 and v2[i] == 1:
            f11 += 1
        elif v1[i] == 1 and v2[i] == 0:
            f10 += 1
        elif v1[i] == 0 and v2[i] == 1:
            f01 += 1
        else:
            f00 += 1
    return f11, f10, f01, f00

def jaccard(f11, f10, f01):
    d = f11 + f10 + f01
    return 0 if d == 0 else f11 / d

def smc(f11, f10, f01, f00):
    t = f11 + f10 + f01 + f00
    return 0 if t == 0 else (f11 + f00) / t


def cosine_similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0
    return np.dot(v1, v2) / (n1 * n2)


def heatmaps(data):
    data20 = data.iloc[:20]
    bin_cols = binary_columns(data20)
    num_data = data20.select_dtypes(include=[np.number])

    n = len(data20)
    jc = np.zeros((n, n))
    sm = np.zeros((n, n))
    cs = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if len(bin_cols) > 0:
                f11, f10, f01, f00 = frequencies(
                    data20.loc[i, bin_cols].values,
                    data20.loc[j, bin_cols].values
                )
                jc[i][j] = jaccard(f11, f10, f01)
                sm[i][j] = smc(f11, f10, f01, f00)
            cs[i][j] = cosine_similarity(num_data.iloc[i].values,
                                         num_data.iloc[j].values)

    sns.heatmap(jc); plt.title("JC"); plt.show()
    sns.heatmap(sm); plt.title("SMC"); plt.show()
    sns.heatmap(cs); plt.title("COS"); plt.show()


def impute_data(data):
    for c in data.columns:
        if data[c].isna().sum() == 0:
            continue
        if data[c].dtype != "object":
            if detect_outliers_iqr(data[c].dropna()):
                data[c].fillna(data[c].median(), inplace=True)
            else:
                data[c].fillna(data[c].mean(), inplace=True)
        else:
            data[c].fillna(data[c].mode()[0], inplace=True)
    return data


def min_max_normalize(data):
    d = data.copy()
    for c in d.select_dtypes(include=[np.number]).columns:
        mn, mx = d[c].min(), d[c].max()
        d[c] = 0 if mx - mn == 0 else (d[c] - mn) / (mx - mn)
    return d


def main():
    file_path = "/home/parameshwar/Machine_Learning-22AIE213/Lab_session_2/Lab Session Data.xlsx"

    # A1
    purchase = load_purchase_data(file_path)
    X, y = create_X_y(purchase)
    print("Rank:", matrix_rank(X))
    print("Costs:", product_cost_pinv(X, y))

    # A2
    purchase["Status"] = label_customers(purchase["Payment (Rs)"].values)
    print(purchase[["Payment (Rs)", "Status"]])

    # A3
    irctc = load_irctc_data(file_path)
    print("Mean:", mean_numpy(irctc["Price"].values))
    print("Variance:", var_numpy(irctc["Price"].values))
    scatter_plot(irctc)

    # A4–A9
    thyroid = load_thyroid_data(file_path)
    thyroid = impute_data(thyroid)
    norm_thyroid = min_max_normalize(thyroid)
    heatmaps(norm_thyroid)

main()
