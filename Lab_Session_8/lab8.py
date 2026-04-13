# IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from numpy.linalg import pinv


# A1: BASIC MODULES
def summation(x, w, b):
    return np.dot(x, w) + b

def step(x): return 1 if x >= 0 else 0
def bipolar_step(x): return 1 if x >= 0 else -1
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def relu(x): return max(0, x)
def leaky_relu(x): return x if x > 0 else 0.01 * x

def comparator(y, y_pred):
    return y - y_pred

# PERCEPTRON TRAINING
def train_perceptron(X, y, w, b, lr, activation, max_epochs=1000):
    errors = []
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(X)):
            net = summation(X[i], w, b)
            y_pred = activation(net)
            e = comparator(y[i], y_pred)

            w = w + lr * e * X[i]
            b = b + lr * e
            total_error += e**2

        errors.append(total_error)
        if total_error <= 0.002:
            break

    return w, b, errors, epoch+1

# DATA
X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0,0,0,1])

X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0,1,1,0])

# A2
def A2():
    w = np.array([0.2, -0.75])
    b = 10
    return train_perceptron(X_and, y_and, w, b, 0.05, step)

# A3
def A3():
    acts = {"Bipolar": bipolar_step, "Sigmoid": sigmoid, "ReLU": relu}
    results = {}
    for name, func in acts.items():
        w = np.array([0.2, -0.75])
        b = 10
        _, _, _, epochs = train_perceptron(X_and, y_and, w, b, 0.05, func)
        results[name] = epochs
    return results

# A4
def A4():
    rates = np.arange(0.1, 1.1, 0.1)
    epochs = []
    for lr in rates:
        w = np.array([0.2, -0.75])
        b = 10
        _, _, _, ep = train_perceptron(X_and, y_and, w, b, lr, step)
        epochs.append(ep)
    return rates, epochs

# A5
def A5():
    w = np.array([0.2, -0.75])
    b = 10
    return train_perceptron(X_xor, y_xor, w, b, 0.05, step)

# A6
def A6():
    X = np.array([
        [20,6,2,386],
        [16,3,6,289],
        [27,6,2,393],
        [19,1,2,110],
        [24,4,2,280],
        [22,1,5,167],
        [15,4,2,271],
        [18,4,2,274],
        [21,1,4,148],
        [16,2,4,198]
    ])
    y = np.array([1,1,1,0,1,0,1,1,0,0])

    w = np.random.rand(4)
    b = np.random.rand()

    return train_perceptron(X, y, w, b, 0.01, sigmoid)

# A7
def A7():
    X = np.hstack((X_and, np.ones((4,1))))
    y = y_and.reshape(-1,1)
    return pinv(X).dot(y)

# A8
def A8():
    np.random.seed(0)
    X = X_and
    y = y_and.reshape(-1,1)

    W1 = np.random.rand(2,2)
    W2 = np.random.rand(2,1)

    lr = 0.05
    errors = []

    for epoch in range(1000):
        h = sigmoid(np.dot(X, W1))
        o = sigmoid(np.dot(h, W2))

        e = y - o
        errors.append(np.sum(e**2))

        if errors[-1] <= 0.002:
            break

        d_o = e * (o*(1-o))
        d_h = d_o.dot(W2.T) * (h*(1-h))

        W2 += lr * h.T.dot(d_o)
        W1 += lr * X.T.dot(d_h)

    return errors, epoch+1

# A9
def A9():
    np.random.seed(0)
    X = X_xor
    y = y_xor.reshape(-1,1)

    W1 = np.random.rand(2,2)
    W2 = np.random.rand(2,1)

    lr = 0.05

    for epoch in range(1000):
        h = sigmoid(np.dot(X, W1))
        o = sigmoid(np.dot(h, W2))

        e = y - o

        d_o = e * (o*(1-o))
        d_h = d_o.dot(W2.T) * (h*(1-h))

        W2 += lr * h.T.dot(d_o)
        W1 += lr * X.T.dot(d_h)

    return epoch+1

# A10
def A10():
    y_multi = np.array([[1,0],[1,0],[1,0],[0,1]])

    w = np.random.rand(2,2)
    b = np.random.rand(2)

    lr = 0.05

    for epoch in range(500):
        for i in range(len(X_and)):
            net = np.dot(X_and[i], w) + b
            y_pred = np.array([step(n) for n in net])

            e = y_multi[i] - y_pred

            w += lr * np.outer(X_and[i], e)
            b += lr * e

    return w, b

# A11
def A11():
    mlp_and = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
    mlp_and.fit(X_and, y_and)

    mlp_xor = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
    mlp_xor.fit(X_xor, y_xor)

    return mlp_and, mlp_xor

# A12 (YOUR DATASET)
def A12():
    df = pd.read_excel('Conf_Text_Labels.xlsx')

    # CLEANING
    df = df.dropna(subset=['Text', 'Conf Label'])
    df['Conf Label'] = pd.to_numeric(df['Conf Label'], errors='coerce')
    df = df.dropna(subset=['Conf Label'])

    X_text = df['Text'].astype(str)
    y = df['Conf Label'].astype(int).values

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X_text).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000)
    mlp.fit(X_train, y_train)

    return mlp.score(X_test, y_test)

# MAIN
if __name__ == "__main__":

    w,b,err,ep = A2()
    print("A2 Epochs:", ep)
    plt.plot(err)
    plt.title("A2 Error")
    plt.show()

    print("A3:", A3())

    r,e = A4()
    plt.plot(r,e)
    plt.title("A4")
    plt.show()

    print("A5 XOR Epochs:", A5()[3])

    print("A6 Epochs:", A6()[3])

    print("A7 Weights:", A7())

    err,ep = A8()
    print("A8 Epochs:", ep)

    print("A9 Epochs:", A9())

    print("A10:", A10())

    print("A11 Done")

    print("A12 Accuracy:", A12())
