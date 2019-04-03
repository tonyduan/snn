import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import scipy.stats
import logging

from argparse import ArgumentParser
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score
from snn.models import SNN


def to_tensors(iterable):
    return [torch.tensor(x, dtype=torch.float) for x in iterable]

def get_concrete_data():
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    df = pd.read_excel(URL)
    X = df.iloc[:, :-1].to_numpy()
    Y = df.iloc[:,-1].to_numpy()
    return X, Y

def get_biodeg_data():
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv"
    df = pd.read_csv(URL, delimiter=";")
    X = df.iloc[:, :-1].to_numpy().astype("float")
    Y = df.iloc[:,-1].to_numpy()
    one_hot_enc = OneHotEncoder(sparse=False)
    Y = one_hot_enc.fit_transform(Y[:,np.newaxis])[:,1]
    return X, Y

def plot_qq(diffs):
    (osm, osr), (slope, intcpt, r) = sp.stats.probplot(diffs, dist="norm")
    plt.figure(figsize=(4, 4))
    plt.scatter(osm, osr, color="black", marker=".")
    plt.xlim((-4, 4))
    plt.ylim((-4, 4))
    plt.xlabel("Theoretical quantiles")
    plt.ylabel("Empirical quantiles")
    plt.plot((-4, 4),  (-4 * slope + intcpt, 4 * slope + intcpt), color="blue")
    plt.plot((-4, 4),  (-4, 4), "--", color="red", alpha=0.5)
    plt.show()

def eval_reg(Y_test, Y_pred):
    print(f"R2: {r2_score(Y_test, Y_pred):.2f}")
    diffs = Y_test - Y_pred
    plot_qq(diffs / np.std(diffs))

def eval_cls(Y_test, Y_pred):
    print(f"Prevalence: {Y_test.mean():.2f}")
    print(f"Accuracy: {(Y_test == Y_pred.round()).mean():.2f}")
    print(f"ROC-AUC: {roc_auc_score(Y_test, Y_pred):.2f}")


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--lr", default=0.01, type=float)
    argparser.add_argument("--iterations", default=750, type=int)
    argparser.add_argument("--reg", action="store_true")
    argparser.add_argument("--cls", dest="reg", action="store_false")
    argparser.set_defaults(reg=False)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    if args.reg:
        X, Y = get_concrete_data()
        loss_fn = nn.MSELoss(reduction="none")
    else:
        X, Y = get_biodeg_data()
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    print(f"Dataset shape: {X.shape}")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    split = train_test_split(X, Y, test_size=0.2, random_state=123)
    X_train, X_test, Y_train, Y_test = to_tensors(split)

    snn = SNN(X.shape[1], 1, 32, n_layers=8, dropout_prob=0.0)
    optimizer = optim.Adamax(snn.parameters(), lr=args.lr)

    for i in range(args.iterations):
        optimizer.zero_grad()
        loss = loss_fn(snn.forward(X_train).squeeze(), Y_train).mean()
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            logger.info(f"Iter: {i}\tLoss: {loss.data:.2f}")

    snn.train(mode=False)
    Y_pred = snn.forward(X_test).squeeze().data

    if args.reg:
        print("== SNN")
        eval_reg(Y_test.numpy(), Y_pred.numpy())
        print("== LinReg")
        linreg = LinearRegression()
        linreg.fit(X_train.numpy(), Y_train.numpy())
        Y_pred = linreg.predict(X_test.numpy())
        eval_reg(Y_test.numpy(), Y_pred)
    else:
        Y_pred = torch.sigmoid(snn.forward(X_test)).squeeze().data
        print("== SNN")
        eval_cls(Y_test.numpy(), Y_pred.numpy())
        print("== LogReg")
        logreg = LogisticRegression()
        logreg.fit(X_train.numpy(), Y_train.numpy())
        Y_pred = logreg.predict_proba(X_test.numpy())[:,1]
        eval_cls(Y_test.numpy(), Y_pred)

    layer_activations = snn.track_layer_activations(X_test)

    plt.style.use('bmh')
    plt.figure(figsize=(8, 4))
    for i, a in enumerate(layer_activations):
        density = sp.stats.gaussian_kde(a, "silverman")
        x_axis = np.linspace(-4, 4, 200)
        plt.plot(x_axis, density(x_axis), label=f"Layer {i}")
        plt.fill_between(x_axis, 0, density(x_axis), alpha=0.2)

    plt.xlabel("Neuron activations")
    plt.legend()
    plt.show()
