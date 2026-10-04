"""Feed-forward ANN (multi-layer perceptron) for streamflow regression.

Built with scikit-learn's MLPRegressor, trained one epoch at a time so we can
record training/validation loss per epoch and keep the best-validation weights
(early stopping on a chronological validation set - NOT a random split).
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor


def train_ann(Xtr, ytr, Xva, yva, hidden=(12,), activation="logistic", lr=0.005,
              max_epochs=200, patience=25, batch_size=32, seed=0):
    """Train an ANN; return (best_model, history dict).

    hidden      tuple of hidden-layer sizes, e.g. (12,) or (16, 8)
    activation  'logistic' (sigmoid, as in the reference paper) or 'relu'
    The output layer is linear (MLPRegressor default), as needed for regression.
    """
    model = MLPRegressor(hidden_layer_sizes=hidden, activation=activation, solver="adam",
                         learning_rate_init=lr, batch_size=batch_size, alpha=0.0,
                         shuffle=True, random_state=seed)
    hist = {"train_mse": [], "val_mse": []}
    best_val, best_model, best_epoch, wait = np.inf, None, 0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for epoch in range(1, max_epochs + 1):
            model.partial_fit(Xtr, ytr)  # one pass over the training data
            tr = float(np.mean((model.predict(Xtr) - ytr) ** 2))
            va = float(np.mean((model.predict(Xva) - yva) ** 2))
            hist["train_mse"].append(tr)
            hist["val_mse"].append(va)
            if va < best_val - 1e-9:
                best_val, best_model, best_epoch, wait = va, copy.deepcopy(model), epoch, 0
            else:
                wait += 1
                if wait >= patience:
                    break
    hist["best_epoch"] = best_epoch
    hist["best_val_mse"] = best_val
    return best_model, hist
