"""
Custom multiclass classifiers.

Classes
-------
GaussianNaiveBayes - An impleentation of Naive Bayes classification model
NN - Custom Tensorflow neural network classifier
"""

import numpy as np
from numpy.typing import ArrayLike


class NN:
    pass


class GaussianNaiveBayes:
    """An implementation of Naive Bayes classifier for data classification."""

    def __init__(self) -> None:
        self.theta = np.array([])

    def fit(
        self, Xin: ArrayLike, yin: ArrayLike, verbose: bool = True
    ) -> None:
        """Fit data to the model."""
        X = np.array(Xin)
        y = np.array(yin)
        class_list = np.unique(y)
        num_features = X.shape[1]
        num_classes = class_list.shape[0]
        if verbose:
            print(f"Fitting data to {self}")
            print(f"Number of features: {num_features}")
            print(f"Number of classes: {num_classes}")
            print(f"Number of training rows: {X.shape[0]}")
        self.theta = np.empty((num_classes, num_features, 3))
        for id, class_ in enumerate(class_list):
            self.theta[id] = np.vstack(
                (
                    np.mean(X[y == class_], axis=0),
                    np.std(X[y == class_], axis=0),
                    np.full(
                        (1, num_features), X[y == class_].shape[0] / X.shape[0]
                    ),
                )
            ).T

    def predict(self, Xin: ArrayLike) -> np.ndarray:
        """Predict outputs."""
        posterior_probs = self.compute_probs(Xin)
        return np.argmax(posterior_probs, axis=1)

    def compute_probs(self, Xin: ArrayLike) -> np.ndarray:
        """Compute posterior probability."""
        X = np.array(Xin)
        num_classes = self.theta.shape[0]
        probs = np.empty((X.shape[0], num_classes))

        # compute probabilities for each feature row
        for k in range(X.shape[0]):
            prob_dist = self.normal(
                np.tile(X[k], (num_classes, 1)),
                self.theta[:, :, 0],
                self.theta[:, :, 1],
            )

            prior_probs = np.log(self.theta[:, 0, 2])
            v = np.add.reduce(
                np.log(prob_dist), axis=1, where=~np.isnan(prob_dist)
            )

            probs[k] = prior_probs + v
        return probs

    @staticmethod
    def normal(Xin: ArrayLike, mean: ArrayLike, std: ArrayLike) -> np.ndarray:
        """Evaluate normal distribution for each data point."""
        return (
            1
            / (np.sqrt(2 * np.pi) / np.array(std))
            * np.exp(
                -1
                / 2
                / np.array(std) ** 2
                * (np.array(Xin) - np.array(mean)) ** 2
            )
        )

    def score(self, Xin: ArrayLike, yin: ArrayLike) -> float:
        """Compute the accuracy score as a fraction."""
        X = np.array(Xin)
        if not X.any():
            return 0
        y = np.array(yin)
        y_pred = self.predict(X)
        return 1 / X.shape[0] * np.sum(y_pred == y)
