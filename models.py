"""
Custom multiclass classifiers.

Classes
-------
NaiveBayes - An impleentation of Naive Bayes classification model
NN - Custom Tensorflow neural network classifier
"""

from typing import Iterable

import numpy as np


class NN:
    pass


class NaiveBayes:
    """An implementation of Naive Bayes algorithm for data classification."""

    def __init__(self) -> None:
        self.theta = np.array([])

    def fit(self, Xin: Iterable, yin: Iterable, verbose: bool = True) -> None:
        """Fit data to the model."""
        X = np.array(Xin)
        y = np.array(yin)
        class_list = np.unique(y)
        num_features = X.shape[0]
        num_classes = class_list.shape[0]
        if verbose:
            print(num_features)
            print(num_classes)
        # X_split_by_class = np.empty((num_classes, num_features, 3))
        for id, class_ in enumerate(class_list):
            # X_split_by_class[id] = X[y==class_]
            self.theta[id] = np.vstack(
                (
                    np.mean(X[y == class_], axis=0),
                    np.std(X[y == class_], axis=0),
                    X[y == class_].shape[1],
                )
            ).T
        if verbose:
            print(self.theta[:5])

    def predict(self, Xin: Iterable) -> np.ndarray:
        """Predict outputs."""
        probs = self.compute_probs(Xin)
        print(probs)
        return np.argmax(probs, axis=0)

    def compute_probs(self, Xin: Iterable) -> np.ndarray:
        """Compute the probabilities of output classes."""
        X = np.array(Xin)
        num_classes = self.theta.shape[0]
        num_training_data = np.sum(self.theta[:, 0, 2])
        probs = np.empty((X.shape[1], num_classes))
        for k in range(X.shape[1]):
            prob_dist = self.log_normal(
                np.tile(X[k], (num_classes, 1)),
                self.theta[:, :, 0],
                self.theta[:, :, 1],
            )
            probs[k] = (
                self.theta[:, 0, 2]
                / num_training_data
                * np.multiply.accumulate(prob_dist, axis=1)
            )
        return probs

    @staticmethod
    def log_normal(
        Xin: Iterable, mean: Iterable[float], std: Iterable[float]
    ) -> np.ndarray:
        return (
            -np.log(np.sqrt(2 * np.pi) * np.array(std))
            - 1
            / 2
            / np.array(std) ** 2
            * (np.array(Xin) - np.array(mean)) ** 2
        )
