"""
A training script based on CoverType dataset.
Usage: train_model.py <model_type: KNN|DecisionTree|Heuristic|NN>
        [-n <trained_model_name>]

Functions
---------
load_split_dataset: prepare dataset for training
train_model:        train classification model
test_model:         test model
save_model:         save model in a pickle format
"""

import json
import os
import sys
from typing import Any, Dict, List, Union

import joblib
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from models import GaussianNaiveBayes, create_model, optimize_model


def load_split_dataset(
    dataset_path: Union[os.PathLike, str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> List:
    print(f"Loading dataset {dataset_path}")
    data = np.loadtxt(dataset_path, delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def save_model(
    model: Any,
    save_path: Union[os.PathLike, str],
    model_name: str,
) -> None:
    print(f"Saving model {model_name} to {save_path}")
    if hasattr(model, "model"):
        model.model.save(os.path.join(save_path, model_name + ".h5"))
    else:
        filename = os.path.join(save_path, model_name + ".pkl")
        joblib.dump(
            model, filename, compress=3
        )  # use compression to avoid large file size


def test_model(
    model: Any,
    X_train: ArrayLike,
    X_test: ArrayLike,
    y_train: ArrayLike,
    y_test: ArrayLike,
    verbose: bool = True,
    save: bool = False,
) -> Dict:
    """Evaluate a classification model."""
    if verbose:
        print(f"Testing model {model}")

    results = {}
    results["train_score"] = model.score(X_train, y_train) * 100
    if verbose:
        print(f"Training accuracy: {results['train_score']:.2f}%")

    if isinstance(model, BaseEstimator):
        score = cross_val_score(
            model, X_test, y_test, scoring="accuracy", cv=10
        ).mean()
        y_pred = cross_val_predict(model, X_test, y_test, cv=10)
    else:
        score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)

    results["cross_val_accuracy"] = score * 100
    if verbose:
        print(
            f"Cross-validation accuracy: \
{results['cross_val_accuracy']:.2f}%"
        )

    results["conf_mat"] = confusion_matrix(y_test, y_pred).tolist()
    if verbose:
        print(f"Confusion matrix: {results['conf_mat']}")

    results["cl_report"] = classification_report(y_test, y_pred)
    if verbose:
        print(results["cl_report"])

    if save:
        if not os.path.exists("trained_models_reports"):
            os.makedirs("trained_models_reports")
        with open(
            os.path.join(
                "trained_models_reports", model_name + "_evaluation.json"
            ),
            "w",
        ) as f:
            json.dump(results, f)

    return results


def init_model(model_name: str) -> Any | None:
    """Initialize one of available models."""
    match model_name:
        case "NaiveBayes":
            model: Any = make_pipeline(MinMaxScaler(), GaussianNaiveBayes())
        case "DecisionTree":
            model = DecisionTreeClassifier(
                criterion="gini", min_samples_split=2
            )
        case "KNN":
            model = make_pipeline(
                MinMaxScaler(), KNeighborsClassifier(n_neighbors=5, n_jobs=2)
            )
        case "NN":
            model = KerasClassifier(build_fn=create_model)
        case _:
            return None
    return model


if __name__ == "__main__":
    # get name of the model to train from command arguments
    model_name: str = sys.argv[1] if len(sys.argv) > 1 else ""
    model_save_path = "trained_models"
    # get model save name if provided
    save_name_arg_idx = sys.argv.index("-n") + 1 if "-n" in sys.argv else -1
    model_save_name = (
        sys.argv[save_name_arg_idx] if save_name_arg_idx >= 0 else model_name
    )

    # load model
    model: Any | None = init_model(model_name)
    if model is None:
        print(
            "Usage: train_model.py <model_name:\
                NaiveBayes|DecisionTree|KNN|NN> [-n <model_save_name>]"
        )
        sys.exit(1)

    # load, prepare and split CoverType dataset
    X_train, X_test, y_train, y_test = load_split_dataset(
        os.path.join("data", "covtype.data")
    )

    # train and save model
    print(f"Training model {model}")
    if isinstance(model, KerasClassifier):
        model = optimize_model(model, X_train, y_train)
    else:
        model.fit(X_train, y_train)
    test_model(model, X_train, X_test, y_train, y_test, save=True)
    save_model(model, model_save_path, model_save_name)
