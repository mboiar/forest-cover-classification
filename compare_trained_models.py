"""
A script used to display and compare different models' performance on a given
dataset with optional report generation.
"""

import json
import os
import sys
from typing import Dict, Union

import joblib

from train_model import load_split_dataset, test_model


def compare_models(
    dirpath: Union[os.PathLike, str], save_report: bool = False
) -> Dict:
    """Load and evaluate models"""
    # load all trained models
    model_filenames = os.listdir(dirpath)
    models = [
        joblib.load(open(os.path.join(dirpath, filename), "rb"))
        for filename in model_filenames
    ]
    if not models:
        print("No trained models found.")
        sys.exit(0)

    for model in models:
        print(f"Loaded model {model}")

    # load evaluation data
    X_train, X_test, y_train, y_test = load_split_dataset(
        os.path.join("data", "covtype.data")
    )

    # generate report
    report = {}
    for model in models:
        report[str(model)] = test_model(
            model, X_train, X_test, y_train, y_test
        )
    if save_report:
        json.dump(report, open("report.json", "a"))
    return report


if __name__ == "__main__":
    compare_models("trained_models", save_report=True)
