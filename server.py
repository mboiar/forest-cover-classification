"""
Main entry point for the CoverType classification API.
"""

import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
from flask import Flask, request

app = Flask(__name__)


@app.route("/classify", methods=["POST"])
def classify_data() -> Any:
    # try to get feature data from request
    request_data: Optional[Dict] = request.get_json()
    if request_data is None:
        return None
    feature_data = np.array(request_data.get("feature_data")).reshape(1, -1)
    if feature_data is None:
        return None

    # try to load trained model
    model_name = request_data.get("model", "")
    if model_name + ".pkl" in os.listdir("trained_models"):
        model = joblib.load(
            os.path.join("trained_models", model_name + ".pkl")
        )
        return {"prediction": model.predict(feature_data).tolist()}
    else:
        return None
