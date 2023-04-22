"""
Main entry point for the CoverType classification API.
"""

import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
from flask import Flask, request

app = Flask(__name__)


@app.route("/classify")
def classify_data() -> Any:
    # try to get feature data from request
    request_data: Optional[Dict] = request.get_json()
    if request_data is None:
        return None
    feature_data = np.array(request_data.get("feature_data"))
    if feature_data is None:
        return None

    # try to load trained model
    model_name = request_data.get("model", "")
    if model_name + ".pkl" in os.listdir("trained_models"):
        model = pickle.load(open(model_name + ".pkl", "rb"))
        return {"prediction": model.predict(feature_data)}
    else:
        return None
