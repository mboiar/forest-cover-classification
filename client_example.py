"""
An example showing how to access application REST API.
"""

import requests

if __name__ == "__main__":
    payload = {
        "feature_data": [
            2596,
            51,
            3,
            258,
            0,
            510,
            221,
            232,
            148,
            6279,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        "model": "DecisionTree",
    }
    res = requests.post("http://192.168.0.101:5000/classify", json=payload)
    print(res.status_code)
    print(res.json()["prediction"])
