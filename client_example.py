"""
An example showing how to access application REST API.
"""

import requests

if __name__ == "__main__":
    payload = {
        "feature_data": [
            2590,
            56,
            2,
            212,
            -6,
            390,
            220,
            235,
            151,
            6225,
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
        "model": "NN",
    }
    res = requests.post("http://192.168.0.101:5000/classify", json=payload)
    print(res.status_code)
    print(res.json()["prediction"])
