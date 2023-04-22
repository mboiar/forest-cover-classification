#!/bin/sh
export FLASK_APP=./classify_data.py
flask --debug run -h 0.0.0.0