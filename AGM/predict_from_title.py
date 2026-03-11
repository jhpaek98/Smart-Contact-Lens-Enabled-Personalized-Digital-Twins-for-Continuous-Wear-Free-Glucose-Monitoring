import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import uuid

DATA_DIR = os.path.join("static", "data")
LABEL_FILE = os.path.join(DATA_DIR, "labels.json")
EVENT_FILE = os.path.join(DATA_DIR, "events.json")
RESULTS_DIR = os.path.join(DATA_DIR, "ml_model")

MODEL_PATH = os.path.join(RESULTS_DIR, "rf_model.joblib")


def predict_from_title(title: str):
    with open(LABEL_FILE, encoding="utf-8") as f:
        graphs = json.load(f)
    with open(EVENT_FILE, encoding="utf-8") as f:
        events = json.load(f)

    graph = next((g for g in graphs if g["title"] == title), None)
    if not graph:
        raise ValueError(f"No graph found for title '{title}'.")

    if title not in events:
        raise ValueError(f"No event information found for '{title}'.")

    valid_meal_event = next(
        (ev for ev in events[title] if ev["type"] == "meal" and 20 <= ev["start"] <= 30),
        None
    )
    if not valid_meal_event:
        raise ValueError(f"No 'meal' event between 20–30 minutes in '{title}'.")

    start_min = valid_meal_event["start"]
    file_path = os.path.join(DATA_DIR, graph["filename"])
    df = pd.read_excel(file_path) if file_path.endswith(".xlsx") else pd.read_csv(file_path)

    x = df.iloc[:, 0].astype(float)
    y = df.iloc[:, 1].astype(float)

    x_interv = x[(x >= start_min) & (x <= 180)]
    y_interv = y[(x >= start_min) & (x <= 180)]

    x_baseline = x[x < start_min]
    y_baseline = y[x < start_min]

    if len(y_baseline) < 5:
        raise ValueError("Insufficient baseline data.")

    baseline_mean = y_baseline.mean()
    baseline_std = y_baseline.std()
    baseline_auc = np.trapz(y_baseline, x_baseline)
    baseline_diff = y_baseline.max() - y_baseline.min()
    baseline_slope = np.polyfit(x_baseline, y_baseline, 1)[0]

    feature_vector = np.array([[
        baseline_mean,
        baseline_std,
        baseline_auc,
        baseline_diff,
        baseline_slope
    ]])

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    prediction = model.predict(feature_vector)
    return {
        "title": title,
        "prediction": prediction[0].tolist(),
        "ground_truth": y_interv.tolist(),
        "x": x_interv.tolist()
    }