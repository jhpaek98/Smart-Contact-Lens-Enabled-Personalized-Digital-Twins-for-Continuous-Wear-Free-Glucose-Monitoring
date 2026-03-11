import os
import pandas as pd
import json
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

DATA_DIR = os.path.join("static", "data")
LABEL_FILE = os.path.join(DATA_DIR, "labels.json")
EVENT_FILE = os.path.join(DATA_DIR, "events.json")
RESULTS_DIR = os.path.join(DATA_DIR, "ml_model")

os.makedirs(RESULTS_DIR, exist_ok=True)


def train_model_from_titles(selected_titles: list):
    if not os.path.exists(LABEL_FILE) or not os.path.exists(EVENT_FILE):
        raise FileNotFoundError("labels.json or events.json file not found.")

    with open(LABEL_FILE, encoding="utf-8") as f:
        graphs = json.load(f)

    with open(EVENT_FILE, encoding="utf-8") as f:
        events = json.load(f)

    dataset = []

    for graph in graphs:
        title = graph["title"]
        if title not in selected_titles:
            continue

        if title not in events:
            raise ValueError(f"No event information found for graph '{title}'.")

        valid_meal_event = any(
            ev["type"] == "meal" and 20 <= ev["start"] <= 30 for ev in events[title]
        )
        if not valid_meal_event:
            raise ValueError(f"No 'meal' event between 20–30 minutes in '{title}'.")

        file_path = os.path.join(DATA_DIR, graph["filename"])
        df = pd.read_excel(file_path) if file_path.endswith(".xlsx") else pd.read_csv(file_path)

        x = df.iloc[:, 0].astype(float)
        y = df.iloc[:, 1].astype(float)

        meal_event = next(
            ev for ev in events[title] if ev["type"] == "meal" and 20 <= ev["start"] <= 30
        )
        start_min = meal_event["start"]

        x_baseline = x[x < start_min]
        y_baseline = y[x < start_min]

        x_interv = x[(x >= start_min) & (x <= 180)]
        y_interv = y[(x >= start_min) & (x <= 180)]

        if len(y_baseline) < 5 or len(y_interv) < 5:
            continue

        baseline_mean = y_baseline.mean()
        baseline_std = y_baseline.std()
        baseline_auc = np.trapz(y_baseline, x_baseline)
        baseline_diff = y_baseline.max() - y_baseline.min()
        baseline_slope = np.polyfit(x_baseline, y_baseline, 1)[0]

        feature_vector = [
            baseline_mean,
            baseline_std,
            baseline_auc,
            baseline_diff,
            baseline_slope
        ]

        dataset.append({
            "X": feature_vector,
            "y": y_interv.tolist(),
            "title": title
        })

    if not dataset:
        raise ValueError("No valid data available for model training.")

    X = np.array([d["X"] for d in dataset])
    y = np.array([d["y"] for d in dataset])

    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=42)
    )
    model.fit(X, y)

    with open(os.path.join(RESULTS_DIR, "rf_model.joblib"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(RESULTS_DIR, "rf_data.pkl"), "wb") as f:
        pickle.dump(dataset, f)

    return {
        "message": f"Model training completed using {len(dataset)} graphs.",
        "model_path": os.path.join(RESULTS_DIR, "rf_model.joblib")
    }