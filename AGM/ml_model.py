import pandas as pd
import os
import json
import numpy as np


DATA_DIR = os.path.join("static", "data")
LABEL_FILE = os.path.join(DATA_DIR, "labels.json")


def extract_feature(biomarker, task_type, disease_status, feature_types, start_min, end_min):
    if isinstance(feature_types, str):
        feature_types = [feature_types]
    """
    Extract selected features (slope, auc, difference, etc.) from all data
    matching the given conditions and return them as a DataFrame.
    """

    if not os.path.exists(LABEL_FILE):
        raise FileNotFoundError("labels.json file does not exist.")

    with open(LABEL_FILE, encoding="utf-8") as f:
        all_graphs = json.load(f)


    matched_graphs = [
        g for g in all_graphs
        if g["biomarker"] == biomarker and
           g["task_type"] == task_type and
           g["disease_status"] == disease_status
    ]

    records = []

    for g in matched_graphs:
        file_path = os.path.join(DATA_DIR, g["filename"])
        if not os.path.exists(file_path):
            continue

        df = pd.read_excel(file_path) if file_path.endswith(".xlsx") else pd.read_csv(file_path)


        x = df.iloc[:, 0].astype(float)
        y = df.iloc[:, 1].astype(float)

        mask = (x >= start_min) & (x <= end_min)
        x_sub = x[mask]
        y_sub = y[mask]

        row = {
            "subject_id": g["subject_id"],
            "title": g["title"]
        }

        for feature in feature_types:
            value = None
            if len(x_sub) >= 2:
                if feature == "slope":
                    value = np.polyfit(x_sub, y_sub, 1)[0]
                elif feature == "auc":
                    value = np.trapz(y_sub, x_sub)
                elif feature == "difference":
                    value = y_sub.max() - y_sub.min()
            row[feature] = value

        records.append(row)

    df_result = pd.DataFrame(records)
    return df_result