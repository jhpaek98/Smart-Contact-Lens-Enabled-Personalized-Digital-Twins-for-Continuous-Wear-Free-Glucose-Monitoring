# SHAP decision plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
import shap

INPUT_PATH = Path("TG_features.xlsx")
OUTPUT_DIR = Path("TG")
LABEL_COL = "file"
GROUP_PREFIX = True

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(INPUT_PATH)

labels_raw = df[LABEL_COL].astype(str)
if GROUP_PREFIX:
    labels_raw = labels_raw.str.extract(r"([A-Za-z]+)", expand=False)

le = LabelEncoder()
y = le.fit_transform(labels_raw)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if LABEL_COL in num_cols:
    num_cols.remove(LABEL_COL)

data = df[[LABEL_COL] + num_cols].dropna().copy()
labels_raw = data[LABEL_COL].astype(str)
if GROUP_PREFIX:
    labels_raw = labels_raw.str.extract(r"([A-Za-z]+)", expand=False)

y = le.fit_transform(labels_raw)
X = data[num_cols].to_numpy(dtype=float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

svm = SVC(kernel="linear", decision_function_shape="ovr", random_state=42)
svm.fit(X_scaled, y)

masker = shap.maskers.Independent(X_scaled)
explainer = shap.Explainer(svm.decision_function, masker)
shap_exp = explainer(X_scaled)

values = shap_exp.values
base = shap_exp.base_values
feature_names = num_cols

def save_fig(path, title=None):
    if title:
        plt.title(title)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

if values.ndim == 2:
    plt.figure(figsize=(10, 6))
    shap.decision_plot(
        base_value=float(np.mean(base)),
        shap_values=values,
        features=X_scaled,
        feature_names=feature_names,
        show=False
    )
    save_fig(OUTPUT_DIR / "shap_decision.png", "SHAP Decision Plot")

elif values.ndim == 3:
    n_classes = values.shape[2]
    for k in range(n_classes):
        class_name = le.classes_[k]

        plt.figure(figsize=(10, 6))
        shap.decision_plot(
            base_value=float(np.mean(base[:, k])),
            shap_values=values[:, :, k],
            features=X_scaled,
            feature_names=feature_names,
            show=False
        )
        save_fig(OUTPUT_DIR / f"shap_decision_{class_name}.png", f"Decision Plot ({class_name})")

else:
    raise ValueError(f"Unexpected SHAP shape: {values.shape}")