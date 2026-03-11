# SVM

import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SRC_PATH = Path("input.xlsx")
LABEL_COL_NAME = None
GROUP_BY_PREFIX = True
OUTPUT_XLSX = Path("svm_3d_export.xlsx")

assert SRC_PATH.exists()

df = pd.read_excel(SRC_PATH)

def infer_label_column(df: pd.DataFrame):
    preferred = ['label','class','group','y','target','condition','state','diagnosis','file','filename']
    lower_map = {c.lower(): c for c in df.columns}
    for p in preferred:
        if p in lower_map:
            return lower_map[p]
    for c in df.columns:
        nunq = df[c].nunique(dropna=True)
        if 2 <= nunq <= 100:
            return c
    return None

label_col = LABEL_COL_NAME or infer_label_column(df)

if label_col is None:
    raise ValueError()

labels_raw = df[label_col].astype(str)

if GROUP_BY_PREFIX:
    def prefix_group(s: str):
        m = re.match(r"([A-Za-z]+)", s)
        return m.group(1) if m else s
    labels_grp = labels_raw.map(prefix_group)
else:
    labels_grp = labels_raw

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if label_col in num_cols:
    num_cols.remove(label_col)

assert len(num_cols) >= 3

work = df[num_cols].copy()
work['__y__'] = labels_grp
work = work.dropna(subset=num_cols + ['__y__'])

n_classes = work['__y__'].nunique()
print(n_classes, work['__y__'].value_counts().to_dict())

le = LabelEncoder()
y = le.fit_transform(work['__y__'])
X = work[num_cols].to_numpy(dtype=float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)

svm = SVC(kernel="linear", C=1.0, random_state=42, decision_function_shape="ovr")
svm.fit(X_pca, y)

cls_counts = pd.Series(y).value_counts()
min_class = cls_counts.min()
cv_text = ""

if min_class >= 2:
    k = min(5, int(min_class))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(svm, X_pca, y, cv=cv)
    cv_text = f"{scores.mean():.4f} ± {scores.std():.4f}"

print(cv_text)

y_pred = svm.predict(X_pca)
print(classification_report(y, y_pred, target_names=list(le.classes_), zero_division=0))
print(confusion_matrix(y, y_pred))

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')

label_text_map = {
    'N': 'Normal',
    'D': 'T2DM'
}

color_map = {
    'N': 'olive',
    'D': 'purple'
}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for cls_idx, cls_name in enumerate(le.classes_):
        idx = (y == cls_idx)
        key = cls_name[0].upper()
        color = color_map.get(key, 'gray')
        disp_name = label_text_map.get(key, cls_name)
        ax.scatter(
            X_pca[idx,0], X_pca[idx,1], X_pca[idx,2],
            label=disp_name,
            s=100,
            depthshade=False,
            edgecolor='black',
            linewidth=0.4,
            zorder=10,
            c=color
        )

x_vals = X_pca[:,0]
y_vals = X_pca[:,1]
z_vals = X_pca[:,2]

pad = 0.5
xr = (x_vals.min()-pad, x_vals.max()+pad)
yr = (y_vals.min()-pad, y_vals.max()+pad)
zr = (z_vals.min()-pad, z_vals.max()+pad)

xs = np.linspace(*xr, 35)
ys = np.linspace(*yr, 35)
zs = np.linspace(*zr, 35)

coefs = svm.coef_
intercepts = svm.intercept_

def _mask_to_data_range(XX, YY, ZZ, xr, yr, zr):
    mask = (XX < xr[0]) | (XX > xr[1]) | (YY < yr[0]) | (YY > yr[1]) | (ZZ < zr[0]) | (ZZ > zr[1])
    XX = XX.copy()
    YY = YY.copy()
    ZZ = ZZ.copy()
    XX[mask] = np.nan
    YY[mask] = np.nan
    ZZ[mask] = np.nan
    return XX, YY, ZZ

planes_mesh_records = []
planes_coef_records = []

for k in range(coefs.shape[0]):
    a, b, c = coefs[k]
    d = intercepts[k]

    absco = np.abs([a, b, c])
    dom = np.argmax(absco)

    if dom == 2:
        XX, YY = np.meshgrid(xs, ys)
        denom = c if c != 0 else 1e-12
        ZZ = (-d - a*XX - b*YY) / denom
        XX, YY, ZZ = _mask_to_data_range(XX, YY, ZZ, xr, yr, zr)
        ax.plot_surface(XX, YY, ZZ, alpha=0.25, linewidth=0, antialiased=True, color="lightgrey", zorder=1)

        planes_mesh_records.append(
            pd.DataFrame({
                "plane_id": k,
                "dom_axis": "z",
                "X": XX.ravel(),
                "Y": YY.ravel(),
                "Z": ZZ.ravel()
            })
        )
    elif dom == 0:
        YY, ZZ = np.meshgrid(ys, zs)
        denom = a if a != 0 else 1e-12
        XX = (-d - b*YY - c*ZZ) / denom
        XX, YY, ZZ = _mask_to_data_range(XX, YY, ZZ, xr, yr, zr)
        ax.plot_surface(XX, YY, ZZ, alpha=0.25, linewidth=0, antialiased=True, color="lightgrey", zorder=1)

        planes_mesh_records.append(
            pd.DataFrame({
                "plane_id": k,
                "dom_axis": "x",
                "X": XX.ravel(),
                "Y": YY.ravel(),
                "Z": ZZ.ravel()
            })
        )
    else:
        XX, ZZ = np.meshgrid(xs, zs)
        denom = b if b != 0 else 1e-12
        YY = (-d - a*XX - c*ZZ) / denom
        XX, YY, ZZ = _mask_to_data_range(XX, YY, ZZ, xr, yr, zr)
        ax.plot_surface(XX, YY, ZZ, alpha=0.25, linewidth=0, antialiased=True, color="lightgrey", zorder=1)

        planes_mesh_records.append(
            pd.DataFrame({
                "plane_id": k,
                "dom_axis": "y",
                "X": XX.ravel(),
                "Y": YY.ravel(),
                "Z": ZZ.ravel()
            })
        )

    planes_coef_records.append({
        "plane_id": k,
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "d": float(d),
        "dom_axis": ["x","y","z"][dom]
    })

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_xlim(*xr)
ax.set_ylim(*yr)
ax.set_zlim(*zr)
ax.set_box_aspect((xr[1]-xr[0], yr[1]-yr[0], zr[1]-zr[0]))

plt.title("SVM hyperplane")
ax.legend()
plt.tight_layout()
plt.show()

points_df = pd.DataFrame({
    "PC1": X_pca[:,0],
    "PC2": X_pca[:,1],
    "PC3": X_pca[:,2],
    "label": le.inverse_transform(y),
    "label_encoded": y
})

hyperplanes_df = pd.DataFrame(planes_coef_records)

plane_mesh_df = pd.concat(planes_mesh_records, ignore_index=True)

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    points_df.to_excel(writer, sheet_name="points", index=False)
    hyperplanes_df.to_excel(writer, sheet_name="hyperplanes", index=False)
    plane_mesh_df.to_excel(writer, sheet_name="plane_mesh", index=False)

print(OUTPUT_XLSX)