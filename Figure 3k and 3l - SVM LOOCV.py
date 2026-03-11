import os
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

file_path = "path/features.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet1')

sample_names = df.iloc[:, 0]
X = df.drop(columns=df.columns[0])

if "Group" in df.columns:
    grp = df["Group"].astype(str).str.strip().str.upper()
    y_full = grp.str[0].map({"N": 0, "D": 1}).to_numpy()
else:
    n = len(df)
    half = n // 2
    y_full = np.array([1]*half + [0]*(n - half))

X = X.apply(pd.to_numeric, errors="coerce")
row_mask = ~X.isna().any(axis=1)
X_valid = X.loc[row_mask].copy()
y_valid = y_full[row_mask.to_numpy()]
samples_valid = sample_names.loc[row_mask].reset_index(drop=True)

if X_valid.shape[0] < 2:
    raise ValueError("Invalid sample size")
if X_valid.shape[1] < 1:
    raise ValueError("No numeric features")

loo = LeaveOneOut()
y_pred = np.empty(len(X_valid), dtype=int)

for i, (tr_idx, te_idx) in enumerate(loo.split(X_valid)):
    X_tr = X_valid.iloc[tr_idx, :].to_numpy()
    X_te = X_valid.iloc[te_idx, :].to_numpy()
    y_tr = y_valid[tr_idx]

    n_pc = min(5, X_tr.shape[1], X_tr.shape[0] - 1)
    if n_pc < 1:
        raise ValueError("Invalid PCA dimension")

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    pca = PCA(n_components=n_pc)
    X_tr_pca = pca.fit_transform(X_tr_sc)
    X_te_pca = pca.transform(X_te_sc)

    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X_tr_pca, y_tr)
    y_pred[te_idx[0]] = svm.predict(X_te_pca)[0]

cm = confusion_matrix(y_valid, y_pred, labels=[0, 1])
acc = accuracy_score(y_valid, y_pred)

print(cm)
print(acc)

out_df = pd.DataFrame({
    "Sample": samples_valid,
    "y_true": y_valid,
    "y_pred": y_pred
})

save_dir = "save_directory"
out_df.to_csv(os.path.join(save_dir, "svm_loocv_predictions.csv"), index=False)