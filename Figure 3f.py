# OGTT features

import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import auc


folder_path = "input_folder"
output_dir = "output_folder"
output_filename = "TG_features.xlsx"
output_path = os.path.join(output_dir, output_filename)

file_pattern = re.compile(r"^[ND][1-5]_final_corrTG_pBG\.xlsx$", re.IGNORECASE)


def val_at_time(df, t, tol=None):
    exact = df.loc[df["Time"] == t, "TG_before_lag"]
    if len(exact) > 0:
        return float(exact.values[0])

    if tol is not None:
        sub = df.loc[(df["Time"] >= t - tol) & (df["Time"] <= t + tol),
                     ["Time", "TG_before_lag"]]
        if not sub.empty:
            idx = (sub["Time"] - t).abs().idxmin()
            return float(sub.loc[idx, "TG_before_lag"])

    return np.nan


def safe_auc(sub):
    sub = sub.dropna(subset=["Time", "TG_before_lag"])
    if len(sub) < 2:
        return np.nan

    xs = sub["Time"].values.astype(float)
    ys = sub["TG_before_lag"].values.astype(float)

    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    if len(np.unique(xs)) < 2:
        return np.nan

    return float(auc(xs, ys))


def process_file(file_path):
    df = pd.read_excel(file_path, sheet_name="Sheet1")

    if not {"Time", "TG_before_lag"}.issubset(df.columns):
        raise ValueError("Required columns are missing.")

    df = df.copy()
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["TG_before_lag"] = pd.to_numeric(df["TG_before_lag"], errors="coerce")

    df = df.dropna(subset=["Time", "TG_before_lag"]) \
           .sort_values("Time") \
           .reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid Time or TG_before_lag data.")

    delta = float(df["TG_before_lag"].iloc[-1] - df["TG_before_lag"].iloc[0])

    tg_1_30 = df[(df["Time"] >= 1) & (df["Time"] <= 30)]
    tg_40_70 = df[(df["Time"] >= 40) & (df["Time"] <= 70)]

    if not tg_1_30.empty and not tg_40_70.empty:
        min_1_30 = tg_1_30.loc[tg_1_30["TG_before_lag"].idxmin()]
        max_40_70 = tg_40_70.loc[tg_40_70["TG_before_lag"].idxmax()]
        denom = float(max_40_70["Time"] - min_1_30["Time"])

        if denom != 0:
            r_slope = float(
                (max_40_70["TG_before_lag"] - min_1_30["TG_before_lag"]) / denom
            )
        else:
            r_slope = np.nan
    else:
        r_slope = np.nan

    FTG = val_at_time(df, 0, tol=0.5)
    G60 = val_at_time(df, 60, tol=0.5)

    G120 = float(df["TG_before_lag"].iloc[-1])
    G_peak = float(df["TG_before_lag"].max())

    std_dev = float(df["TG_before_lag"].std()) if len(df) > 1 else np.nan
    mean_val = float(df["TG_before_lag"].mean())

    auc_1h = safe_auc(df[df["Time"] <= 60])
    auc_2h = safe_auc(df[df["Time"] <= 120])

    MCR = float(50 / (auc_2h * 120)) if auc_2h and auc_2h > 0 else np.nan

    return {
        "file": os.path.basename(file_path),
        "delta": delta,
        "r": r_slope,
        "FTG": FTG,
        "Standard Deviation": std_dev,
        "AUC_1h": auc_1h,
        "AUC_2h": auc_2h,
        "mean": mean_val,
        "MCR": MCR,
        "G60": G60,
        "G120": G120,
        "G_peak": G_peak,
    }


def main():
    results = []

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".xlsx") and file_pattern.match(fname):
            path = os.path.join(folder_path, fname)

            try:
                results.append(process_file(path))
            except Exception as e:
                print(f"Error processing {fname}: {e}")

    if not results:
        print("No files processed.")
        return

    df_results = pd.DataFrame(results).sort_values("file").reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_excel(output_path, index=False)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()