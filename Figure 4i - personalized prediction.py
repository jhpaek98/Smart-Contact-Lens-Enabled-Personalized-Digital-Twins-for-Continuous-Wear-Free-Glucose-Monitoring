import os
import numpy as np
import pandas as pd
import pickle
import joblib

from collections import Counter
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = "path"

CONDITION_FOLDERS = [
    "15E_Highcarb",
    "15E_Lowcarb",
    "15E_Modcarb",
    "30E_Highcarb",
    "30E_Modcarb",
    "30E_Lowcarb",
    "Lowcarb_15E",
    "Lowcarb_30E",
    "Modcarb_15E",
    "Modcarb_30E",
    "Highcarb_15E",
    "Highcarb_30E",
    "30 Exercise",
    "15 Exercise",
    "Highcarb",
    "Modcarb",
    "Lowcarb"
]

INTERVENTION_TYPES = [
    "Highcarb",
    "Modcarb",
    "Lowcarb",
    "15 Exercise",
    "30 Exercise",
]

MIN_BASELINE_LEN = 20
MAX_LEN = 180


def compute_baseline_features(curve: np.ndarray) -> np.ndarray:
    baseline_mean = np.mean(curve)
    baseline_std = np.std(curve)
    baseline_auc = np.sum(curve)
    baseline_peak = np.max(curve)
    baseline_trough = np.min(curve)

    if len(curve) >= 5:
        x_axis = np.arange(len(curve))
        coeffs = np.polyfit(x_axis[-5:], curve[-5:], 1)
        baseline_slope = coeffs[0]
    else:
        baseline_slope = 0.0

    return np.array(
        [
            baseline_mean,
            baseline_std,
            baseline_auc,
            baseline_peak,
            baseline_trough,
            baseline_slope,
        ],
        dtype=np.float32,
    )


def encode_intervention_onehot(series: pd.Series, types: list) -> np.ndarray:
    s = series.fillna("").astype(str).str.strip()
    unknown = set(s.unique()) - set([""] + types)
    if unknown:
        print("unknown Intervention values:", unknown)

    flags = []
    for t in types:
        flags.append((s == t).astype(np.float32).values)
    flags = np.stack(flags, axis=1)
    return flags


def build_features_from_file_case1(predict_file: str):
    df = pd.read_excel(predict_file)
    df.columns = df.columns.str.strip()

    required_cols = ["TG", "Intervention"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[{os.path.basename(predict_file)}] missing column: {col}")

    tg_full = df["TG"].values.astype(np.float32)
    n_points = len(tg_full)
    max_len = min(MAX_LEN, n_points)

    intervention_flags = encode_intervention_onehot(df["Intervention"], INTERVENTION_TYPES)
    summed_signal = intervention_flags.sum(axis=1)
    diff_signal = np.diff(summed_signal, prepend=0)
    on_indices = np.where(diff_signal >= 1)[0]

    if len(on_indices) != 1:
        raise ValueError(f"[{os.path.basename(predict_file)}] not Case1. len(on_indices)={len(on_indices)}")

    T1 = on_indices[0]

    if T1 < MIN_BASELINE_LEN:
        raise ValueError(f"[{os.path.basename(predict_file)}] baseline too short. T1={T1}")

    baseline_curve = tg_full[0:T1]
    baseline_feat = compute_baseline_features(baseline_curve)

    intervention_seq = intervention_flags[T1:max_len, :]
    intervention_sum = intervention_seq.sum(axis=0)
    intervention_present = (intervention_sum > 0).astype(np.float32)

    intervention_feat = np.concatenate([intervention_sum, intervention_present])

    X = np.concatenate([baseline_feat, intervention_feat])
    y = tg_full[T1:max_len]

    return {
        "case": "case1",
        "X": X,
        "y": y,
        "T1": int(T1),
        "n_points": n_points,
        "max_len": max_len,
        "tg_full": tg_full,
    }


def build_features_from_file_case2(predict_file: str):
    df = pd.read_excel(predict_file)
    df.columns = df.columns.str.strip()

    required_cols = ["TG", "Intervention"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[{os.path.basename(predict_file)}] missing column: {col}")

    tg_full = df["TG"].values.astype(np.float32)
    n_points = len(tg_full)
    max_len = min(MAX_LEN, n_points)

    intervention_flags = encode_intervention_onehot(df["Intervention"], INTERVENTION_TYPES)
    summed_signal = intervention_flags.sum(axis=1)
    diff_signal = np.diff(summed_signal, prepend=0)
    on_indices = np.where(diff_signal >= 1)[0]

    if len(on_indices) < 2:
        raise ValueError(f"[{os.path.basename(predict_file)}] does not satisfy Case2")

    T1 = on_indices[0]
    T2 = on_indices[1]

    if T1 < MIN_BASELINE_LEN:
        raise ValueError(f"[{os.path.basename(predict_file)}] baseline too short. T1={T1}")

    baseline_curve_PH1 = tg_full[0:T1]
    baseline_feat_PH1 = compute_baseline_features(baseline_curve_PH1)

    intervention_seq_PH1 = intervention_flags[T1:T2, :]
    intervention_sum_PH1 = intervention_seq_PH1.sum(axis=0)
    intervention_present_PH1 = (intervention_sum_PH1 > 0).astype(np.float32)
    intervention_feat_PH1 = np.concatenate([intervention_sum_PH1, intervention_present_PH1])

    X_PH1 = np.concatenate([baseline_feat_PH1, intervention_feat_PH1])
    y_PH1 = tg_full[T1:T2]

    baseline_curve_PH2 = tg_full[T1:T2]
    baseline_feat_PH2 = compute_baseline_features(baseline_curve_PH2)

    intervention_seq_PH2 = intervention_flags[T2:max_len, :]
    intervention_sum_PH2 = intervention_seq_PH2.sum(axis=0)
    intervention_present_PH2 = (intervention_sum_PH2 > 0).astype(np.float32)
    intervention_feat_PH2 = np.concatenate([intervention_sum_PH2, intervention_present_PH2])

    X_PH2 = np.concatenate([baseline_feat_PH2, intervention_feat_PH2])
    y_PH2 = tg_full[T2:max_len]

    return {
        "case": "case2",
        "X_PH1": X_PH1,
        "y_PH1": y_PH1,
        "X_PH2": X_PH2,
        "y_PH2": y_PH2,
        "T1": int(T1),
        "T2": int(T2),
        "n_points": n_points,
        "max_len": max_len,
        "tg_full": tg_full,
    }


def build_pkls_and_models_for_folder_case1(folder: str):
    print(f"\n=== Start folder processing (PKL+MODEL, Case1): {folder} ===")

    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".xlsx")
    ]
    files = sorted(files)

    dataset_case1 = []

    for file in files:
        try:
            feats = build_features_from_file_case1(file)
        except ValueError as e:
            print("skip:", e)
            continue

        dataset_case1.append({
            "X": feats["X"],
            "y": feats["y"],
            "curve_len": len(feats["y"]),
            "file": file,
            "T1": feats["T1"],
        })

    if not dataset_case1:
        print("no Case1 data. PKL/MODEL not generated.")
        return

    pkl_case1_path = os.path.join(folder, "rf_data_case1.pkl")
    with open(pkl_case1_path, "wb") as f:
        pickle.dump(dataset_case1, f)
    print(f"PKL saved: {pkl_case1_path}")

    lengths = [d["curve_len"] for d in dataset_case1]
    L, _ = Counter(lengths).most_common(1)[0]
    sel = [d for d in dataset_case1 if d["curve_len"] == L]

    X_case1 = np.stack([d["X"] for d in sel], axis=0)
    Y_case1 = np.stack([d["y"] for d in sel], axis=0)

    print(f"Case1 model training: n={X_case1.shape[0]}, L={L}")
    rf_case1 = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )
    rf_case1.fit(X_case1, Y_case1)

    model_case1_path = os.path.join(folder, f"rf_model_case1_len{L}.joblib")
    joblib.dump(rf_case1, model_case1_path)
    print(f"model saved: {model_case1_path}")


def build_pkls_and_models_for_folder_case2(folder: str):
    print(f"\n=== Start folder processing (PKL+MODEL, Case2): {folder} ===")

    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".xlsx")
    ]
    files = sorted(files)

    dataset_case2_PH1 = []
    dataset_case2_PH2 = []

    for file in files:
        try:
            feats = build_features_from_file_case2(file)
        except ValueError as e:
            print("skip:", e)
            continue

        dataset_case2_PH1.append({
            "X": feats["X_PH1"],
            "y": feats["y_PH1"],
            "curve_len": len(feats["y_PH1"]),
            "file": file,
            "T1": feats["T1"],
            "T2": feats["T2"],
        })

        dataset_case2_PH2.append({
            "X": feats["X_PH2"],
            "y": feats["y_PH2"],
            "curve_len": len(feats["y_PH2"]),
            "file": file,
            "T1": feats["T1"],
            "T2": feats["T2"],
        })

    if not dataset_case2_PH1 or not dataset_case2_PH2:
        print("insufficient Case2 data. PKL/MODEL not generated.")
        return

    pkl_PH1_path = os.path.join(folder, "rf_data_case2_PH1.pkl")
    pkl_PH2_path = os.path.join(folder, "rf_data_case2_PH2.pkl")

    with open(pkl_PH1_path, "wb") as f:
        pickle.dump(dataset_case2_PH1, f)
    with open(pkl_PH2_path, "wb") as f:
        pickle.dump(dataset_case2_PH2, f)

    print(f"PKL saved: {pkl_PH1_path}")
    print(f"PKL saved: {pkl_PH2_path}")

    lengths_PH1 = [d["curve_len"] for d in dataset_case2_PH1]
    L1, _ = Counter(lengths_PH1).most_common(1)[0]
    sel_PH1 = [d for d in dataset_case2_PH1 if d["curve_len"] == L1]

    X_PH1 = np.stack([d["X"] for d in sel_PH1], axis=0)
    Y_PH1 = np.stack([d["y"] for d in sel_PH1], axis=0)

    lengths_PH2 = [d["curve_len"] for d in dataset_case2_PH2]
    L2, _ = Counter(lengths_PH2).most_common(1)[0]
    sel_PH2 = [d for d in dataset_case2_PH2 if d["curve_len"] == L2]

    X_PH2 = np.stack([d["X"] for d in sel_PH2], axis=0)
    Y_PH2 = np.stack([d["y"] for d in sel_PH2], axis=0)

    print(f"PH1 model training: n={X_PH1.shape[0]}, L1={L1}")
    rf_PH1 = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )
    rf_PH1.fit(X_PH1, Y_PH1)

    print(f"PH2 model training: n={X_PH2.shape[0]}, L2={L2}")
    rf_PH2 = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )
    rf_PH2.fit(X_PH2, Y_PH2)

    model_PH1_path = os.path.join(folder, f"rf_model_case2_PH1_len{L1}.joblib")
    model_PH2_path = os.path.join(folder, f"rf_model_case2_PH2_len{L2}.joblib")

    joblib.dump(rf_PH1, model_PH1_path)
    joblib.dump(rf_PH2, model_PH2_path)

    print(f"model saved: {model_PH1_path}")
    print(f"model saved: {model_PH2_path}")


def run_loocv_on_folder_case1(base_dir: str, results_dir: str):
    print(f"\n=== Start folder processing (LOOCV, Case1): {base_dir} ===")
    os.makedirs(results_dir, exist_ok=True)

    all_files = [
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if f.endswith(".xlsx")
    ]
    all_files = sorted(all_files)

    print("number of target files:", len(all_files))
    if not all_files:
        print("no .xlsx files found.")
        return

    metrics_records = []

    for test_file in all_files:
        test_name = os.path.basename(test_file)
        print("\n  ------------------------------")
        print(f"  LOOCV fold - Test file (Case1): {test_name}")
        print("  ------------------------------")

        try:
            test_feats = build_features_from_file_case1(test_file)
        except ValueError as e:
            print("skip (failed to create Case1 test feature):", e)
            continue

        L_test = len(test_feats["y"])

        train_X, train_y = [], []

        for train_file in all_files:
            if train_file == test_file:
                continue

            try:
                feats = build_features_from_file_case1(train_file)
            except ValueError as e:
                print(f"skip training file [{os.path.basename(train_file)}]: {e}")
                continue

            if len(feats["y"]) != L_test:
                print(f"skip training file [{os.path.basename(train_file)}]: length match")
                continue

            train_X.append(feats["X"])
            train_y.append(feats["y"])

        if len(train_X) == 0:
            print("insufficient training samples. skipping.")
            continue

        X_train = np.stack(train_X, axis=0)
        Y_train = np.stack(train_y, axis=0)

        print(f"Case1 train shape: {X_train.shape} -> y: {Y_train.shape}")

        rf_case1 = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        )
        rf_case1.fit(X_train, Y_train)

        X_test = test_feats["X"].reshape(1, -1)
        y_true_seg = test_feats["y"]

        y_pred_seg = rf_case1.predict(X_test).reshape(-1)
        y_pred_seg = np.round(y_pred_seg, 3)

        if len(y_true_seg) != len(y_pred_seg):
            print("segment length mismatch:", len(y_true_seg), len(y_pred_seg))
            continue

        T1 = test_feats["T1"]
        max_len = test_feats["max_len"]
        tg_full = test_feats["tg_full"]
        n_points = len(tg_full)

        print(f"T1={T1}, max_len={max_len}, n_points={n_points}")
        print(f"segment length: {len(y_true_seg)} (expected {max_len - T1})")

        TG_true_all = tg_full.copy()
        TG_pred_all = np.full_like(tg_full, np.nan, dtype=np.float32)

        end_idx = T1 + len(y_pred_seg)
        if end_idx > n_points:
            print("end_idx > n_points, truncated:", end_idx, ">", n_points)
            end_idx = n_points
            y_pred_seg = y_pred_seg[: (end_idx - T1)]
            y_true_seg = y_true_seg[: (end_idx - T1)]

        TG_pred_all[T1:end_idx] = y_pred_seg

        mask = ~np.isnan(TG_pred_all)
        if mask.sum() < 2:
            print("predicted segment too short for metric calculation.")
            continue

        r_value, p_value = pearsonr(TG_true_all[mask], TG_pred_all[mask])
        rmse = np.sqrt(np.mean((TG_true_all[mask] - TG_pred_all[mask]) ** 2))
        mae = np.mean(np.abs(TG_true_all[mask] - TG_pred_all[mask]))
        bias = np.mean(TG_pred_all[mask] - TG_true_all[mask])

        print(f"r={r_value:.4f}, RMSE={rmse:.6f}, MAE={mae:.6f}, Bias={bias:.6f}")

        metrics_records.append({
            "file": test_name,
            "T1": T1,
            "n_points": n_points,
            "max_len": max_len,
            "L": L_test,
            "r": r_value,
            "RMSE": rmse,
            "MAE": mae,
            "Bias": bias,
        })

        df_out = pd.DataFrame({
            "Time_idx": np.arange(len(TG_true_all)),
            "TG_true_full": TG_true_all,
            "TG_pred_full": TG_pred_all,
            "has_pred": (~np.isnan(TG_pred_all)).astype(int),
        })

        out_name = f"pred_full_case1_{os.path.splitext(test_name)[0]}.xlsx"
        out_path = os.path.join(results_dir, out_name)
        df_out.to_excel(out_path, index=False)
        print("per-file result saved:", out_path)

    if metrics_records:
        df_metrics = pd.DataFrame(metrics_records)
        metrics_path = os.path.join(results_dir, "metrics_loocv_case1.xlsx")
        df_metrics.to_excel(metrics_path, index=False)
        print("LOOCV metrics saved:", metrics_path)
    else:
        print("no recorded metrics. all files may have been skipped.")


def run_loocv_on_folder_case2(base_dir: str, results_dir: str):
    print(f"\n=== Start folder processing (LOOCV, Case2): {base_dir} ===")
    os.makedirs(results_dir, exist_ok=True)

    all_files = [
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if f.endswith(".xlsx")
    ]
    all_files = sorted(all_files)

    print("number of target files:", len(all_files))
    if not all_files:
        print("no .xlsx files found.")
        return

    metrics_records = []

    for test_file in all_files:
        test_name = os.path.basename(test_file)
        print("\n  ------------------------------")
        print(f"  LOOCV fold - Test file (Case2): {test_name}")
        print("  ------------------------------")

        try:
            test_feats = build_features_from_file_case2(test_file)
        except ValueError as e:
            print("skip (failed to create Case2 test feature):", e)
            continue

        L1_test = len(test_feats["y_PH1"])
        L2_test = len(test_feats["y_PH2"])

        train_X_PH1, train_y_PH1 = [], []
        train_X_PH2, train_y_PH2 = [], []

        for train_file in all_files:
            if train_file == test_file:
                continue

            try:
                feats = build_features_from_file_case2(train_file)
            except ValueError as e:
                print(f"skip training file [{os.path.basename(train_file)}]: {e}")
                continue

            if len(feats["y_PH1"]) != L1_test or len(feats["y_PH2"]) != L2_test:
                print(f"skip training file [{os.path.basename(train_file)}]: length mismatch")
                continue

            train_X_PH1.append(feats["X_PH1"])
            train_y_PH1.append(feats["y_PH1"])
            train_X_PH2.append(feats["X_PH2"])
            train_y_PH2.append(feats["y_PH2"])

        if len(train_X_PH1) == 0 or len(train_X_PH2) == 0:
            print("insufficient training samples. skipping.")
            continue

        X_PH1_train = np.stack(train_X_PH1, axis=0)
        Y_PH1_train = np.stack(train_y_PH1, axis=0)
        X_PH2_train = np.stack(train_X_PH2, axis=0)
        Y_PH2_train = np.stack(train_y_PH2, axis=0)

        print(f"PH1 train shape: {X_PH1_train.shape} -> y: {Y_PH1_train.shape}")
        print(f"PH2 train shape: {X_PH2_train.shape} -> y: {Y_PH2_train.shape}")

        rf_PH1 = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        )
        rf_PH1.fit(X_PH1_train, Y_PH1_train)

        rf_PH2 = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        )
        rf_PH2.fit(X_PH2_train, Y_PH2_train)

        X_PH1_test = test_feats["X_PH1"].reshape(1, -1)
        X_PH2_test = test_feats["X_PH2"].reshape(1, -1)

        y_true_PH1 = test_feats["y_PH1"]
        y_true_PH2 = test_feats["y_PH2"]

        y_pred_PH1 = rf_PH1.predict(X_PH1_test).reshape(-1)
        y_pred_PH2 = rf_PH2.predict(X_PH2_test).reshape(-1)

        y_pred_PH1 = np.round(y_pred_PH1, 3)
        y_pred_PH2 = np.round(y_pred_PH2, 3)

        y_true_seg = np.concatenate([y_true_PH1, y_true_PH2])
        y_pred_seg = np.concatenate([y_pred_PH1, y_pred_PH2])

        if len(y_true_seg) != len(y_pred_seg):
            print("segment length mismatch:", len(y_true_seg), len(y_pred_seg))
            continue

        T1 = test_feats["T1"]
        T2 = test_feats["T2"]
        max_len = test_feats["max_len"]
        tg_full = test_feats["tg_full"]
        n_points = len(tg_full)

        print(f"T1={T1}, T2={T2}, max_len={max_len}, n_points={n_points}")
        print(f"segment length: {len(y_true_seg)} (expected {max_len - T1})")

        TG_true_all = tg_full.copy()
        TG_pred_all = np.full_like(tg_full, np.nan, dtype=np.float32)

        end_idx = T1 + len(y_pred_seg)
        if end_idx > n_points:
            print("end_idx > n_points, truncated:", end_idx, ">", n_points)
            end_idx = n_points
            y_pred_seg = y_pred_seg[: (end_idx - T1)]
            y_true_seg = y_true_seg[: (end_idx - T1)]

        TG_pred_all[T1:end_idx] = y_pred_seg

        mask = ~np.isnan(TG_pred_all)
        if mask.sum() < 2:
            print("predicted segment too short for metric calculation.")
            continue

        r_value, p_value = pearsonr(TG_true_all[mask], TG_pred_all[mask])
        rmse = np.sqrt(np.mean((TG_true_all[mask] - TG_pred_all[mask]) ** 2))
        mae = np.mean(np.abs(TG_true_all[mask] - TG_pred_all[mask]))
        bias = np.mean(TG_pred_all[mask] - TG_true_all[mask])

        print(f"r={r_value:.4f}, RMSE={rmse:.6f}, MAE={mae:.6f}, Bias={bias:.6f}")

        metrics_records.append({
            "file": test_name,
            "T1": T1,
            "T2": T2,
            "n_points": n_points,
            "max_len": max_len,
            "L1": L1_test,
            "L2": L2_test,
            "r": r_value,
            "RMSE": rmse,
            "MAE": mae,
            "Bias": bias,
        })

        df_out = pd.DataFrame({
            "Time_idx": np.arange(len(TG_true_all)),
            "TG_true_full": TG_true_all,
            "TG_pred_full": TG_pred_all,
            "has_pred": (~np.isnan(TG_pred_all)).astype(int),
        })

        out_name = f"pred_full_case2_{os.path.splitext(test_name)[0]}.xlsx"
        out_path = os.path.join(results_dir, out_name)
        df_out.to_excel(out_path, index=False)
        print("per-file result saved:", out_path)

    if metrics_records:
        df_metrics = pd.DataFrame(metrics_records)
        metrics_path = os.path.join(results_dir, "metrics_loocv_case2.xlsx")
        df_metrics.to_excel(metrics_path, index=False)
        print("LOOCV metrics saved:", metrics_path)
    else:
        print("no recorded metrics. all files may have been skipped.")


def run_all_on_subject_folder(subject_folder: str):
    print("\n" + "-" * 80)
    print(f"Subject folder processing: {subject_folder}")
    print("-" * 80)

    build_pkls_and_models_for_folder_case1(subject_folder)
    results_dir_case1 = os.path.join(subject_folder, "results_loocv_case1")
    run_loocv_on_folder_case1(subject_folder, results_dir_case1)

    build_pkls_and_models_for_folder_case2(subject_folder)
    results_dir_case2 = os.path.join(subject_folder, "results_loocv_case2")
    run_loocv_on_folder_case2(subject_folder, results_dir_case2)


def run_all_on_condition_folder(condition_root: str):
    print("\n" + "=" * 80)
    print(f"Condition folder processing start: {condition_root}")
    print("=" * 80)

    subdirs = [
        os.path.join(condition_root, d)
        for d in os.listdir(condition_root)
        if os.path.isdir(os.path.join(condition_root, d))
    ]
    subdirs = sorted(subdirs)

    data_subdirs = [
        d for d in subdirs
        if not os.path.basename(d).startswith("results_")
    ]

    print("subfolder list:", data_subdirs)

    if not data_subdirs:
        run_all_on_subject_folder(condition_root)
        return

    for folder in data_subdirs:
        run_all_on_subject_folder(folder)


if __name__ == "__main__":
    seen = set()
    unique_condition_folders = []
    for name in CONDITION_FOLDERS:
        if name not in seen:
            seen.add(name)
            unique_condition_folders.append(name)

    for cond_name in unique_condition_folders:
        top_root = os.path.join(BASE_DIR, cond_name)

        if not os.path.isdir(top_root):
            print(f"\nskip because folder does not exist: {top_root}")
            continue

        run_all_on_condition_folder(top_root)