import os
import glob
import stat
import shutil
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = r"C:\Users\jhpaek98\OneDrive\data\work\Fig5\260309_Fig5\generalized"

TARGET_COLUMN = "TG"

VALID_SUBJECTS = {"A", "B", "C", "D", "E"}

EXCLUDE_COLUMNS = {
    TARGET_COLUMN,
    "subject",
    "scenario",
    "file",
    "filename",
    "name",
    "ID",
    "id",
}

N_TRAIN_FILES = 9
N_REPEATS = 1
RANDOM_SEED = 42

RESULT_DIR_NAME = "results_random9_same_scenario"

SAVE_PREDICTIONS_PER_TEST = True

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


def remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def cleanup_generated_artifacts_in_folder(folder: str):
    print(f"\n[start cleanup] {folder}")

    result_dirs = [
        RESULT_DIR_NAME,
    ]

    for dname in result_dirs:
        dpath = os.path.join(folder, dname)
        if os.path.isdir(dpath):
            try:
                shutil.rmtree(dpath, onerror=remove_readonly)
                print(f"deleted dir: {dpath}")
            except Exception as e:
                print(f"failed dir: {dpath} | {e}")

    result_files = [
        "metrics_random9_same_scenario.xlsx",
        "summary_random9_same_scenario.xlsx",
    ]

    for fname in result_files:
        fpath = os.path.join(folder, fname)
        if os.path.isfile(fpath):
            try:
                os.chmod(fpath, stat.S_IWRITE)
                os.remove(fpath)
                print(f"deleted file: {fpath}")
            except Exception as e:
                print(f"failed file: {fpath} | {e}")

    print(f"[cleanup done] {folder}")


def safe_sheet_name(name: str) -> str:
    invalid = r'[]:*?/\\'
    for ch in invalid:
        name = name.replace(ch, "_")
    return name[:31]



def find_input_files(base_dir: str):

    pattern = os.path.join(base_dir, "*.xlsx")
    files = sorted(glob.glob(pattern))

    infos = []

    for fpath in files:
        infos.append({
            "path": fpath,
            "file": os.path.basename(fpath),
            "subject": "unknown",
            "scenario": "default",
            "number": "",
        })

    return infos


def read_excel_data(filepath: str):
    try:
        return pd.read_excel(filepath)
    except Exception as e:
        print(f"[read fail] {filepath} | {e}")
        return None


def choose_feature_columns(train_dfs, test_df, target_col, exclude_columns):
    common_cols = set(test_df.columns)

    for df in train_dfs:
        common_cols &= set(df.columns)

    feature_cols = []
    for col in sorted(common_cols):
        if col in exclude_columns:
            continue
        if col == target_col:
            continue

        if pd.api.types.is_numeric_dtype(test_df[col]):
            ok = True
            for df in train_dfs:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    ok = False
                    break
            if ok:
                feature_cols.append(col)

    return feature_cols


def build_train_test_arrays(train_dfs, test_df, feature_cols, target_col):
    train_concat = pd.concat(train_dfs, axis=0, ignore_index=True)

    train_concat = train_concat[train_concat[target_col].notna()].copy()
    test_df = test_df[test_df[target_col].notna()].copy()

    if len(train_concat) == 0 or len(test_df) == 0:
        return None, None, None, None

    X_train = train_concat[feature_cols].copy()
    y_train = train_concat[target_col].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    return X_train, y_train, X_test, y_test


def create_model(random_state: int):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=RF_PARAMS["n_estimators"],
            max_depth=RF_PARAMS["max_depth"],
            min_samples_split=RF_PARAMS["min_samples_split"],
            min_samples_leaf=RF_PARAMS["min_samples_leaf"],
            random_state=random_state,
            n_jobs=RF_PARAMS["n_jobs"],
        ))
    ])
    return model


def evaluate_one_test_file(test_info, same_scenario_infos, results_dir):
    candidate_train_infos = [
        x for x in same_scenario_infos
        if x["path"] != test_info["path"]
    ]

    if len(candidate_train_infos) < N_TRAIN_FILES:
        return [{
            "scenario": test_info["scenario"],
            "subject": test_info["subject"],
            "number": test_info["number"],
            "test_file": test_info["file"],
            "repeat": None,
            "n_candidates": len(candidate_train_infos),
            "n_train_files": 0,
            "n_train_rows": 0,
            "n_test_rows": 0,
            "n_features": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
            "status": f"skip_not_enough_train_files(<{N_TRAIN_FILES})",
            "train_files": "",
            "feature_columns": "",
        }]

    test_df = read_excel_data(test_info["path"])
    if test_df is None:
        return [{
            "scenario": test_info["scenario"],
            "subject": test_info["subject"],
            "number": test_info["number"],
            "test_file": test_info["file"],
            "repeat": None,
            "n_candidates": len(candidate_train_infos),
            "n_train_files": 0,
            "n_train_rows": 0,
            "n_test_rows": 0,
            "n_features": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
            "status": "skip_test_read_fail",
            "train_files": "",
            "feature_columns": "",
        }]

    if TARGET_COLUMN not in test_df.columns:
        return [{
            "scenario": test_info["scenario"],
            "subject": test_info["subject"],
            "number": test_info["number"],
            "test_file": test_info["file"],
            "repeat": None,
            "n_candidates": len(candidate_train_infos),
            "n_train_files": 0,
            "n_train_rows": 0,
            "n_test_rows": 0,
            "n_features": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
            "status": f"skip_target_not_found({TARGET_COLUMN})",
            "train_files": "",
            "feature_columns": "",
        }]

    metrics_records = []
    pred_rows_to_save = []

    for repeat_idx in range(N_REPEATS):
        rng = random.Random(RANDOM_SEED + repeat_idx)
        sampled_train_infos = rng.sample(candidate_train_infos, N_TRAIN_FILES)

        train_dfs = []
        bad_train = False

        for info in sampled_train_infos:
            df = read_excel_data(info["path"])
            if df is None:
                bad_train = True
                break
            if TARGET_COLUMN not in df.columns:
                bad_train = True
                break
            train_dfs.append(df)

        if bad_train:
            metrics_records.append({
                "scenario": test_info["scenario"],
                "subject": test_info["subject"],
                "number": test_info["number"],
                "test_file": test_info["file"],
                "repeat": repeat_idx + 1,
                "n_candidates": len(candidate_train_infos),
                "n_train_files": len(sampled_train_infos),
                "n_train_rows": 0,
                "n_test_rows": 0,
                "n_features": 0,
                "RMSE": np.nan,
                "MAE": np.nan,
                "R2": np.nan,
                "status": "skip_train_read_fail_or_target_missing",
                "train_files": " | ".join([x["file"] for x in sampled_train_infos]),
                "feature_columns": "",
            })
            continue

        feature_cols = choose_feature_columns(
            train_dfs=train_dfs,
            test_df=test_df,
            target_col=TARGET_COLUMN,
            exclude_columns=EXCLUDE_COLUMNS
        )

        if len(feature_cols) == 0:
            metrics_records.append({
                "scenario": test_info["scenario"],
                "subject": test_info["subject"],
                "number": test_info["number"],
                "test_file": test_info["file"],
                "repeat": repeat_idx + 1,
                "n_candidates": len(candidate_train_infos),
                "n_train_files": len(sampled_train_infos),
                "n_train_rows": 0,
                "n_test_rows": 0,
                "n_features": 0,
                "RMSE": np.nan,
                "MAE": np.nan,
                "R2": np.nan,
                "status": "skip_no_common_numeric_features",
                "train_files": " | ".join([x["file"] for x in sampled_train_infos]),
                "feature_columns": "",
            })
            continue

        arrays = build_train_test_arrays(
            train_dfs=train_dfs,
            test_df=test_df,
            feature_cols=feature_cols,
            target_col=TARGET_COLUMN
        )

        if arrays[0] is None:
            metrics_records.append({
                "scenario": test_info["scenario"],
                "subject": test_info["subject"],
                "number": test_info["number"],
                "test_file": test_info["file"],
                "repeat": repeat_idx + 1,
                "n_candidates": len(candidate_train_infos),
                "n_train_files": len(sampled_train_infos),
                "n_train_rows": 0,
                "n_test_rows": 0,
                "n_features": len(feature_cols),
                "RMSE": np.nan,
                "MAE": np.nan,
                "R2": np.nan,
                "status": "skip_empty_train_or_test_after_target_dropna",
                "train_files": " | ".join([x["file"] for x in sampled_train_infos]),
                "feature_columns": " | ".join(feature_cols),
            })
            continue

        X_train, y_train, X_test, y_test = arrays

        model = create_model(random_state=RANDOM_SEED + repeat_idx)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred) if len(y_test) >= 2 else np.nan

        metrics_records.append({
            "scenario": test_info["scenario"],
            "subject": test_info["subject"],
            "number": test_info["number"],
            "test_file": test_info["file"],
            "repeat": repeat_idx + 1,
            "n_candidates": len(candidate_train_infos),
            "n_train_files": len(sampled_train_infos),
            "n_train_rows": len(X_train),
            "n_test_rows": len(X_test),
            "n_features": len(feature_cols),
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "status": "ok",
            "train_files": " | ".join([x["file"] for x in sampled_train_infos]),
            "feature_columns": " | ".join(feature_cols),
        })

        if SAVE_PREDICTIONS_PER_TEST:
            pred_df = pd.DataFrame({
                "repeat": repeat_idx + 1,
                "y_true": np.asarray(y_test),
                "y_pred": np.asarray(y_pred),
                "error": np.asarray(y_pred) - np.asarray(y_test),
            })
            pred_rows_to_save.append(pred_df)

    if SAVE_PREDICTIONS_PER_TEST and len(pred_rows_to_save) > 0:
        pred_out = pd.concat(pred_rows_to_save, axis=0, ignore_index=True)
        pred_fname = f"pred_{os.path.splitext(test_info['file'])[0]}.xlsx"
        pred_path = os.path.join(results_dir, pred_fname)
        try:
            pred_out.to_excel(pred_path, index=False)
        except Exception as e:
            print(f"[prediction save fail] {pred_path} | {e}")

    return metrics_records


def run_random9_same_scenario(base_dir: str):
    print("\n" + "=" * 100)
    print(f"[start] {base_dir}")
    print("=" * 100)

    cleanup_generated_artifacts_in_folder(base_dir)

    results_dir = os.path.join(base_dir, RESULT_DIR_NAME)
    os.makedirs(results_dir, exist_ok=True)

    file_infos = find_input_files(base_dir)

    if not file_infos:
        print(f"[skip] no input files: {base_dir}")
        return

    print(f"[number of input files] {len(file_infos)}")

    scenario_to_infos = {}
    for info in file_infos:
        scenario_to_infos.setdefault(info["scenario"], []).append(info)

    for scenario, infos in sorted(scenario_to_infos.items()):
        print(f"scenario={scenario} | n_files={len(infos)}")

    all_metrics = []

    for test_info in file_infos:
        scenario_infos = scenario_to_infos[test_info["scenario"]]

        print(
            f"\n[test] {test_info['file']} | "
            f"scenario={test_info['scenario']} | "
            f"subject={test_info['subject']}"
        )

        result_list = evaluate_one_test_file(
            test_info=test_info,
            same_scenario_infos=scenario_infos,
            results_dir=results_dir
        )

        all_metrics.extend(result_list)

    if not all_metrics:
        print("[no results]")
        return

    df_metrics = pd.DataFrame(all_metrics)
    df_ok = df_metrics[df_metrics["status"] == "ok"].copy()

    if len(df_ok) > 0:
        df_summary_by_test = (
            df_ok.groupby(["subject", "scenario", "test_file"], as_index=False)
            .agg(
                repeats=("repeat", "count"),
                mean_RMSE=("RMSE", "mean"),
                std_RMSE=("RMSE", "std"),
                mean_MAE=("MAE", "mean"),
                std_MAE=("MAE", "std"),
                mean_R2=("R2", "mean"),
                std_R2=("R2", "std"),
                n_candidates=("n_candidates", "first"),
                n_train_files=("n_train_files", "first"),
                n_train_rows=("n_train_rows", "mean"),
                n_test_rows=("n_test_rows", "mean"),
                n_features=("n_features", "mean"),
            )
            .sort_values(["subject", "scenario", "test_file"])
        )

        df_summary_by_scenario = (
            df_ok.groupby(["scenario"], as_index=False)
            .agg(
                n_test_files=("test_file", "nunique"),
                mean_RMSE=("RMSE", "mean"),
                std_RMSE=("RMSE", "std"),
                mean_MAE=("MAE", "mean"),
                std_MAE=("MAE", "std"),
                mean_R2=("R2", "mean"),
                std_R2=("R2", "std"),
            )
            .sort_values(["scenario"])
        )

        df_summary_by_subject = (
            df_ok.groupby(["subject"], as_index=False)
            .agg(
                n_test_files=("test_file", "nunique"),
                mean_RMSE=("RMSE", "mean"),
                std_RMSE=("RMSE", "std"),
                mean_MAE=("MAE", "mean"),
                std_MAE=("MAE", "std"),
                mean_R2=("R2", "mean"),
                std_R2=("R2", "std"),
            )
            .sort_values(["subject"])
        )
    else:
        df_summary_by_test = pd.DataFrame()
        df_summary_by_scenario = pd.DataFrame()
        df_summary_by_subject = pd.DataFrame()

    metrics_path = os.path.join(base_dir, "metrics_random9_same_scenario.xlsx")
    summary_path = os.path.join(base_dir, "summary_random9_same_scenario.xlsx")

    try:
        with pd.ExcelWriter(metrics_path, engine="openpyxl") as writer:
            df_metrics.to_excel(writer, sheet_name="metrics_all", index=False)

            for subject, df_sub in df_metrics.groupby("subject"):
                sname = safe_sheet_name(f"sub_{subject}")
                df_sub.to_excel(writer, sheet_name=sname, index=False)

        with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
            df_summary_by_test.to_excel(writer, sheet_name="summary_by_test_file", index=False)
            df_summary_by_scenario.to_excel(writer, sheet_name="summary_by_scenario", index=False)
            df_summary_by_subject.to_excel(writer, sheet_name="summary_by_subject", index=False)

            for subject, df_sub in df_ok.groupby("subject"):
                sname = safe_sheet_name(f"sub_{subject}")
                df_sub.to_excel(writer, sheet_name=sname, index=False)

    except Exception as e:
        print(f"[excel save fail] {base_dir} | {e}")
        return

    print(f"\n[done] {base_dir}")
    print(f"metrics saved: {metrics_path}")
    print(f"summary saved: {summary_path}")
    print(f"prediction output dir: {results_dir}")


if __name__ == "__main__":
    run_random9_same_scenario(BASE_DIR)