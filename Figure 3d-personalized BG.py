# pBG for Parkes error grid
# Requirements:
# - BG, TG before lag correction, corrected TG

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

IN_DIR = Path("input_folder")
OUT_DIR = Path("output_folder")
FILE_PATTERN = "*.xlsx"
SHEET_NAME = "Sheet1"

TIME_COL_CANDIDATES = ["Time"]
BG_COL_CANDIDATES = ["BG"]
TG_BEFORE_COL_CANDIDATES = ["TG_before_lag"]
TG_CORR_COL_CANDIDATES = ["corrected_TG"]

BG_SCALE = 1.0


def norm(c):
    return " ".join(str(c).strip().split()).lower()


def find_col(df, name):
    m = {norm(c): c for c in df.columns}
    return m.get(norm(name), None)


def find_first_col(df, candidates):
    for name in candidates:
        col = find_col(df, name)
        if col is not None:
            return col
    return None


def iter_input_files(in_path: Path):
    if in_path.is_file():
        return [in_path]
    if in_path.is_dir():
        return sorted(in_path.glob(FILE_PATTERN))
    return []


def safe_save_excel(df, out_path: Path):
    try:
        df.to_excel(out_path, index=False)
        return out_path, None
    except PermissionError as e:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = out_path.with_name(out_path.stem + f"_{stamp}" + out_path.suffix)
        df.to_excel(alt, index=False)
        return alt, e


def compute_pbg_fit_on_corrected_apply_on_before(
    df,
    bg_col,
    tg_before_col,
    tg_corrected_col,
):
    out = df.copy()

    out[bg_col] = pd.to_numeric(out[bg_col], errors="coerce")
    out[tg_before_col] = pd.to_numeric(out[tg_before_col], errors="coerce")
    out[tg_corrected_col] = pd.to_numeric(out[tg_corrected_col], errors="coerce")

    pair_fit = out[[bg_col, tg_corrected_col]].dropna()

    if len(pair_fit) >= 3:
        a, b = np.polyfit(pair_fit[tg_corrected_col].values, pair_fit[bg_col].values, 1)
        out["pBG"] = out[tg_corrected_col] * a + b
        return out, float(a), float(b), int(len(pair_fit))
    else:
        out["pBG"] = np.nan
        return out, np.nan, np.nan, int(len(pair_fit))


def run():
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    files = iter_input_files(IN_DIR)
    if not files:
        print("[ERROR] No input files found.")
        return

    print(f"[INFO] Number of files to process: {len(files)}")

    for p in files:
        print("\n------------------------------")
        print(f"[START] {p.name}")

        try:
            df = pd.read_excel(p, sheet_name=SHEET_NAME)
            df.columns = [" ".join(str(c).strip().split()) for c in df.columns]

            time_col = find_first_col(df, TIME_COL_CANDIDATES)
            bg_col = find_first_col(df, BG_COL_CANDIDATES)
            tg_before_col = find_first_col(df, TG_BEFORE_COL_CANDIDATES)
            tg_corr_col = find_first_col(df, TG_CORR_COL_CANDIDATES)

            if bg_col is None:
                raise ValueError(f"Missing BG column. Candidates={BG_COL_CANDIDATES}; Available={list(df.columns)}")
            if tg_before_col is None:
                raise ValueError(
                    f"Missing TG_before_lag column. Candidates={TG_BEFORE_COL_CANDIDATES}; Available={list(df.columns)}"
                )
            if tg_corr_col is None:
                raise ValueError(
                    f"Missing corrected_TG column. Candidates={TG_CORR_COL_CANDIDATES}; Available={list(df.columns)}"
                )

            df[bg_col] = pd.to_numeric(df[bg_col], errors="coerce") * BG_SCALE

            out, a, b, n_fit = compute_pbg_fit_on_corrected_apply_on_before(
                df,
                bg_col=bg_col,
                tg_before_col=tg_before_col,
                tg_corrected_col=tg_corr_col,
            )

            for col in [bg_col, tg_before_col, tg_corr_col, "pBG"]:
                if col in out.columns:
                    out[col] = pd.to_numeric(out[col], errors="coerce").round(3)

            out_path = OUT_DIR / f"{p.stem}_pBG_only.xlsx"
            saved_path, perm_err = safe_save_excel(out, out_path)
            save_note = "Saved successfully" if perm_err is None else "PermissionError detected; saved with alternative filename"

            msg = (
                f"[DONE] {p.name} | {save_note}\n"
                f"       Saved to: {saved_path}\n"
                f"       pBG: (fit=corrected_TG→BG) a={a if pd.notna(a) else np.nan}, "
                f"b={b if pd.notna(b) else np.nan}, n_fit={n_fit} | (applied to=TG_before_lag)"
            )
            if time_col is not None:
                msg += f"\n       Time column: {time_col}"
            print(msg)

        except Exception as e:
            print(f"[ERROR] {p.name} | {type(e).__name__}: {e}")


if __name__ == "__main__":
    run()