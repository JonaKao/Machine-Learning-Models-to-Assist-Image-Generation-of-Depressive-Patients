#!/usr/bin/env python3
import os, re
import pandas as pd

CSV_PATH = r"C:\Users\jonam\OneDrive\Desktop\Coding\BAProper\OpenFace\results\Depressive Class.csv"
CONFIDENCE_THRESHOLD = 0.9

AU_PAT = re.compile(r'^AU(\d{2})_(r|c)$', re.IGNORECASE)

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = []
    for c in df.columns:
        if isinstance(c, str):
            c2 = c.replace('\ufeff', '')  # remove BOM
            c2 = c2.strip()
            c2 = re.sub(r'\s+', ' ', c2)
            cleaned.append(c2)
        else:
            cleaned.append(c)
    df.columns = cleaned
    # Also remove all whitespace entirely
    no_ws_map = {c: re.sub(r'\s+', '', c) if isinstance(c, str) else c for c in df.columns}
    df = df.rename(columns=no_ws_map)
    return df

def extract_per_frame(csv_path, conf_thresh=0.9):
    df = pd.read_csv(csv_path)
    df = normalize_headers(df)

    # Apply quality filter
    if "success" in df and "confidence" in df:
        df = df[(df["success"] == 1) & (df["confidence"] >= conf_thresh)]

    if df.empty:
        print(f"[WARN] No valid frames after filtering in: {csv_path}")
        return None

    # Keep only AU columns plus frame index
    au_cols = [c for c in df.columns if isinstance(c, str) and AU_PAT.match(c)]
    keep_cols = ["frame"] + au_cols

    if not au_cols:
        print("[WARN] No AU columns found. Writing only frame numbers.")
        keep_cols = ["frame"]

    return df[keep_cols].copy()

if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(CSV_PATH), "au_features_per_frame.csv")
    df_out = extract_per_frame(CSV_PATH, conf_thresh=CONFIDENCE_THRESHOLD)
    if df_out is None:
        print("[INFO] No output written.")
    else:
        df_out.to_csv(out_path, index=False)
        print(f"[OK] Wrote per-frame AU features → {out_path}")
        print(f"[INFO] Rows: {len(df_out)}, Cols: {len(df_out.columns)}")
