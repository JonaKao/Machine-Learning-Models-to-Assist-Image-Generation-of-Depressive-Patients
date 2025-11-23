#!/usr/bin/env python3
import os, re
import pandas as pd
import numpy as np

CSV_PATH = r"C:\Users\jonam\OneDrive\Desktop\Coding\BAProper\OpenFace\results\Depressive Class.csv"
CONFIDENCE_THRESHOLD = 0.9

# Regex for AU columns, ignore case, allow exactly AU + 2 digits + _r/_c
AU_PAT = re.compile(r'^AU(\d{2})_(r|c)$', re.IGNORECASE)

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    # Original names
    orig = list(df.columns)
    # Cleaned (strip, remove BOM, collapse internal spaces)
    cleaned = []
    for c in orig:
        if isinstance(c, str):
            c2 = c.replace('\ufeff', '')  # BOM
            c2 = c.strip()
            c2 = re.sub(r'\s+', ' ', c2)  # collapse internal spaces
            cleaned.append(c2)
        else:
            cleaned.append(c)
    df.columns = cleaned

    # Also create a mapping that removes ALL whitespace entirely for a second pass
    # e.g., " AU01_r " -> "AU01_r"
    no_ws_map = {c: re.sub(r'\s+', '', c) if isinstance(c, str) else c for c in df.columns}
    df = df.rename(columns=no_ws_map)
    return df

def get_cols(df, base):
    # return first matching column for any reasonable variant of a name
    variants = [base, base.strip(), base.strip().lower()]
    lower_map = {c.lower(): c for c in df.columns if isinstance(c, str)}
    for v in variants:
        if isinstance(v, str) and v in df.columns:
            return v
        if isinstance(v, str) and v.lower() in lower_map:
            return lower_map[v.lower()]
    return None

def summarize_file(csv_path, conf_thresh=0.9):
    df = pd.read_csv(csv_path)
    df = normalize_headers(df)

    # Apply quality filter if present
    success_col = get_cols(df, "success")
    conf_col    = get_cols(df, "confidence")
    if success_col and conf_col:
        df = df[(df[success_col] == 1) & (df[conf_col] >= conf_thresh)]

    if df.empty:
        print(f"[WARN] No valid frames after filtering in: {csv_path}")
        return None

    # Identify AU columns robustly (after normalization)
    au_cols = [c for c in df.columns if isinstance(c, str) and AU_PAT.match(c)]
    au_r_cols = [c for c in au_cols if c.lower().endswith("_r")]
    au_c_cols = [c for c in au_cols if c.lower().endswith("_c")]

    # Diagnostics
    au_like = [c for c in df.columns if isinstance(c, str) and 'au' in c.lower()]
    print(f"[INFO] Frames kept: {len(df)}")
    print(f"[INFO] AU-like columns found: {len(au_like)} | AU matched: {len(au_cols)} "
          f"(r:{len(au_r_cols)}, c:{len(au_c_cols)})")
    # Write a debug file so we can see what headers exist on your machine
    dbg_path = os.path.join(os.path.dirname(csv_path), "au_debug_columns.txt")
    with open(dbg_path, "w", encoding="utf-8") as f:
        f.write("ALL COLUMNS:\n")
        for c in df.columns:
            f.write(repr(c) + "\n")
        f.write("\nAU-LIKE COLUMNS:\n")
        for c in au_like:
            f.write(repr(c) + "\n")
    print(f"[INFO] Wrote column debug → {dbg_path}")

    feats = {"source_csv": os.path.basename(csv_path), "n_frames": int(len(df))}

    # AU intensities: mean/std/median + simple dynamics
    for c in au_r_cols:
        vals = pd.to_numeric(df[c], errors="coerce").values
        feats[f"{c}_mean"]   = float(np.nanmean(vals))
        feats[f"{c}_std"]    = float(np.nanstd(vals))
        feats[f"{c}_median"] = float(np.nanmedian(vals))
        feats[f"{c}_diff_std"] = float(np.nanstd(np.diff(vals))) if len(vals) > 1 else np.nan

    # AU presence: proportion active
    for c in au_c_cols:
        vals = pd.to_numeric(df[c], errors="coerce").astype(float).values
        feats[f"{c}_prop"] = float(np.nanmean(vals))

    return feats

if __name__ == "__main__":
    feats = summarize_file(CSV_PATH, conf_thresh=CONFIDENCE_THRESHOLD)
    out_path = os.path.join(os.path.dirname(CSV_PATH), "au_features_test.csv")
    if feats is None:
        print("[INFO] Nothing to write (no valid frames).")
    else:
        pd.DataFrame([feats]).to_csv(out_path, index=False)
        print(f"[OK] Wrote AU features → {out_path}")
