#!/usr/bin/env python3
"""
Gentle AU-based matching between generated and real data.

- Compares every generated AU vector to every real AU vector.
- Uses weighted Euclidean distance in AU space.
- Normalizes similarity PER GENERATED ROW so that:
    best match for each generated -> similarity = 1.0
    worst match for each generated -> similarity ≈ 0.0
- Fills missing AU values with column means (or 0 if all missing).
- Outputs top_k best matches per generated sample.

Usage (PowerShell example):

    python gentle_match.py `
        --real_csv ".\\per_patient_means.csv" `
        --generated_csv ".\\newimgreduced.csv" `
        --out_csv ".\\best_matches_gentle.csv" `
        --top_k 5
"""

import argparse
import sys
from typing import Dict, List

import numpy as np
import pandas as pd


# =========================
# CONFIG – EDIT THIS PART
# =========================

# AU columns (regression intensities) to use.
# Make sure these exist in BOTH CSVs. The script will intersect if some are missing.
PREFERRED_AU_COLUMNS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]

# Weights per AU (others default to 1.0).
AU_WEIGHTS: Dict[str, float] = {
    "AU01_r": 2.0,
    "AU04_r": 2.0,
    "AU06_r": 2.0,
    "AU07_r": 2.0,
    "AU14_r": 2.0,
    "AU15_r": 3.0,
    "AU23_r": 3.0,
    "AU26_r": 3.0,
}

# Column in generated CSV that identifies each generated sample.
GENERATED_ID_COL_DEFAULT = "Participant_ID"  # change if needed

# Optional: columns in real CSV to keep as metadata in output.
REAL_META_COLS_DEFAULT = [
    "Participant_ID",  # adjust/remove if not present
    "frame",           # adjust/remove if not present
]


# =========================
# ARGUMENT PARSING
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gentle weighted-AU matching between generated and real CSVs."
    )
    parser.add_argument(
        "--real_csv",
        required=True,
        help="CSV with real AU vectors.",
    )
    parser.add_argument(
        "--generated_csv",
        required=True,
        help="CSV with generated AU vectors.",
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        help="Output CSV for best matches.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of best matches per generated row (default: 5).",
    )
    parser.add_argument(
        "--generated_id_col",
        default=GENERATED_ID_COL_DEFAULT,
        help=f"ID column name in generated CSV (default: {GENERATED_ID_COL_DEFAULT}).",
    )
    return parser.parse_args()


# =========================
# HELPERS
# =========================

def choose_au_columns(real_df: pd.DataFrame, gen_df: pd.DataFrame) -> List[str]:
    """Pick AU columns that exist in BOTH real and generated CSVs."""
    real_cols = set(real_df.columns)
    gen_cols = set(gen_df.columns)
    common = [c for c in PREFERRED_AU_COLUMNS if c in real_cols and c in gen_cols]
    if not common:
        raise ValueError("No common AU columns found between real and generated CSVs.")
    return common


def build_weight_vector(au_columns: List[str]) -> np.ndarray:
    """Create weight vector aligned with au_columns."""
    return np.array([AU_WEIGHTS.get(c, 1.0) for c in au_columns], dtype=float)


def fill_au_nans(df: pd.DataFrame, au_columns: List[str], label: str) -> pd.DataFrame:
    """Fill NaNs in AU columns with column mean (or 0 if all NaN)."""
    df = df.copy()
    for col in au_columns:
        if col not in df.columns:
            raise ValueError(f"{label} CSV is missing AU column: {col}")
        if df[col].isna().any():
            mean_val = df[col].mean()
            if pd.isna(mean_val):
                mean_val = 0.0
            df[col] = df[col].fillna(mean_val)
    return df


def compute_weighted_distance_matrix(
    gen_mat: np.ndarray,
    real_mat: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Weighted Euclidean distance between each generated vector and each real vector.

    gen_mat:  (N_gen, D)
    real_mat: (N_real, D)
    weights:  (D,)
    """
    diff = gen_mat[:, None, :] - real_mat[None, :, :]   # (N_gen, N_real, D)
    weighted_sq = (diff ** 2) * weights[None, None, :]  # broadcast weights
    distances = np.sqrt(np.sum(weighted_sq, axis=2))    # (N_gen, N_real)
    return distances


def normalize_to_similarity(distances: np.ndarray) -> np.ndarray:
    """
    Gentle per-generated normalization:

    For each generated row i:
        d_min_i = min_j d_ij
        d_max_i = max_j d_ij

        sim_ij = 1 - (d_ij - d_min_i) / (d_max_i - d_min_i)

    => best match for each generated row has similarity = 1.0
       worst match has similarity = 0.0
    """
    d_min = np.nanmin(distances, axis=1, keepdims=True)  # (N_gen, 1)
    d_max = np.nanmax(distances, axis=1, keepdims=True)  # (N_gen, 1)

    denom = d_max - d_min
    denom[denom == 0] = 1.0  # avoid division by zero

    sim = 1.0 - (distances - d_min) / denom
    sim = np.clip(sim, 0.0, 1.0)
    # Replace any remaining NaNs with 0.0 (shouldn't normally happen after NaN filling)
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    return sim


# =========================
# MAIN
# =========================

def main():
    args = parse_args()

    # 1) Load CSVs and clean column names
    try:
        real_df = pd.read_csv(args.real_csv)
        gen_df = pd.read_csv(args.generated_csv)
        real_df.columns = real_df.columns.str.strip()
        gen_df.columns = gen_df.columns.str.strip()
    except Exception as e:
        print(f"Error reading CSVs: {e}", file=sys.stderr)
        sys.exit(1)

    # 2) Check generated ID column
    if args.generated_id_col not in gen_df.columns:
        print(f"Generated ID column '{args.generated_id_col}' not found.", file=sys.stderr)
        print("Available columns in generated CSV:", gen_df.columns.tolist(), file=sys.stderr)
        sys.exit(1)

    # 3) Choose AU columns that exist in both
    try:
        au_columns = choose_au_columns(real_df, gen_df)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    print("Using AU columns:", au_columns)

    # 4) Fill NaNs in AU columns to avoid NaN distances/similarities
    real_df = fill_au_nans(real_df, au_columns, label="real")
    gen_df = fill_au_nans(gen_df, au_columns, label="generated")

    # 5) Build matrices
    weights = build_weight_vector(au_columns)
    gen_mat = gen_df[au_columns].to_numpy(dtype=float)
    real_mat = real_df[au_columns].to_numpy(dtype=float)

    n_gen, dim = gen_mat.shape
    n_real, _ = real_mat.shape
    print(f"Generated samples: {n_gen}, Real samples: {n_real}, AU dim: {dim}")

    if n_gen == 0 or n_real == 0:
        print("No data to compare (empty CSV?).", file=sys.stderr)
        sys.exit(1)

    # 6) Distances and similarities
    print("Computing distance matrix...")
    distances = compute_weighted_distance_matrix(gen_mat, real_mat, weights)
    similarities = normalize_to_similarity(distances)

    # sanity check: no NaNs
    print("Distance matrix stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        float(np.nanmin(distances)),
        float(np.nanmax(distances)),
        float(np.nanmean(distances)),
    ))
    print("Similarity matrix stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        float(np.nanmin(similarities)),
        float(np.nanmax(similarities)),
        float(np.nanmean(similarities)),
    ))

    # 7) For each generated sample, pick top_k best real samples
    top_k = max(1, args.top_k)
    all_rows = []

    real_meta_cols = [c for c in REAL_META_COLS_DEFAULT if c in real_df.columns]

    for i in range(n_gen):
        gen_id = gen_df.iloc[i][args.generated_id_col]
        d_row = distances[i]      # (N_real,)
        s_row = similarities[i]   # (N_real,)

        # indices of top-k smallest distances
        idx = np.argsort(d_row)[:top_k]

        for rank, j in enumerate(idx):
            row = {
                "generated_index": i,
                "generated_id": gen_id,
                "rank": rank + 1,
                "distance": float(d_row[j]),
                "similarity": float(s_row[j]),
            }
            # attach real metadata
            for col in real_meta_cols:
                row[f"real_{col}"] = real_df.iloc[j][col]
            all_rows.append(row)

    matches_df = pd.DataFrame(all_rows)

    try:
        matches_df.to_csv(args.out_csv, index=False)
    except Exception as e:
        print(f"Error writing output CSV: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Done. Saved best matches to: {args.out_csv}")
    print(f"Rows: {len(matches_df)} (should be n_gen * top_k = {n_gen} * {top_k})")


if __name__ == "__main__":
    main()
