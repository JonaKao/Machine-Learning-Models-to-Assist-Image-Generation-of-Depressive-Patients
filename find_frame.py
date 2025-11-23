#!/usr/bin/env python3
"""
Find the real frames (from multiple CSV files) that best match each generated image
based on a weighted AU similarity score.

For each generated image:
    - take its AU vector
    - compute weighted Euclidean distance to every real frame's AU vector
    - select the top-k closest real frames
    - output a CSV listing those matches

USAGE (example):

    python find_best_matching_frames.py \
        --real_dir "./real_csvs" \
        --generated_csv "./newimgreduced.csv" \
        --out_csv "./best_matches.csv" \
        --top_k 5

You can adjust AU columns and weights in the CONFIG section.
"""

import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =========================
# CONFIG – EDIT THIS PART
# =========================

# AU columns (regression intensities) to use
# Make sure these exist in BOTH real and generated CSVs
DEFAULT_AU_COLUMNS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]

# Weights per AU (others default to 1.0)
DEFAULT_AU_WEIGHTS: Dict[str, float] = {
    "AU01_r": 2.0,
    "AU04_r": 2.0,
    "AU06_r": 2.0,
    "AU07_r": 2.0,
    "AU14_r": 2.0,
    "AU15_r": 3.0,
    "AU23_r": 3.0,
    "AU26_r": 3.0,
}

# Column in generated CSV that identifies each generated image (edit if needed)
DEFAULT_GENERATED_ID_COL = "Participant_ID"  # or "image_id", etc.

# Optional: columns in real CSVs you want to keep as metadata for matches
# (e.g., frame number, original filename, patient ID, etc.)
DEFAULT_REAL_META_COLS = [
    "Participant_ID",   # adjust to your real CSV
    "frame",            # if you have a frame index column
    # add more if you like
]


# =========================
# ARGPARSE
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find best-matching real frames for each generated image using weighted AU similarity."
    )
    parser.add_argument(
        "--real_dir",
        required=True,
        help="Directory containing real CSV files (all *.csv will be loaded).",
    )
    parser.add_argument(
        "--generated_csv",
        required=True,
        help="CSV containing AU vectors for generated images.",
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        help="Output CSV to store best matches.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of best-matching frames to keep per generated image (default: 5).",
    )
    parser.add_argument(
        "--au_columns",
        nargs="+",
        default=DEFAULT_AU_COLUMNS,
        help="AU columns to use (must exist in both real and generated CSVs).",
    )
    parser.add_argument(
        "--generated_id_col",
        default=DEFAULT_GENERATED_ID_COL,
        help=f"ID column name in generated CSV (default: {DEFAULT_GENERATED_ID_COL}).",
    )
    return parser.parse_args()


# =========================
# CORE HELPERS
# =========================

def build_weight_vector(
    au_columns: List[str],
    weight_map: Dict[str, float]
) -> np.ndarray:
    """Create weight vector aligned with au_columns."""
    return np.array([weight_map.get(c, 1.0) for c in au_columns], dtype=float)


def load_all_real_csvs(real_dir: str) -> pd.DataFrame:
    """Load and concatenate all CSV files from real_dir."""
    pattern = os.path.join(real_dir, "*.csv")
    files = glob.glob(pattern)
    if not files:
        raise ValueError(f"No CSV files found in directory: {real_dir}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            df["__source_file"] = os.path.basename(f)  # keep track of origin
            dfs.append(df)
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}", file=sys.stderr)

    if not dfs:
        raise ValueError("No valid CSVs could be loaded from real_dir.")

    real_df = pd.concat(dfs, ignore_index=True)
    return real_df


def check_au_columns(
    df: pd.DataFrame,
    au_columns: List[str],
    label: str
) -> None:
    """Ensure all AU columns exist in df."""
    missing = [c for c in au_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing AU columns in {label} data: {missing}")


def compute_weighted_distance_matrix(
    gen_mat: np.ndarray,
    real_mat: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    Compute weighted Euclidean distance between each generated vector and each real frame.

    gen_mat:  shape (N_gen, D)
    real_mat: shape (N_real, D)
    weights:  shape (D,)

    Returns:
        distances: shape (N_gen, N_real)
    """
    # (N_gen, 1, D) - (1, N_real, D) -> (N_gen, N_real, D)
    diff = gen_mat[:, None, :] - real_mat[None, :, :]
    # Weighted squared diff along last axis
    weighted_sq = (diff ** 2) * weights[None, None, :]
    distances = np.sqrt(np.sum(weighted_sq, axis=2))  # (N_gen, N_real)
    return distances


def normalize_to_similarity(distances: np.ndarray) -> np.ndarray:
    """
    Convert distances to similarity scores in [0, 1].
    sim = 1 - d / d_max (per matrix).
    """
    d_max = np.max(distances)
    if d_max <= 0:
        return np.ones_like(distances)
    sim = 1.0 - distances / d_max
    return np.clip(sim, 0.0, 1.0)


# =========================
# MAIN
# =========================

def main():
    args = parse_args()

    # 1) Load generated data
    try:
        gen_df = pd.read_csv(args.generated_csv)
        gen_df.columns = gen_df.columns.str.strip()
    except Exception as e:
        print(f"Error reading generated CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if args.generated_id_col not in gen_df.columns:
        print(f"Generated ID column '{args.generated_id_col}' not found in generated CSV.", file=sys.stderr)
        sys.exit(1)

    au_columns = args.au_columns
    # Ensure AU columns exist in generated CSV
    check_au_columns(gen_df, au_columns, "generated")

    # 2) Load real data (all CSVs in real_dir)
    try:
        real_df = load_all_real_csvs(args.real_dir)
    except Exception as e:
        print(f"Error loading real CSVs: {e}", file=sys.stderr)
        sys.exit(1)

    check_au_columns(real_df, au_columns, "real")

    # 3) Build matrices
    weights = build_weight_vector(au_columns, DEFAULT_AU_WEIGHTS)

    gen_mat = gen_df[au_columns].to_numpy(dtype=float)  # (N_gen, D)
    real_mat = real_df[au_columns].to_numpy(dtype=float)  # (N_real, D)

    # 4) Compute distances + similarities
    print("Computing distance matrix... (this may take a moment)")
    distances = compute_weighted_distance_matrix(gen_mat, real_mat, weights)  # (N_gen, N_real)
    similarities = normalize_to_similarity(distances)

    # 5) For each generated image, pick top-k real frames
    top_k = max(1, args.top_k)
    n_gen, n_real = distances.shape
    all_matches = []

    real_meta_cols = [c for c in DEFAULT_REAL_META_COLS if c in real_df.columns]
    meta_cols_to_keep = ["__source_file"] + real_meta_cols

    for i in range(n_gen):
        gen_id = gen_df.iloc[i][args.generated_id_col]
        d_row = distances[i]      # shape (N_real,)
        s_row = similarities[i]   # shape (N_real,)

        # indices of top-k smallest distances
        idx = np.argsort(d_row)[:top_k]

        for rank, j in enumerate(idx):
            match = {
                "generated_index": i,
                "generated_id": gen_id,
                "rank": rank + 1,
                "distance": float(d_row[j]),
                "similarity": float(s_row[j]),
            }
            # Attach real metadata
            for col in meta_cols_to_keep:
                match[f"real_{col}"] = real_df.iloc[j][col]

            all_matches.append(match)

    matches_df = pd.DataFrame(all_matches)

    # 6) Save
    try:
        matches_df.to_csv(args.out_csv, index=False)
    except Exception as e:
        print(f"Error writing output CSV: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Done. Saved best matches to: {args.out_csv}")
    print(f"Rows: {len(matches_df)} (N_gen * top_k = {n_gen} * {top_k})")


if __name__ == "__main__":
    main()
