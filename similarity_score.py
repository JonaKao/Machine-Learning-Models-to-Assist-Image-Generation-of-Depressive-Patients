#!/usr/bin/env python3
"""
Compute a weighted AU similarity score between real and generated images.

Inputs:
    - CSV with real AU vectors
    - CSV with generated AU vectors
    - Both must contain:
        * an ID column (same name in both files)
        * the same AU columns (e.g., AU01_r, AU04_r, ...)

Output:
    - A CSV with per-pair similarity scores
    - Printed summary statistics (mean, std, min, max)

Adapt the CONFIG section to your setup (paths, ID column, AU columns, weights).
"""

import argparse
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =========================
# CONFIG – EDIT THIS PART
# =========================

# 1) Default AU columns (example; adjust to your OpenFace output)
#    For OpenFace, AU columns often look like: AU01_r, AU04_r, ...
DEFAULT_AU_COLUMNS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"
]

# 2) Weights for each AU (edit as needed).
#    Any AU not listed here will default to weight 1.0.
#    Example: emphasize your key AUs (1, 4, 15, 23, 26, 6, 7, 14).
DEFAULT_AU_WEIGHTS: Dict[str, float] = {
    "AU01_r": 2.0,
    "AU04_r": 2.0,
    "AU06_r": 2.0,
    "AU07_r": 2.0,
    "AU14_r": 2.0,
    "AU15_r": 3.0,
    "AU23_r": 3.0,
    "AU26_r": 3.0,
    # others will implicitly get weight 1.0
}

# 3) Default ID column name (must exist in both CSVs)
DEFAULT_ID_COLUMN = "image_id"


# =========================
# CORE LOGIC
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute weighted AU similarity between real and generated images."
    )
    parser.add_argument(
        "--real_csv",
        required=True,
        help="Path to CSV with real images' AU vectors."
    )
    parser.add_argument(
        "--generated_csv",
        required=True,
        help="Path to CSV with generated images' AU vectors."
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        help="Path to write CSV with per-image similarity scores."
    )
    parser.add_argument(
        "--id_column",
        default=DEFAULT_ID_COLUMN,
        help=f"Name of ID column present in both CSVs (default: {DEFAULT_ID_COLUMN})."
    )
    parser.add_argument(
        "--au_columns",
        nargs="+",
        default=DEFAULT_AU_COLUMNS,
        help="List of AU columns to use. If not set, uses DEFAULT_AU_COLUMNS in script."
    )
    return parser.parse_args()


def build_weight_vector(
    au_columns: List[str],
    weight_map: Dict[str, float]
) -> np.ndarray:
    """
    Create a weight vector aligned with au_columns.
    Any AU not in weight_map gets weight 1.0.
    """
    weights = []
    for au in au_columns:
        w = weight_map.get(au, 1.0)
        weights.append(w)
    return np.array(weights, dtype=float)


def align_and_merge(
    real_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    id_column: str,
    au_columns: List[str]
) -> pd.DataFrame:
    """
    Inner-join real and generated dataframes on id_column.
    Ensures all required AU columns are present in both.
    """
    missing_real = [c for c in au_columns if c not in real_df.columns]
    missing_gen = [c for c in au_columns if c not in gen_df.columns]

    if missing_real:
        raise ValueError(f"The following AU columns are missing in real CSV: {missing_real}")
    if missing_gen:
        raise ValueError(f"The following AU columns are missing in generated CSV: {missing_gen}")
    if id_column not in real_df.columns:
        raise ValueError(f"ID column '{id_column}' not found in real CSV.")
    if id_column not in gen_df.columns:
        raise ValueError(f"ID column '{id_column}' not found in generated CSV.")

    merged = pd.merge(
        real_df[[id_column] + au_columns],
        gen_df[[id_column] + au_columns],
        on=id_column,
        suffixes=("_real", "_gen"),
        how="inner",
    )

    if merged.empty:
        raise ValueError("No matching IDs found between real and generated CSVs.")

    return merged


def compute_weighted_distances(
    merged: pd.DataFrame,
    au_columns: List[str],
    weights: np.ndarray
) -> np.ndarray:
    """
    Compute weighted Euclidean distance in AU space for each row.
    distance_i = sqrt(sum_k w_k * (AU_real_k - AU_gen_k)^2)
    """
    # Build matrices for real and generated AUs in same order
    real_cols = [f"{c}_real" for c in au_columns]
    gen_cols = [f"{c}_gen" for c in au_columns]

    real_mat = merged[real_cols].to_numpy(dtype=float)
    gen_mat = merged[gen_cols].to_numpy(dtype=float)

    # shape: (N, num_AUs)
    diff = real_mat - gen_mat
    # Apply weights: (diff^2 * w)
    weighted_sq = (diff ** 2) * weights  # broadcasting over rows
    distances = np.sqrt(np.sum(weighted_sq, axis=1))

    return distances


def normalize_to_similarity(distances: np.ndarray) -> np.ndarray:
    """
    Convert distances to similarity scores in [0, 1]:
    sim_i = 1 - distance_i / D_max, where D_max is max distance.
    If all distances are zero, similarity is 1.0 for all.
    """
    d_max = distances.max()
    if d_max <= 0:
        # All identical
        return np.ones_like(distances)
    similarity = 1.0 - distances / d_max
    # Numerical safety
    similarity = np.clip(similarity, 0.0, 1.0)
    return similarity


def summarize_similarity(similarity: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Return (mean, std, min, max) of similarity scores.
    """
    return (
        float(np.mean(similarity)),
        float(np.std(similarity)),
        float(np.min(similarity)),
        float(np.max(similarity)),
    )


def main():
    args = parse_args()

    # Load data
    try:
        real_df = pd.read_csv(args.real_csv)
        gen_df = pd.read_csv(args.generated_csv)
        real_df.columns = real_df.columns.str.strip()
        gen_df.columns = gen_df.columns.str.strip()
    except Exception as e:
        print(f"Error reading CSVs: {e}", file=sys.stderr)
        sys.exit(1)

    au_columns = args.au_columns
    weight_vec = build_weight_vector(au_columns, DEFAULT_AU_WEIGHTS)

    try:
        merged = align_and_merge(real_df, gen_df, args.id_column, au_columns)
    except ValueError as e:
        print(f"Error aligning data: {e}", file=sys.stderr)
        sys.exit(1)

    # Compute weighted distances and similarities
    distances = compute_weighted_distances(merged, au_columns, weight_vec)
    similarity = normalize_to_similarity(distances)

    merged_out = merged[[args.id_column]].copy()
    merged_out["weighted_au_distance"] = distances
    merged_out["weighted_au_similarity"] = similarity

    # Save output
    try:
        merged_out.to_csv(args.out_csv, index=False)
    except Exception as e:
        print(f"Error writing output CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary
    mean_sim, std_sim, min_sim, max_sim = summarize_similarity(similarity)
    print("Weighted AU similarity summary")
    print("------------------------------")
    print(f"N pairs          : {len(similarity)}")
    print(f"Mean similarity  : {mean_sim:.4f}")
    print(f"Std similarity   : {std_sim:.4f}")
    print(f"Min similarity   : {min_sim:.4f}")
    print(f"Max similarity   : {max_sim:.4f}")
    print()
    print(f"Per-image scores saved to: {args.out_csv}")


if __name__ == "__main__":
    main()
