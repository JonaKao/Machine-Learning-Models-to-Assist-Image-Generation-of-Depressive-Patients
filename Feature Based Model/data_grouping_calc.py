#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate OpenFace features by PHQ-8 severity groups (DAIC-WOZ style structure).

Steps:
1) Read metadata CSV with columns: Participant_ID, PHQ_Score
2) For each participant:
   - Locate CSV at <data_root>/<XXX_P>/features/<XXX>_OpenFace2.1.0_Pose_gaze_AUs.csv
   - Compute mean of all numeric columns EXCLUDING ["frame", "timestamp", "confidence", "success"]
3) Save per-patient means to per_patient_means.csv
4) Map PHQ_Score to PHQ category and compute group-wise means across patients
5) Save group means to group_means.csv

Raises:
- FileNotFoundError if any expected patient folder/file is missing
- ValueError if PHQ_Score is missing or out of [0, 24]
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

EXCLUDED_COLS = {"frame", "timestamp", "confidence", "success"}

PHQ_BINS = [
    (0, 4, "Real images with PHQ-8 score 0-4 (no depression)"),
    (5, 9, "Real images with PHQ-8 score 5-9 (mild depression)"),
    (10, 14, "Real images with PHQ-8 score 10-14 (moderate depression)"),
    (15, 19, "Real images with PHQ-8 score 15-19 (moderately severe depression)"),
    (20, 24, "Real images with PHQ-8 score 20-24 (severe depression)"),
]

CATEGORY_ORDER = [label for _, _, label in PHQ_BINS]


def phq_to_category(score):
    if pd.isna(score):
        raise ValueError("Encountered NaN PHQ_Score in metadata.")
    try:
        s = float(score)
    except Exception:
        raise ValueError(f"PHQ_Score '{score}' is not numeric.")
    if s < 0 or s > 24:
        raise ValueError(f"PHQ_Score {s} is out of the expected range [0, 24].")
    for lo, hi, label in PHQ_BINS:
        if lo <= s <= hi:
            return label
    raise ValueError(f"PHQ_Score {s} did not match any category bin.")


def compute_patient_means(csv_path: Path, verbose: bool = True) -> pd.Series:
    if verbose:
        print(f"  - Reading {csv_path}")
    df = pd.read_csv(csv_path)

    drop_these = [c for c in df.columns if c in EXCLUDED_COLS]
    if drop_these and verbose:
        print(f"    > Dropping excluded columns: {drop_these}")
    df = df.drop(columns=drop_these, errors="ignore")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        raise ValueError(f"No numeric columns found in {csv_path.name} after excluding {EXCLUDED_COLS}.")

    means = df[num_cols].mean(numeric_only=True)
    return means


def main():
    parser = argparse.ArgumentParser(description="Aggregate OpenFace features by PHQ-8 severity groups.")
    parser.add_argument("--metadata_csv", required=True,
                        help="Path to metadata CSV with columns: Participant_ID, PHQ_Score (e.g., metadata_mapped.csv)")
    parser.add_argument("--data_root", required=True,
                        help=r"Root folder that contains one folder per patient (e.g., C:\...\DAIC-WOZ Data\data)")
    parser.add_argument("--outdir", default="outputs",
                        help="Output directory where per_patient_means.csv and group_means.csv will be saved.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    metadata_csv = Path(args.metadata_csv)
    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    verbose = args.verbose

    if verbose:
        print("=== Configuration ===")
        print(f"metadata_csv: {metadata_csv}")
        print(f"data_root:    {data_root}")
        print(f"outdir:       {outdir}")
        print(f"verbose:      {verbose}")
        print("=====================")

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    outdir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"> Output directory ensured: {outdir.resolve()}")

    if verbose:
        print(f"> Loading metadata from {metadata_csv}")
    meta = pd.read_csv(metadata_csv)

    required_cols = {"Participant_ID", "PHQ_Score"}
    missing_cols = required_cols - set(meta.columns)
    if missing_cols:
        raise ValueError(f"Metadata CSV missing required columns: {sorted(missing_cols)}")

    rows = []
    missing_anything = []

    for idx, (pid, phq) in enumerate(meta[["Participant_ID", "PHQ_Score"]].itertuples(index=False, name=None), start=1):
        if verbose:
            print(f"\n[{idx}/{len(meta)}] Processing Participant_ID={pid} (PHQ_Score={phq})")

        category = phq_to_category(phq)

        # UPDATED: include the 'features' subfolder
        folder_name = f"{int(pid)}_P"
        file_name = f"{int(pid)}_OpenFace2.1.0_Pose_gaze_AUs.csv"
        patient_folder = data_root / folder_name
        patient_file = patient_folder / "features" / file_name  # <- change here

        if not patient_folder.exists():
            missing_anything.append(f"Missing folder: {patient_folder}")
        elif not patient_file.exists():
            missing_anything.append(f"Missing file:   {patient_file}")
        else:
            if verbose:
                print(f"  Folder OK: {patient_folder}")
                print(f"  File   OK: {patient_file}")

            means = compute_patient_means(patient_file, verbose=verbose)
            row = {
                "Participant_ID": int(pid),
                "PHQ_Score": float(phq),
                "PHQ_Category": category,
            }
            for c, v in means.items():
                row[c] = v
            rows.append(row)

    if missing_anything:
        msg = "\n".join(missing_anything)
        raise FileNotFoundError(
            "One or more expected folders/files were not found.\n"
            "Details:\n" + msg
        )

    if verbose:
        print("\n> Compiling per-patient DataFrame...")
    per_patient_df = pd.DataFrame(rows)

    per_patient_csv = outdir / "per_patient_means.csv"
    if verbose:
        print(f"> Saving per-patient means to: {per_patient_csv}")
    per_patient_df.to_csv(per_patient_csv, index=False)

    if verbose:
        print("> Computing group-wise means...")
    numeric_cols = per_patient_df.select_dtypes(include="number").columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in {"Participant_ID", "PHQ_Score"}]
    if not feature_cols:
        raise ValueError("No feature columns found to aggregate in per_patient_means.csv.")

    grouped = (
        per_patient_df
        .groupby("PHQ_Category", sort=False)[feature_cols]
        .mean(numeric_only=True)
        .reindex(CATEGORY_ORDER)
    )

    group_means_csv = outdir / "group_means.csv"
    if verbose:
        print(f"> Saving group means to: {group_means_csv}")
    grouped.to_csv(group_means_csv, index=True)

    if verbose:
        print("\nAll done! ✅")
        print(f"- Per-patient means: {per_patient_csv.resolve()}")
        print(f"- Group-wise means:  {group_means_csv.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nERROR:", str(e), file=sys.stderr)
        raise
