import argparse, os, glob
import pandas as pd
import numpy as np

def is_au_r(col): return col.startswith("AU") and col.endswith("_r")
def is_au_c(col): return col.startswith("AU") and col.endswith("_c")
def summarize_file(csv_path, conf_thresh=0.9):
    df = pd.read_csv(csv_path)
    # Quality filter
    if "success" in df and "confidence" in df:
        df = df[(df["success"] == 1) & (df["confidence"] >= conf_thresh)]
    if df.empty:
        return None  # no valid frames

    au_r_cols = [c for c in df.columns if is_au_r(c)]
    au_c_cols = [c for c in df.columns if is_au_c(c)]

    feats = {"source_csv": os.path.basename(csv_path), "n_frames": len(df)}

    # AU intensities: mean, std, median
    for c in au_r_cols:
        feats[f"{c}_mean"]   = float(np.nanmean(df[c].values))
        feats[f"{c}_std"]    = float(np.nanstd(df[c].values))
        feats[f"{c}_median"] = float(np.nanmedian(df[c].values))

    # AU presence: proportion of frames where AU present
    for c in au_c_cols:
        vals = df[c].values.astype(float)
        feats[f"{c}_prop"] = float(np.nanmean(vals))  # 0..1

    # Optional: simple dynamics (frame-to-frame variability) for intensities
    for c in au_r_cols:
        series = df[c].values
        if len(series) > 1:
            diffs = np.diff(series)
            feats[f"{c}_diff_std"] = float(np.nanstd(diffs))
        else:
            feats[f"{c}_diff_std"] = np.nan

    return feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, help="Folder with OpenFace CSVs")
    ap.add_argument("--out_csv", required=True, help="Where to save the AU features table")
    ap.add_argument("--confidence", type=float, default=0.9, help="Min confidence to keep a frame")
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.results_dir, "*.csv")))
    if not csvs:
        raise SystemExit(f"No CSVs found in {args.results_dir}")

    rows = []
    for p in csvs:
        feats = summarize_file(p, conf_thresh=args.confidence)
        if feats is not None:
            rows.append(feats)

    if not rows:
        raise SystemExit("No valid frames across files after filtering. Try lowering --confidence.")

    df = pd.DataFrame(rows).sort_values("source_csv")
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote AU features → {args.out_csv}")
    print(f"[INFO] {len(df)} files summarized, {df.shape[1]-2} feature columns")

if __name__ == "__main__":
    main()
