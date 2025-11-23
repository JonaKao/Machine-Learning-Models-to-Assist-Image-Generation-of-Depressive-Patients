import os
import argparse
import json
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
from glob import glob

def normalize_pid(pid_val) -> str:
    """Normalize IDs e.g. '713.0' -> '713'."""
    try:
        return str(int(float(str(pid_val).strip())))
    except Exception:
        return str(pid_val).strip()

def read_split_csv(path: str, split_name: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[{split_name}] Missing split CSV: {path}")
    df = pd.read_csv(path)
    if "Participant_ID" not in df.columns:
        raise ValueError(f"[{split_name}] Split file must contain 'Participant_ID'")
    df["Participant_ID"] = df["Participant_ID"].apply(normalize_pid)
    return df[["Participant_ID"]].drop_duplicates()

def expected_openface_path(data_root: str, pid: str, subdir: str,
                           filename_template: str, pid_suffix: str) -> str:
    """Build .../{pid}{pid_suffix}/{subdir}/{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv"""
    part_dir = os.path.join(data_root, f"{pid}{pid_suffix}", subdir)
    fname = filename_template.format(pid=pid)
    return os.path.join(part_dir, fname)

def find_openface_fallback(data_root: str, pid: str, pid_suffix: str) -> str:
    """Loose glob if exact file missing, within the pid folder."""
    part_dir = os.path.join(data_root, f"{pid}{pid_suffix}")
    pats = [
        os.path.join(part_dir, "**", f"{pid}_OpenFace*.csv"),
        os.path.join(part_dir, "**", "*OpenFace*Pose*gaze*AU*.csv"),
        os.path.join(part_dir, "**", "*openface*.csv"),
    ]
    for pat in pats:
        hits = glob(pat, recursive=True)
        if hits:
            hits.sort(key=lambda p: (p.count(os.sep), len(p)))
            return hits[0]
    return ""

def load_openface_csv(path: str, pid: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.insert(0, "Participant_ID", pid)
    return df

def merge_split(data_root: str, split_df: pd.DataFrame, subdir: str,
                filename_template: str, pid_suffix: str,
                verbose: bool = True, loose_match: bool = False,
                debug_first: int = 0) -> Tuple[pd.DataFrame, List[str], List[str]]:
    merged_frames: List[pd.DataFrame] = []
    included, skipped = [], []
    ids = split_df["Participant_ID"].tolist()

    for idx, pid in enumerate(ids):
        pid = normalize_pid(pid)
        part_dir = os.path.join(data_root, f"{pid}{pid_suffix}")
        exp_path = expected_openface_path(data_root, pid, subdir, filename_template, pid_suffix)

        if debug_first and idx < debug_first:
            print(f"[debug] PID={pid} part_dir={part_dir} exists_dir={os.path.isdir(part_dir)}")
            print(f"[debug] expected={exp_path} exists_file={os.path.isfile(exp_path)}")

        if not os.path.isdir(part_dir):
            skipped.append(pid)
            if verbose:
                print(f"[warn] Participant dir not found: {part_dir}")
            continue

        of_path = exp_path if os.path.isfile(exp_path) else ""
        if not of_path and loose_match:
            of_path = find_openface_fallback(data_root, pid, pid_suffix)
            if debug_first and idx < debug_first:
                print(f"[debug] fallback -> {of_path if of_path else 'NO MATCH'}")

        if not of_path:
            skipped.append(pid)
            if verbose:
                print(f"[warn] OpenFace CSV missing for {pid}: {exp_path}")
            continue

        try:
            df = load_openface_csv(of_path, pid)
            merged_frames.append(df)
            included.append(pid)
        except Exception as e:
            skipped.append(pid)
            if verbose:
                print(f"[warn] Failed to read OpenFace for {pid} at {of_path}: {e}")

    merged_df = pd.concat(merged_frames, axis=0, ignore_index=True) if merged_frames else pd.DataFrame(columns=["Participant_ID"])
    return merged_df, included, skipped

def main():
    parser = argparse.ArgumentParser(description="Merge OpenFace features per split (keeps originals untouched).")
    parser.add_argument("--data_root", required=True,
                        help="Path to dataset root that contains 'data/' or the participant folders directly.")
    parser.add_argument("--labels_dir", required=True,
                        help="Folder with train_split.csv, dev_split.csv, test_split.csv")
    parser.add_argument("--out_dir", default="features_merged_openface",
                        help="Where to write merged CSVs and manifests")
    parser.add_argument("--subdir", default="features", help="Subfolder under each participant (e.g., 'features')")
    parser.add_argument("--filename_template", default="{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv",
                        help="Use {pid} as placeholder for the participant ID")
    parser.add_argument("--pid_suffix", default="_P", help="Suffix on participant folder names (e.g., '_P')")
    parser.add_argument("--loose_match", action="store_true", help="Fallback to glob search if exact file missing")
    parser.add_argument("--debug_first", type=int, default=5, help="Print debug info for first N participants")
    parser.add_argument("--no_verbose", action="store_true")
    args = parser.parse_args()

    verbose = not args.no_verbose
    os.makedirs(args.out_dir, exist_ok=True)

    # Read splits
    train_split = read_split_csv(os.path.join(args.labels_dir, "train_split.csv"), "train")
    dev_split   = read_split_csv(os.path.join(args.labels_dir, "dev_split.csv"), "dev")
    test_split  = read_split_csv(os.path.join(args.labels_dir, "test_split.csv"), "test")

    print("[info] Processing TRAIN split...")
    train_df, train_included, train_skipped = merge_split(
        args.data_root, train_split, args.subdir, args.filename_template, args.pid_suffix,
        verbose, args.loose_match, args.debug_first)

    print("[info] Processing DEV split...")
    dev_df, dev_included, dev_skipped = merge_split(
        args.data_root, dev_split, args.subdir, args.filename_template, args.pid_suffix,
        verbose, args.loose_match, args.debug_first)

    print("[info] Processing TEST split...")
    test_df, test_included, test_skipped = merge_split(
        args.data_root, test_split, args.subdir, args.filename_template, args.pid_suffix,
        verbose, args.loose_match, args.debug_first)

    # Outputs
    train_out = os.path.join(args.out_dir, "train_features.csv")
    dev_out   = os.path.join(args.out_dir, "dev_features.csv")
    test_out  = os.path.join(args.out_dir, "test_features.csv")
    train_df.to_csv(train_out, index=False)
    dev_df.to_csv(dev_out, index=False)
    test_df.to_csv(test_out, index=False)

    # Manifest
    manifests = {
        "train": {"included": train_included, "skipped": train_skipped},
        "dev":   {"included": dev_included,   "skipped": dev_skipped},
        "test":  {"included": test_included,  "skipped": test_skipped},
        "config": {
            "data_root": args.data_root,
            "labels_dir": args.labels_dir,
            "subdir": args.subdir,
            "filename_template": args.filename_template,
            "pid_suffix": args.pid_suffix,
            "loose_match": args.loose_match,
        },
    }
    with open(os.path.join(args.out_dir, "merge_manifest.json"), "w") as f:
        json.dump(manifests, f, indent=2)

    def _sum(inc, skip): return f"included={len(inc)}, skipped={len(skip)}"
    print("\n[summary]")
    print(f"train -> {train_out}  ({_sum(train_included, train_skipped)})")
    print(f"dev   -> {dev_out}    ({_sum(dev_included, dev_skipped)})")
    print(f"test  -> {test_out}   ({_sum(test_included, test_skipped)})")
    print(f"manifest -> {os.path.join(args.out_dir, 'merge_manifest.json')}")
    print("[ok] Done.")
if __name__ == "__main__":
    main()

#python data_processing.py --data_root ".\DAIC-WOZ Data\data" --labels_dir ".\DAIC-WOZ Data\labels" --out_dir ".\DAIC-WOZ Data\processed_data" --pid_suffix "_P" --loose_match --debug_first 5
