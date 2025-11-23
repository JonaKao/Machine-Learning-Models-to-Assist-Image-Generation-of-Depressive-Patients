# save as extract_headshots.py
import os, re, math, cv2, hashlib, pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from imagehash import phash
from insightface.app import FaceAnalysis

# ------------------ CONFIG ------------------
INPUT_VIDEOS_DIR = "dataset/videos"
LABELS_CSV       = "dataset/labels.csv"
OUTPUT_IMG_DIR   = "output/images"
OUTPUT_META_CSV  = "output/images_meta.csv"

# labels.csv expected columns (edit if your CSV uses different names)
COL_SUBJECT = "subject_id"
COL_SCORE   = "score"      # PHQ-8/PHQ-9/BDI-II numeric
COL_SPLIT   = "split"      # e.g., train/dev/test (optional; if absent, will be 'unknown')

# sampling & crop params
FPS_SAMPLE          = 0.5          # frames per second to sample
MIN_FACE_SIZE_PX    = 120        # discard smaller faces
CROP_SCALE          = 1.3        # scale bbox to include head/forehead
OUT_SIZE            = 224        # output image size (square)
BLUR_VAR_THRESHOLD  = 60.0       # variance of Laplacian; lower => blurry
PHASH_DIST_MAX      = 4         # near-duplicate threshold (lower = stricter)
# label threshold; common: PHQ>=10 or BDI-II>=20/29 → depressed
DEPRESSED_THRESHOLD = 10.0   # adjust for your dataset if needed
MAX_FRAMES_PER_SUBJECT = 100
# --------------------------------------------

Path(OUTPUT_IMG_DIR).mkdir(parents=True, exist_ok=True)

# Load labels
labels = pd.read_csv(LABELS_CSV)
if COL_SPLIT not in labels.columns:
    labels[COL_SPLIT] = "unknown"

labels[COL_SUBJECT] = labels[COL_SUBJECT].astype(str)

# Build a quick subject map
subject_info = labels.set_index(COL_SUBJECT)[[COL_SCORE, COL_SPLIT]].to_dict("index")

# Init face detector (RetinaFace via insightface)
app = FaceAnalysis(name="buffalo_l")  # robust default model suite
app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU (ctx_id=-1)

def subject_from_filename(fn: str) -> str:
    # strip extension; keep alnum/_/-
    base = Path(fn).stem
    # common AVEC/DAIC patterns already fine; else adapt this if needed
    return base

def variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def expand_square_crop(x1, y1, x2, y2, scale, W, H):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * scale
    nx1 = int(max(0, cx - side/2))
    ny1 = int(max(0, cy - side/2))
    nx2 = int(min(W, cx + side/2))
    ny2 = int(min(H, cy + side/2))
    return nx1, ny1, nx2, ny2

def crop_and_align(img, face):
    # Use detected bbox; (optional) could use landmarks for similarity alignment
    x1, y1, x2, y2 = map(int, face.bbox.astype(int))
    H, W = img.shape[:2]
    x1, y1, x2, y2 = expand_square_crop(x1, y1, x2, y2, CROP_SCALE, W, H)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
    return crop

def is_duplicate(prev_hashes, img_pil):
    h = phash(img_pil)
    # compare to recent hashes (keep last N per subject to speed up)
    for old in prev_hashes[-20:]:
        if h - old <= PHASH_DIST_MAX:
            return True
    prev_hashes.append(h)
    return False

rows = []
video_files = sorted([p for p in Path(INPUT_VIDEOS_DIR).glob("*") if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}])

for vf in tqdm(video_files, desc="Videos"):
    sid = subject_from_filename(vf.name)
    meta = subject_info.get(sid)
    if meta is None:
        # Skip videos with no labels
        continue
    score = float(meta[COL_SCORE])
    split = meta[COL_SPLIT]
    # Skip this subject entirely if not depressed
    if score < DEPRESSED_THRESHOLD:
        continue
    
    cap = cv2.VideoCapture(str(vf))
    if not cap.isOpened():
        continue

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(round(fps / FPS_SAMPLE)))

    frame_idx = 0
    saved_idx = 0
    prev_hashes = []

    # Per-subject output dir
    subj_dir = Path(OUTPUT_IMG_DIR) / sid
    subj_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ret = cap.grab()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            ret2, frame = cap.retrieve()
            if not ret2:
                break

            # detect faces
            faces = app.get(frame)
            if not faces:
                frame_idx += 1
                continue

            # choose largest face
            faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
            face = faces[0]
            w = face.bbox[2]-face.bbox[0]
            h = face.bbox[3]-face.bbox[1]
            if min(w, h) < MIN_FACE_SIZE_PX:
                frame_idx += 1
                continue

            crop = crop_and_align(frame, face)
            if crop is None:
                frame_idx += 1
                continue

            # quality filter (blur)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if variance_of_laplacian(gray) < BLUR_VAR_THRESHOLD:
                frame_idx += 1
                continue

            # de-dup (near duplicates)
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if is_duplicate(prev_hashes, pil_img):
                frame_idx += 1
                continue

            out_name = f"{sid}_f{frame_idx:06d}.jpg"
            out_path = subj_dir / out_name
            cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved_idx += 1
            if saved_idx >= MAX_FRAMES_PER_SUBJECT:
                break

            rows.append({
                "path": str(out_path),
                "subject_id": sid,
                "score": score,
                "split": split,
                # editable threshold; common: PHQ>=10 or BDI>=20 → depressed
                "depressed": 1 
            })
            saved_idx += 1

        frame_idx += 1

    cap.release()

# Write CSV
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_META_CSV, index=False)
print(f"Saved {len(df)} headshots → {OUTPUT_META_CSV}")
