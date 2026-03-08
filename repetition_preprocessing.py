"""
Repetition Score Model — Preprocessing, Split & Augmentation Pipeline
======================================================================
Fixes overfitting from previous version by splitting BEFORE augmenting.

Previous (wrong) order:          Correct order (this file):
  all clips → augment             all clips → split
       ↓                                ↓
     split                        train → augment (×5 variants)
       ↓                          val   → original only
  leakage: aug variants of        test  → original only
  same clip in train & test

Dataset structure expected:
  Wake Words/
    Negatives/          *.wav  → label 0
    Positives/
      Blocked/          *.wav  → label 1
      Card/             *.wav  → label 1
      Code/             *.wav  → label 1
      Emergency/        *.wav  → label 1
      Fee/              *.wav  → label 1
      Last Warning/     *.wav  → label 1
      Lottery/          *.wav  → label 1
      OTP/              *.wav  → label 1
      Password/         *.wav  → label 1
      Prize/            *.wav  → label 1
      Refund/           *.wav  → label 1
      Reward/           *.wav  → label 1
      UPI/              *.wav  → label 1
      Urgent/           *.wav  → label 1
      Verify/           *.wav  → label 1

Output structure:
  rep_features/
    train/
      Positives/
        Blocked/    mfcc_<stem>_orig.npy
                    mfcc_<stem>_slow.npy   ← augmented (train only)
                    mfcc_<stem>_fast.npy
                    mfcc_<stem>_pitch.npy
                    mfcc_<stem>_vol.npy
        Card/       ...
        ... (all 15 category folders)
      Negatives/    mfcc_<stem>_orig.npy + augmented
    val/
      Positives/
        Blocked/    mfcc_<stem>_orig.npy   ← original only, no augmentation
        ...
      Negatives/    mfcc_<stem>_orig.npy
    test/
      Positives/
        Blocked/    mfcc_<stem>_orig.npy   ← original only, no augmentation
        ...
      Negatives/    mfcc_<stem>_orig.npy
  rep_features/
    train_labels.csv
    val_labels.csv
    test_labels.csv
"""

import os
import csv
import logging
import random
import numpy as np
import librosa
from pathlib import Path

from scam_detection.config import SR, HOP_LENGTH
from scam_detection.audio_pipeline import vad_filter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
N_MFCC_REP       = 40
N_FFT_REP        = 400

# Split ratios — must sum to 1.0
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
TEST_RATIO       = 0.15

RANDOM_SEED      = 42     # fixed seed for reproducibility

# Augmentation parameters (applied to TRAIN only)
STRETCH_SLOW     = 0.85
STRETCH_FAST     = 1.15
PITCH_SEMITONES  = 2
VOLUME_MIN       = 0.80
VOLUME_MAX       = 1.20

DATASET_ROOT     = "Wake_Words_processed"
FEATURE_ROOT     = "rep_features"

# Subfolder names inside DATASET_ROOT
POSITIVES_DIR    = "Positives"
NEGATIVES_DIR    = "Negatives"

# All 15 positive category folders
POSITIVE_CATEGORIES = [
    "Blocked", "Card", "Code", "Emergency", "Fee",
    "Last Warning", "Lottery", "OTP", "Password", "Prize",
    "Refund", "Reward", "UPI", "Urgent", "Verify"
]


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
def load_audio_rep(path: str) -> np.ndarray:
    """Load any audio file → mono float32 at SR Hz."""
    try:
        audio, _ = librosa.load(path, sr=SR, mono=True)
    except Exception as e:
        raise RuntimeError(f"Could not load '{path}': {e}")
    if len(audio) == 0:
        raise ValueError(f"'{path}' is empty after loading.")
    return audio


# ─────────────────────────────────────────────
# SPLIT
# Splits a list of file paths into train/val/test.
# Done per-category so every category is represented
# in all three splits (stratified by folder).
# ─────────────────────────────────────────────
def split_files(files: list, seed: int = RANDOM_SEED) -> tuple:
    """
    Split a list of file paths into (train, val, test).
    Splitting is done BEFORE augmentation to prevent leakage.
    Shuffle is seeded for reproducibility.
    """
    files = list(files)
    random.seed(seed)
    random.shuffle(files)

    n        = len(files)
    n_train  = int(n * TRAIN_RATIO)
    n_val    = int(n * VAL_RATIO)

    train = files[:n_train]
    val   = files[n_train : n_train + n_val]
    test  = files[n_train + n_val:]

    return train, val, test


# ─────────────────────────────────────────────
# AUGMENTATION (train split only)
# ─────────────────────────────────────────────
def augment_time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    return librosa.effects.time_stretch(audio, rate=rate).astype(np.float32)


def augment_pitch_shift(audio: np.ndarray) -> np.ndarray:
    n_steps = np.random.uniform(-PITCH_SEMITONES, PITCH_SEMITONES)
    return librosa.effects.pitch_shift(
        audio, sr=SR, n_steps=n_steps
    ).astype(np.float32)


def augment_volume_scale(audio: np.ndarray) -> np.ndarray:
    scale = np.random.uniform(VOLUME_MIN, VOLUME_MAX)
    return (audio * scale).astype(np.float32)


def get_augmented_variants(audio: np.ndarray,
                            is_train: bool) -> list:
    """
    Returns (tag, audio) pairs.
    - Train split : original + 4 augmented variants (5 total)
    - Val/Test    : original only (1 total) — no augmentation
                    This ensures val/test reflect real-world performance.
    """
    variants = [("orig", audio)]

    if is_train:
        variants += [
            ("slow",  augment_time_stretch(audio, STRETCH_SLOW)),
            ("fast",  augment_time_stretch(audio, STRETCH_FAST)),
            ("pitch", augment_pitch_shift(audio)),
            ("vol",   augment_volume_scale(audio)),
        ]

    return variants


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_mfcc_raw(audio: np.ndarray) -> np.ndarray:
    """Raw MFCC only. Output: (T, 40)"""
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SR,
        n_mfcc=N_MFCC_REP,
        n_fft=N_FFT_REP,
        hop_length=HOP_LENGTH
    )
    return mfcc.T.astype(np.float32)


# def pad_mfcc_rep(mfcc: np.ndarray) -> np.ndarray:
#     """Pad or truncate to (MAX_PHRASE_LEN, 40)."""
#     T = mfcc.shape[0]
#     if T < MAX_PHRASE_LEN:
#         pad  = np.zeros((MAX_PHRASE_LEN - T, mfcc.shape[1]), dtype=np.float32)
#         return np.vstack((mfcc, pad))
#     return mfcc[:MAX_PHRASE_LEN, :]


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
def save_rep_feature(mfcc: np.ndarray, split: str,
                     class_subpath: str,
                     file_stem: str, aug_tag: str) -> str:
    """
    Save feature to:
      rep_features/<split>/<class_subpath>/mfcc_<stem>_<tag>.npy

    class_subpath examples:
      Positives/Blocked
      Positives/OTP
      Negatives
    """
    out_dir = os.path.join(FEATURE_ROOT, split, class_subpath)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"mfcc_{file_stem}_{aug_tag}.npy")
    np.save(path, mfcc)
    return path


# ─────────────────────────────────────────────
# PROCESS — One Clip
# ─────────────────────────────────────────────
def process_clip(wav_path: str, split: str,
                 class_subpath: str, label: int,
                 csv_writer, is_train: bool) -> int:
    """
    load → VAD → augment (train only) → extract MFCC → pad → save
    Returns number of files saved.
    """
    try:
        audio = load_audio_rep(wav_path)
    except Exception as e:
        log.error(f"    Skipped {Path(wav_path).name} — {e}")
        return 0

    speech = vad_filter(audio)
    if len(speech) < SR * 0.3:
        log.warning(f"    Skipped {Path(wav_path).name} — "
                    f"too little speech ({len(speech)/SR:.2f}s)")
        return 0

    file_stem = Path(wav_path).stem
    saved     = 0

    for aug_tag, aug_audio in get_augmented_variants(speech, is_train):
        mfcc   = extract_mfcc_raw(aug_audio)    # (T, 40)
        path   = save_rep_feature(
            mfcc, split, class_subpath, file_stem, aug_tag
        )
        csv_writer.writerow([path, label])
        saved += 1

    return saved


# ─────────────────────────────────────────────
# COLLECT ALL FILES
# Returns a dict: { class_subpath: (label, [wav_paths]) }
# ─────────────────────────────────────────────
def collect_all_files() -> dict:
    """
    Walk the dataset folder and collect all wav paths grouped by
    their class_subpath (e.g. 'Positives/Blocked', 'Negatives').
    """
    collection = {}

    # ── Positives (15 category subfolders) ───────────────────────────
    for category in POSITIVE_CATEGORIES:
        folder = Path(DATASET_ROOT) / POSITIVES_DIR / category
        if not folder.is_dir():
            log.warning(f"Category folder not found: '{folder}' — skipping.")
            continue
        wavs = sorted(folder.glob("*.wav"))
        if not wavs:
            log.warning(f"No .wav files in '{folder}'")
            continue
        subpath = f"{POSITIVES_DIR}/{category}"
        collection[subpath] = (1, wavs)
        log.info(f"  {subpath}: {len(wavs)} clips found")

    # ── Negatives (flat folder) ───────────────────────────────────────
    neg_folder = Path(DATASET_ROOT) / NEGATIVES_DIR
    if neg_folder.is_dir():
        wavs = sorted(neg_folder.glob("*.wav"))
        if wavs:
            collection[NEGATIVES_DIR] = (0, wavs)
            log.info(f"  {NEGATIVES_DIR}: {len(wavs)} clips found")
        else:
            log.warning(f"No .wav files in '{neg_folder}'")
    else:
        log.warning(f"Negatives folder not found: '{neg_folder}'")

    return collection


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def process_repetition_dataset() -> None:
    """
    Full pipeline:
      1. Collect all wav files grouped by category
      2. Split each category into train/val/test (stratified per folder)
      3. Process train split with augmentation (×5 per clip)
      4. Process val/test splits with original only (×1 per clip)
      5. Write three separate label CSVs
    """
    os.makedirs(FEATURE_ROOT, exist_ok=True)

    # Open all 3 CSVs at once
    train_csv = os.path.join(FEATURE_ROOT, "train_labels.csv")
    val_csv   = os.path.join(FEATURE_ROOT, "val_labels.csv")
    test_csv  = os.path.join(FEATURE_ROOT, "test_labels.csv")

    log.info(f"\n{'='*55}")
    log.info(f"Dataset root : {DATASET_ROOT}")
    log.info(f"Split ratios : train={TRAIN_RATIO} val={VAL_RATIO} test={TEST_RATIO}")
    log.info(f"Seed         : {RANDOM_SEED}")
    log.info(f"{'='*55}\n")

    collection = collect_all_files()

    totals = {"train": 0, "val": 0, "test": 0}

    with open(train_csv, "w", newline="") as tf, \
         open(val_csv,   "w", newline="") as vf, \
         open(test_csv,  "w", newline="") as xf:

        writers = {
            "train": csv.writer(tf),
            "val":   csv.writer(vf),
            "test":  csv.writer(xf),
        }
        for w in writers.values():
            w.writerow(["filepath", "label"])

        for class_subpath, (label, wavs) in collection.items():
            train_files, val_files, test_files = split_files(wavs)

            log.info(f"\n{class_subpath}  (label={label})")
            log.info(f"  Total: {len(wavs)} | "
                     f"Train: {len(train_files)} | "
                     f"Val: {len(val_files)} | "
                     f"Test: {len(test_files)}")

            # Train — with augmentation
            for wav in train_files:
                n = process_clip(str(wav), "train", class_subpath,
                                 label, writers["train"], is_train=True)
                totals["train"] += n

            # Val — original only
            for wav in val_files:
                n = process_clip(str(wav), "val", class_subpath,
                                 label, writers["val"], is_train=False)
                totals["val"] += n

            # Test — original only
            for wav in test_files:
                n = process_clip(str(wav), "test", class_subpath,
                                 label, writers["test"], is_train=False)
                totals["test"] += n

    log.info(f"\n{'='*55}")
    log.info(f"DONE")
    log.info(f"  Train features : {totals['train']}  (orig + 4 aug per clip)")
    log.info(f"  Val features   : {totals['val']}   (orig only)")
    log.info(f"  Test features  : {totals['test']}   (orig only)")
    log.info(f"  Train CSV → {train_csv}")
    log.info(f"  Val CSV   → {val_csv}")
    log.info(f"  Test CSV  → {test_csv}")
    log.info(f"{'='*55}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    process_repetition_dataset()