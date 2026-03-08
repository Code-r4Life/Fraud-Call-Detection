import os
import csv
import logging
from pathlib import Path
from scam_detection.config import DATASET_ROOT, FEATURE_ROOT, LABEL_MAP, MIN_SPEECH_S, SR
from scam_detection.audio_pipeline import load_audio, rolling_buffer, vad_filter, dominant_speaker_filter
from scam_detection.feature_extraction import extract_mfcc, extract_prosody, save_feature

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PIPELINE — Single File
# ─────────────────────────────────────────────
def process_file(wav_path: str, folder_name: str,
                 label: int, mfcc_writer, prosody_writer) -> int:
    """
    Run all stages on one .wav file.
    Saves mfcc_*.npy AND prosody_*.npy for every valid window.
    Returns number of windows saved.
    """
    log.info(f"Processing: {wav_path}")
    try:
        audio = load_audio(wav_path)
    except Exception as e:
        log.error(f"  Skipped — {e}")
        return 0

    file_stem = Path(wav_path).stem
    windows   = rolling_buffer(audio)
    saved     = 0

    for idx, window in enumerate(windows):

        # Stage 2a — VAD
        speech = vad_filter(window)
        if len(speech) < SR * MIN_SPEECH_S:
            log.debug(f"  Window {idx}: skipped (insufficient speech after VAD)")
            continue

        # Stage 2b — Dominant speaker filter
        dominant = dominant_speaker_filter(speech)
        if len(dominant) < SR * MIN_SPEECH_S:
            log.debug(f"  Window {idx}: skipped (no dominant speech found)")
            continue

        # Stage 3a — MFCC → Phoneme CNN
        mfcc      = extract_mfcc(dominant)
        mfcc_path = save_feature(mfcc, "mfcc", folder_name, file_stem, saved)
        mfcc_writer.writerow([mfcc_path, label])

        # Stage 3b — Prosody → Urgency MLP
        prosody   = extract_prosody(dominant)
        pros_path = save_feature(prosody, "prosody", folder_name, file_stem, saved)
        prosody_writer.writerow([pros_path, label])

        log.info(f"  Window {idx}: MFCC {mfcc.shape} | Prosody {prosody.shape}")
        saved += 1

    log.info(f"  → {saved}/{len(windows)} windows saved for {file_stem}")
    return saved


# ─────────────────────────────────────────────
# PIPELINE — Full Dataset
# ─────────────────────────────────────────────
def process_dataset() -> None:
    """
    Auto-discovers processed_dataset/NORMAL_CALLS and SCAM_CALLS,
    processes all 48 .wav files, and writes two label CSVs.
    """
    os.makedirs(FEATURE_ROOT, exist_ok=True)
    mfcc_csv_path    = os.path.join(FEATURE_ROOT, "mfcc_labels.csv")
    prosody_csv_path = os.path.join(FEATURE_ROOT, "prosody_labels.csv")

    total_files = total_windows = 0

    with open(mfcc_csv_path, "w", newline="") as mf, \
         open(prosody_csv_path, "w", newline="") as pf:

        mw = csv.writer(mf)
        pw = csv.writer(pf)
        mw.writerow(["filepath", "label"])
        pw.writerow(["filepath", "label"])

        for folder_name, label in LABEL_MAP.items():
            folder_path = os.path.join(DATASET_ROOT, folder_name)

            if not os.path.isdir(folder_path):
                log.warning(f"Folder not found: '{folder_path}' — skipping.")
                continue

            wav_files = sorted(Path(folder_path).glob("*.wav"))
            log.info(f"\n{'='*55}")
            log.info(f"Folder : {folder_name}  (label={label})")
            log.info(f"Files  : {len(wav_files)} .wav files found")
            log.info(f"{'='*55}")

            for wav_path in wav_files:
                count = process_file(str(wav_path), folder_name, label, mw, pw)
                total_windows += count
                total_files   += 1

    log.info(f"\n{'='*55}")
    log.info(f"DONE — {total_files} files | {total_windows} windows per feature type")
    log.info(f"MFCC labels    → {mfcc_csv_path}")
    log.info(f"Prosody labels → {prosody_csv_path}")
    log.info(f"{'='*55}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    process_dataset()