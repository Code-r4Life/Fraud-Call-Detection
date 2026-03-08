"""
Scam Call Detection — Audio Feature Extraction Pipeline
========================================================
Aligned to architecture diagram:

  Stage 1 → Rolling Buffer (5-sec windows, in-memory, no storage)
  Stage 2 → Voice Activity Detection (removes silence & non-speech)
  Stage 3a → MFCC Feature Extraction     → feeds Phoneme CNN model
  Stage 3b → Prosody Feature Extraction  → feeds Urgency/Prosody MLP model

Dataset expected at:
  processed_dataset/
    NORMAL_CALLS/   *.wav   → label 0
    SCAM_CALLS/     *.wav   → label 1

Outputs saved to:
  features/
    NORMAL_CALLS/
      mfcc_<filename>_<window>.npy        shape: (T, 120)  ← CNN input
      prosody_<filename>_<window>.npy     shape: (12,)     ← MLP input
    SCAM_CALLS/
      ...
  features/mfcc_labels.csv               → (filepath, label)
  features/prosody_labels.csv            → (filepath, label)
"""

import os
import csv
import logging
import numpy as np
import librosa
import webrtcvad
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIG — all hyperparameters in one place
# ─────────────────────────────────────────────
SR           = 16000  # Sample rate — WebRTC VAD requires 8k/16k/32k
WINDOW_SEC   = 4      # Rolling buffer size (diagram specifies 5–10 sec)
OVERLAP      = 0.5    # 50% overlap so no speech is missed at window edges
VAD_MODE     = 2      # 0–3: higher = stricter. 2 is best for phone calls
VAD_FRAME_MS = 30     # VAD frame size — MUST be 10, 20, or 30 ms
MIN_SPEECH_S = 1.0    # Skip window if speech after VAD is shorter than this
MIN_SEG_S    = 1.2    # Minimum continuous speech block to keep (dominant speaker)
N_MFCC       = 40     # Number of MFCC coefficients
N_FFT        = 400    # ~25 ms window at 16kHz
HOP_LENGTH   = 160    # ~10 ms hop at 16kHz (standard for speech tasks)
PITCH_FMIN   = 75     # Min pitch Hz — below this is not human speech
PITCH_FMAX   = 300    # Max pitch Hz — above this is not human speech

DATASET_ROOT = "processed_dataset"
FEATURE_ROOT = "features"

LABEL_MAP = {
    "NORMAL_CALLS": 0,
    "SCAM_CALLS":   1,
}


# ─────────────────────────────────────────────
# STAGE 1 — Rolling Buffer
# ─────────────────────────────────────────────
def load_audio(path: str) -> np.ndarray:
    """Load any audio file → mono float32 at SR Hz."""
    try:
        audio, _ = librosa.load(path, sr=SR, mono=True)
    except Exception as e:
        raise RuntimeError(f"Could not load '{path}': {e}")
    if len(audio) == 0:
        raise ValueError(f"'{path}' is empty after loading.")
    return audio


def rolling_buffer(audio: np.ndarray) -> list:
    """
    Stage 1: Split audio into overlapping windows (in-memory only).
    Last partial window is zero-padded — no audio is ever dropped.
    """
    size = int(WINDOW_SEC * SR)
    step = int(size * (1 - OVERLAP))
    windows = []
    for start in range(0, len(audio), step):
        chunk = audio[start : start + size]
        if len(chunk) < size:
            chunk = np.pad(chunk, (0, size - len(chunk)))
        windows.append(chunk)
    return windows


# ─────────────────────────────────────────────
# STAGE 2 — Voice Activity Detection
# ─────────────────────────────────────────────
_vad = webrtcvad.Vad(VAD_MODE)

def vad_filter(audio: np.ndarray) -> np.ndarray:
    """
    Stage 2: Remove non-speech frames using WebRTC VAD.
    Neighbour smoothing prevents choppy artefacts.
    """
    frame_len = int(SR * VAD_FRAME_MS / 1000)
    frames, flags = [], []

    for i in range(0, len(audio) - frame_len + 1, frame_len):
        frame = audio[i : i + frame_len]
        frame_bytes = (frame * 32768).astype(np.int16).tobytes()
        try:
            is_speech = _vad.is_speech(frame_bytes, SR)
        except Exception:
            is_speech = False
        frames.append(frame)
        flags.append(is_speech)

    if not frames:
        return np.array([], dtype=np.float32)

    flags = np.array(flags, dtype=bool)
    smoothed = flags.copy()
    smoothed[:-1] |= flags[1:]
    smoothed[1:]  |= flags[:-1]

    kept = [f for f, keep in zip(frames, smoothed) if keep]
    return np.concatenate(kept) if kept else np.array([], dtype=np.float32)


def dominant_speaker_filter(audio: np.ndarray) -> np.ndarray:
    """
    Keep only long continuous speech blocks (dominant speaker).
    Short bursts < MIN_SEG_S are removed (likely the other party).
    """
    if len(audio) == 0:
        return audio

    energy = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
    threshold = np.mean(energy) * 0.6
    mask = energy > threshold

    segments, start = [], None
    for i, active in enumerate(mask):
        if active and start is None:
            start = i
        elif not active and start is not None:
            if (i - start) * HOP_LENGTH / SR >= MIN_SEG_S:
                segments.append((start, i))
            start = None
    if start is not None:
        if (len(mask) - start) * HOP_LENGTH / SR >= MIN_SEG_S:
            segments.append((start, len(mask)))

    if not segments:
        return np.array([], dtype=np.float32)

    pieces = [audio[int(s * HOP_LENGTH) : min(int(e * HOP_LENGTH), len(audio))]
              for s, e in segments]
    return np.concatenate(pieces)


# ─────────────────────────────────────────────
# STAGE 3a — MFCC Feature Extraction
# Output shape: (T, 120) → input for Phoneme CNN model
# ─────────────────────────────────────────────
def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    """
    Extract MFCC + delta + delta-delta, then apply CMVN.
    Output shape: (T, N_MFCC * 3) = (T, 120)
    """
    mfcc   = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC,
                                   n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2]).T     # (T, 120)

    mean = features.mean(axis=0, keepdims=True)
    std  = features.std(axis=0, keepdims=True) + 1e-8
    return ((features - mean) / std).astype(np.float32)


# ─────────────────────────────────────────────
# STAGE 3b — Prosody Feature Extraction
# Output shape: (12,) flat vector → input for Urgency MLP model
#
# 12 features:
#   [0]  pitch_mean         — average vocal frequency (Hz)
#   [1]  pitch_std          — pitch instability → emotional pressure
#   [2]  pitch_range        — expressiveness of tone
#   [3]  pitch_slope        — rising trend → persuasion marker
#   [4]  energy_mean        — overall loudness level
#   [5]  energy_std         — dynamic variation → aggressive delivery
#   [6]  energy_peak        — loudness burst (shouting/emphasis)
#   [7]  speech_rate        — voiced frames per second → fast talker
#   [8]  voiced_ratio       — proportion of time speaking → dominance
#   [9]  spectral_flux      — rate of spectral change → emotional volatility
#   [9]  pause_ratio        — low pauses = urgency pressure
#   [10] avg_pause_duration — short pauses = no thinking time allowed
#
# All features normalised to [0, 1] using fixed physical bounds.
# ─────────────────────────────────────────────
def extract_prosody(audio: np.ndarray) -> np.ndarray:
    """
    Extract 12 prosodic urgency features from a clean speech segment.

    Fixes vs GPT skeleton:
      ✔ pitch_slope added (described but missing in GPT code)
      ✔ avg_pause_duration added (listed but never implemented by GPT)
      ✔ speech_rate fixed: voiced_frames/duration (not voiced_ratio/duration)
      ✔ per-feature normalisation so all 12 values share the same scale
      ✔ graceful zero-return when audio is silent or too short
    """
    duration = len(audio) / SR
    if duration == 0:
        return np.zeros(12, dtype=np.float32)

    # ── Pitch ─────────────────────────────────────────────────────────
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=PITCH_FMIN, fmax=PITCH_FMAX, sr=SR
    )
    valid_f0 = f0[~np.isnan(f0)]

    if len(valid_f0) > 1:
        pitch_mean  = float(np.mean(valid_f0))
        pitch_std   = float(np.std(valid_f0))
        pitch_range = float(np.ptp(valid_f0))
        t           = np.where(~np.isnan(f0))[0].astype(np.float32)
        t_seconds   = t * HOP_LENGTH / SR
        pitch_slope = float(np.polyfit(t_seconds, valid_f0, 1)[0])
    else:
        pitch_mean = pitch_std = pitch_range = pitch_slope = 0.0

    # ── Energy ────────────────────────────────────────────────────────
    energy      = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
    energy_mean = float(energy.mean())
    energy_std  = float(energy.std())
    energy_peak = float(energy.max())

    # ── Speech Rate & Voiced Ratio ────────────────────────────────────
    n_voiced     = int(np.sum(voiced_flag)) if voiced_flag is not None else 0
    n_frames     = len(voiced_flag)         if voiced_flag is not None else 1
    voiced_ratio = n_voiced / n_frames                 # proportion 0–1
    speech_rate  = n_voiced / duration                 # voiced frames / sec

    # ── Spectral Flux ────────────────────────────────────
    S = np.abs(librosa.stft(audio, hop_length=HOP_LENGTH))
    flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
    spectral_flux = float(np.mean(flux))

    # ── Pause Behaviour ───────────────────────────────────────────────
    silence_mask = energy < (energy_mean * 0.4)
    pause_ratio  = float(silence_mask.sum() / len(silence_mask))

    # avg length of each contiguous silent run (in seconds)
    pause_durations, run = [], 0
    for s in silence_mask:
        if s:
            run += 1
        elif run > 0:
            pause_durations.append(run * HOP_LENGTH / SR)
            run = 0
    if run > 0:
        pause_durations.append(run * HOP_LENGTH / SR)
    avg_pause_duration = float(np.mean(pause_durations)) if pause_durations else 0.0

    # ── Assemble raw vector ───────────────────────────────────────────
    raw = np.array([
        pitch_mean,
        pitch_std,
        pitch_range,
        pitch_slope,
        energy_mean,
        energy_std,
        energy_peak,
        speech_rate,
        voiced_ratio,
        spectral_flux,
        pause_ratio,
        avg_pause_duration,
    ], dtype=np.float32)

    # ── Normalise to [0, 1] using fixed physical bounds ───────────────
    # Fixed bounds (not per-file) so normalisation is consistent at
    # inference time when processing a single live call window.
    bounds = np.array([
        [75,   300 ],   # pitch_mean        Hz
        [0,    100 ],   # pitch_std         Hz
        [0,    225 ],   # pitch_range       Hz
        [-5,   5   ],   # pitch_slope       Hz/frame
        [0,    0.5 ],   # energy_mean       RMS
        [0,    0.3 ],   # energy_std        RMS
        [0,    1.0 ],   # energy_peak       RMS
        [0,    500 ],   # speech_rate       voiced frames/sec
        [0,    1.0 ],   # voiced_ratio
        [0,    50  ],   # spectral_flux       mean L2 norm across frames
        [0,    1.0 ],   # pause_ratio
        [0,    2.0 ],   # avg_pause_duration seconds
    ], dtype=np.float32)

    lo, hi = bounds[:, 0], bounds[:, 1]
    return np.clip((raw - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
def save_feature(array: np.ndarray, prefix: str,
                 folder_name: str, file_stem: str, window_idx: int) -> str:
    out_dir = os.path.join(FEATURE_ROOT, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}_{file_stem}_{window_idx:05d}.npy")
    np.save(path, array)
    return path


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