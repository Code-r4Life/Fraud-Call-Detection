import os
import librosa
import numpy as np
from scam_detection.config import SR, N_MFCC, N_FFT, HOP_LENGTH, PITCH_FMAX, PITCH_FMIN, FEATURE_ROOT

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
    Extract 11 prosodic urgency features from a clean speech segment.

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
        # [75,   300 ],   # pitch_mean        Hz
        # [0,    100 ],   # pitch_std         Hz
        # [0,    225 ],   # pitch_range       Hz
        # [-5,   5   ],   # pitch_slope       Hz/frame
        # [0,    0.5 ],   # energy_mean       RMS
        # [0,    0.3 ],   # energy_std        RMS
        # [0,    1.0 ],   # energy_peak       RMS
        # [0,    500 ],   # speech_rate       voiced frames/sec
        # [0,    1.0 ],   # voiced_ratio
        # [0,    50  ],   # spectral_flux       mean L2 norm across frames
        # [0,    1.0 ],   # pause_ratio
        # [0,    2.0 ],   # avg_pause_duration seconds

        [50,   500 ],   # pitch_mean (covers male + female voices)
        [0,    200 ],   # pitch_std
        [0,    400 ],   # pitch_range
        [-20,  20  ],   # pitch_slope
        [0,    1.0 ],   # energy_mean
        [0,    0.6 ],   # energy_std
        [0,    2.0 ],   # energy_peak
        [0,    800 ],   # speech_rate
        [0,    1.0 ],   # voiced_ratio
        [0,    120 ],   # spectral_flux
        [0,    1.0 ],   # pause_ratio
        [0,    4.0 ],   # avg_pause_duration
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