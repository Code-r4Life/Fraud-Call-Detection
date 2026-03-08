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
      prosody_<filename>_<window>.npy     shape: (11,)     ← MLP input
    SCAM_CALLS/
      ...
  features/mfcc_labels.csv               → (filepath, label)
  features/prosody_labels.csv            → (filepath, label)
"""

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

DATASET_ROOT = r"processed_dataset"
FEATURE_ROOT = r"features"

LABEL_MAP = {
    "NORMAL_CALLS": 0,
    "SCAM_CALLS":   1,
}
