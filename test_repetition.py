from repetition_preprocessing import load_audio_rep, extract_mfcc_raw, vad_filter
from scam_detection.config import SR, MIN_SPEECH_S
from tensorflow.keras.models import load_model
import numpy as np
import sounddevice as sd
import logging

log = logging.getLogger(__name__)

MAX_LEN        = 470   # pad target — must match training
REP_THRESHOLD  = 0.75
WINDOW_SEC     = 4     # mic capture window
SLICE_SEC      = 1     # slice size — matches your clip length
REP_MIN_HITS   = 2
model          = load_model(r"models\best_repetition_model.keras")


def pad_features(mfcc: np.ndarray, max_len: int) -> np.ndarray:
    if mfcc.shape[0] < max_len:
        padding = np.zeros((max_len - mfcc.shape[0], mfcc.shape[1]))
        mfcc = np.vstack((mfcc, padding))
    else:
        mfcc = mfcc[:max_len, :]
    return mfcc


def predict_on_slices(speech: np.ndarray) -> float:
    """
    Slice VAD-cleaned speech into SLICE_SEC chunks.
    Pad each chunk to MAX_LEN and predict.
    Return MAX probability across all slices.

    Why max?
      One strong keyword hit in any slice is enough to flag the window.
      Mean would dilute a single strong hit across silent/irrelevant slices.

    Why this matches training?
      Training clips were 1-2 sec padded to 4 sec (MAX_LEN frames).
      Each slice here is 1 sec padded to MAX_LEN — same distribution.
    """
    slice_samples = int(SLICE_SEC * SR)
    probs = []

    for start in range(0, len(speech), slice_samples):
        chunk = speech[start : start + slice_samples]

        # Skip chunks too short to contain a word
        if len(chunk) < SR * 0.3:
            continue

        mfcc = extract_mfcc_raw(chunk)          # (T, 40) — T ≈ 100 for 1 sec
        mfcc = pad_features(mfcc, MAX_LEN)      # (470, 40) — zeros fill the rest
        prob = float(model.predict(
            np.expand_dims(mfcc, axis=0), verbose=0
        )[0][0])
        probs.append(prob)

    return float(np.max(probs)) if probs else 0.0


def run_realtime_repetition() -> None:
    window_samples = int(WINDOW_SEC * SR)
    keyword_hits   = 0
    processed      = 0

    print(f"\n{'='*60}")
    print(f"REAL-TIME REPETITION DETECTION")
    print(f"Window: {WINDOW_SEC}s → sliced into {SLICE_SEC}s chunks → padded to {MAX_LEN} frames")
    print(f"SR: {SR}Hz  |  Threshold: {REP_THRESHOLD}  |  Min hits: {REP_MIN_HITS}")
    print(f"Speak into your mic. Press Ctrl+C to stop.")
    print(f"{'='*60}\n")

    try:
        while True:
            print(f"  Listening...", end="\r")
            audio = sd.rec(
                frames     = window_samples,
                samplerate = SR,
                channels   = 1,
                dtype      = "float32"
            )
            sd.wait()
            audio = audio.flatten()

            # VAD
            speech = vad_filter(audio)
            if len(speech) < SR * MIN_SPEECH_S:
                print(f"  Window {processed:03d} | skipped (silence)              ")
                continue

            # Slice → pad → predict → max
            prob = predict_on_slices(speech)

            hit = prob >= REP_THRESHOLD
            if hit:
                keyword_hits += 1
            processed += 1

            density = (keyword_hits / processed
                       if keyword_hits >= REP_MIN_HITS else 0.0)

            if density == 0.0:
                risk = "normal"
            elif density < 0.3:
                risk = "suspicious ⚠"
            elif density < 0.6:
                risk = "scam likely ⚠⚠"
            else:
                risk = "SCAM SCRIPT 🚨"

            tag = "HIT ⚠" if hit else "     "
            print(
                f"  Window {processed:03d} | "
                f"prob: {prob:.4f}  {tag} | "
                f"density: {density:.3f}  → {risk}"
            )

    except KeyboardInterrupt:
        final_density = keyword_hits / processed if processed > 0 else 0.0
        print(f"\n\n{'='*60}")
        print(f"  Session ended.")
        print(f"  Processed     : {processed} windows")
        print(f"  Keyword hits  : {keyword_hits}")
        print(f"  Final density : {final_density:.4f}")
        print(f"{'='*60}\n")


run_realtime_repetition()