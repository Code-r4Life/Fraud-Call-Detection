"""
Real-Time Scam Detector — 4-Model Parallel Fusion
==================================================
Models (run in parallel per window):
    1. Phoneme CNN        → scripted speech patterns    (0–1 continuous)
    2. Urgency detector   → pitch/energy/rate features  (score 1–3, mapped to 0–1)
    3. Repetition CNN     → scam keyword density        (0–1 continuous)
    4. Sentence transformer → intent stage progression  (0–1 via stage risk)

Fusion (linear combination, weights sum to 1.0):
    raw_risk = 0.4×stage_risk + 0.3×rep_prob + 0.2×phoneme_prob + 0.1×urgency_norm

Running risk score (smooths out single-window spikes):
    running_risk = 0.7×prev_running_risk + 0.3×raw_risk

Usage:
    python realtime_inference.py
    python realtime_inference.py --window 4 --threshold 0.5
"""

import sys
import os
import subprocess
import tempfile
import argparse
import warnings
import time
import numpy as np
import librosa
import joblib
import sounddevice as sd
import scipy.io.wavfile as wav
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from scam_detection.config import SR, MIN_SPEECH_S, HOP_LENGTH
from scam_detection.audio_pipeline import vad_filter, dominant_speaker_filter
from scam_detection.feature_extraction import extract_mfcc
from repetition_preprocessing import extract_mfcc_raw

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
WINDOW_SEC      = 4

# Phoneme CNN
PHONEME_MODEL_PATH    = "models/best_phoneme_model.keras"
MAX_LEN               = 370

# Repetition CNN
REPETITION_MODEL_PATH = "models/best_repetition_model.keras"
MAX_PHRASE_LEN        = 470
SLICE_SEC             = 1
REP_THRESHOLD         = 0.6
REP_MIN_HITS          = 2

# Whisper (for sentence transformer)
WHISPER_CLI   = r"C:\Users\RadhaKrishna\whisper.cpp\build\bin\whisper-cli.exe"
MODEL_PATH_W  = r"C:\Users\RadhaKrishna\whisper.cpp\models\ggml-tiny.en.bin"
WHISPER_LANG  = "en"
SIMILARITY_GATE = 0.55

# Urgency score → normalized value mapping
URGENCY_MAP = {0: 0.0, 1: 0.2, 2: 0.5, 3: 0.9}

# Fusion weights — must sum to 1.0
W_STAGE    = 0.60   # sentence transformer
W_REP      = 0.20   # repetition CNN
W_PHONEME  = 0.10   # phoneme CNN
W_URGENCY  = 0.10   # urgency detector

# Running risk smoothing
ALPHA      = 0.7    # weight of previous running risk
BETA       = 0.3    # weight of current raw risk

# Final verdict threshold
THRESHOLD  = 0.45


# ─────────────────────────────────────────────
# INTENT DICTIONARY (sentence transformer)
# ─────────────────────────────────────────────
INTENTS = {
    "greeting": [
        "hello sir how are you", "hello maam how are you",
        "good morning sir", "good afternoon maam",
        "am I speaking with the account holder",
        "this is a service call regarding your account",
        "namaste sir", "namaste maam",
        "hello this is customer service",
        "hello I am calling regarding your bank account",
        "hello we are calling from bank support",
    ],
    "authority": [
        "I am calling from SBI bank", "I am calling from HDFC bank",
        "I am calling from ICICI bank", "I am calling from Axis bank",
        "I am calling from bank verification team",
        "I am calling from KYC verification department",
        "This is the bank verification team",
        "This is UPI support team",
        "main bank se bol raha hoon",
        "main SBI bank se bol raha hoon",
        "ye bank verification call hai",
    ],
    "problem": [
        "your account has suspicious activity",
        "we detected unusual transactions",
        "your ATM card has been temporarily blocked",
        "your bank account is under review",
        "your KYC is incomplete", "your KYC update is pending",
        "your account is temporarily suspended",
        "your account has been locked",
        "aapka account block ho sakta hai",
        "aapka ATM card block ho gaya hai",
        "aapka KYC pending hai",
    ],
    "urgency": [
        "you must verify immediately", "this is very urgent",
        "immediate action is required",
        "your account will be blocked today",
        "your account will be permanently blocked",
        "this needs to be done right now",
        "your card will be blocked today",
        "this cannot be delayed",
        "abhi verify karna hoga", "turant verify karna padega",
        "warna account block ho jayega",
    ],
    "data_request": [
        "tell me the OTP sent to your phone",
        "share the OTP you received",
        "read the verification code",
        "tell me the six digit code",
        "confirm the number sent to your phone",
        "share the verification code with me",
        "OTP bata dijiye", "OTP bol dijiye",
        "jo OTP aya hai wo bataiye",
        "SMS mein jo code aya hai wo boliye",
        "OTP share kar dijiye",
    ],
}

STAGE_MAP  = {"greeting":1, "authority":2, "problem":3, "urgency":4, "data_request":5}
STAGE_RISK = {0:0.05, 1:0.10, 2:0.25, 3:0.50, 4:0.70, 5:0.95}
STAGE_LABELS = {0:"none", 1:"GREETING", 2:"AUTHORITY", 3:"PROBLEM", 4:"URGENCY", 5:"DATA_REQUEST"}


# ─────────────────────────────────────────────
# LOAD ALL MODELS
# ─────────────────────────────────────────────
print("Loading models...")
phoneme_model    = load_model(PHONEME_MODEL_PATH)
repetition_model = load_model(REPETITION_MODEL_PATH)
embedder         = SentenceTransformer("all-MiniLM-L6-v2")
intent_embeddings = {
    intent: embedder.encode(sentences)
    for intent, sentences in INTENTS.items()
}
print("All models loaded.\n")


# ─────────────────────────────────────────────
# MODEL 1 — PHONEME CNN
# ─────────────────────────────────────────────
def run_phoneme(dominant: np.ndarray) -> float:
    mfcc = extract_mfcc(dominant)
    if mfcc.shape[0] < MAX_LEN:
        pad  = np.zeros((MAX_LEN - mfcc.shape[0], mfcc.shape[1]))
        mfcc = np.vstack((mfcc, pad))
    else:
        mfcc = mfcc[:MAX_LEN, :]
    inp = np.expand_dims(mfcc, axis=0)
    return float(phoneme_model.predict(inp, verbose=0)[0][0])


# ─────────────────────────────────────────────
# MODEL 2 — URGENCY DETECTOR
# ─────────────────────────────────────────────
# Baseline — update these from your environment
BASELINE = {
    "pitch_mean": 372.61,
    "pitch_std": 0.0,
    "energy_mean": 3.23e-6,
    "energy_var": 1.33e-9,
    "spectral_centroid": 49257.50,
    "speech_rate": 12.5
}

def extract_urgency_features(audio: np.ndarray) -> dict:
    pitches, magnitudes = librosa.piptrack(y=audio, sr=SR)
    pitch_vals = []
    for i in range(pitches.shape[1]):
        idx = magnitudes[:, i].argmax()
        p   = pitches[idx, i]
        if p > 0:
            pitch_vals.append(p)
    pitch_vals = np.array(pitch_vals)

    return {
        "pitch_mean":        float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0,
        "pitch_std":         float(np.std(pitch_vals))  if len(pitch_vals) > 0 else 0.0,
        "energy_mean":       float(np.mean(librosa.feature.rms(y=audio)[0])),
        "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=SR))),
        "speech_rate":       float(len(librosa.util.peak_pick(
            librosa.onset.onset_strength(y=audio, sr=SR),
            pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=5
        )) / (len(audio) / SR + 1e-8)),
    }

def run_urgency(audio: np.ndarray) -> tuple:
    """Returns (raw_score 0-3, normalized 0-1)"""
    f = extract_urgency_features(audio)
    score = 0

    if f["energy_mean"] < 0.005:
        return 0, 0.0
    if BASELINE["energy_mean"] > 0.001:
        if f["energy_mean"] < BASELINE["energy_mean"] * 1.3:
            return 0, 0.0

    if f["energy_mean"]        > max(BASELINE["energy_mean"]        * 1.8, 0.002): score += 1
    if f["pitch_mean"]         > BASELINE["pitch_mean"]         * 1.3:             score += 1
    if f["pitch_std"]          > BASELINE["pitch_std"]          * 1.5:             score += 1
    if f["speech_rate"]        > BASELINE["speech_rate"]        * 1.5:             score += 1
    if f["spectral_centroid"]  > BASELINE["spectral_centroid"]  * 1.25:            score += 1

    score      = min(score, 3)   # cap at 3 for mapping
    normalized = URGENCY_MAP.get(score, 0.0)
    return score, normalized


# ─────────────────────────────────────────────
# MODEL 3 — REPETITION CNN
# ─────────────────────────────────────────────
def run_repetition(speech: np.ndarray) -> float:
    slice_samples = int(SLICE_SEC * SR)
    probs = []
    for start in range(0, len(speech), slice_samples):
        chunk = speech[start : start + slice_samples]
        if len(chunk) < SR * 0.3:
            continue
        mfcc = extract_mfcc_raw(chunk)
        T    = mfcc.shape[0]
        if T < MAX_PHRASE_LEN:
            pad  = np.zeros((MAX_PHRASE_LEN - T, mfcc.shape[1]), dtype=np.float32)
            mfcc = np.vstack((mfcc, pad))
        else:
            mfcc = mfcc[:MAX_PHRASE_LEN, :]
        prob = float(repetition_model.predict(
            np.expand_dims(mfcc, axis=0), verbose=0
        )[0][0])
        probs.append(prob)
    return float(np.max(probs)) if probs else 0.0


# ─────────────────────────────────────────────
# MODEL 4 — SENTENCE TRANSFORMER (via Whisper)
# ─────────────────────────────────────────────
def transcribe(audio: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        wav.write(tmp_path, SR, (audio * 32767).astype(np.int16))
        result = subprocess.run(
            [WHISPER_CLI, "-m", MODEL_PATH_W, "-f", tmp_path,
             "-l", WHISPER_LANG, "--no-timestamps"],
            capture_output=True, text=True, timeout=30,
        )
        transcript = result.stdout.strip()
        for artifact in ["[BLANK_AUDIO]", "(music)", "[Music]", "(Music)"]:
            transcript = transcript.replace(artifact, "").strip()
        return transcript
    except Exception:
        return ""
    finally:
        os.unlink(tmp_path)

def run_stage(audio: np.ndarray, session) -> tuple:
    """Returns (stage_risk 0-1, stage_number, intent_name, similarity_score, transcript)"""
    transcript = transcribe(audio)
    if not transcript:
        return STAGE_RISK[session.current_stage], session.current_stage, "none", 0.0, ""

    text_emb = embedder.encode([transcript.lower().strip()])
    scores   = {
        intent: float(np.max(cosine_similarity(text_emb, embs)))
        for intent, embs in intent_embeddings.items()
    }
    best_intent = max(scores, key=scores.get)
    best_score  = scores[best_intent]

    if best_score >= SIMILARITY_GATE:
        stage = STAGE_MAP[best_intent]
        if stage > session.current_stage:
            session.current_stage = stage
        session.stage_history.append((best_intent, best_score))

    return (
        STAGE_RISK[session.current_stage],
        session.current_stage,
        best_intent,
        best_score,
        transcript,
    )


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
class SessionState:
    def __init__(self):
        self.current_stage  = 0
        self.stage_history  = []
        self.running_risk   = 0.0
        self.keyword_hits   = 0
        self.processed      = 0

    def update_running_risk(self, raw_risk: float) -> float:
        self.running_risk = ALPHA * self.running_risk + BETA * raw_risk
        return self.running_risk


# ─────────────────────────────────────────────
# MAIN REAL-TIME LOOP
# ─────────────────────────────────────────────
def run_realtime(window_sec: int = WINDOW_SEC,
                 threshold: float = THRESHOLD) -> None:

    window_samples = int(window_sec * SR)
    state          = SessionState()

    print(f"\n{'='*70}")
    print(f"REAL-TIME SCAM DETECTOR  —  4-Model Parallel Fusion")
    print(f"Window: {window_sec}s  |  SR: {SR}Hz  |  Threshold: {threshold}")
    print(f"Weights: Stage={W_STAGE} Rep={W_REP} Phoneme={W_PHONEME} Urgency={W_URGENCY}")
    print(f"Running risk: {ALPHA}×prev + {BETA}×current")
    print(f"Press Ctrl+C to stop.")
    print(f"{'='*70}\n")

    with ThreadPoolExecutor(max_workers=4) as executor:
        try:
            window_idx = 0
            while True:
                # ── Capture ───────────────────────────────────────────
                print(f"  [{window_idx:03d}] Listening ({window_sec}s)...", end="\r")
                audio = sd.rec(
                    frames     = window_samples,
                    samplerate = SR,
                    channels   = 1,
                    dtype      = "float32"
                )
                sd.wait()
                audio = audio.flatten()

                # ── VAD ───────────────────────────────────────────────
                speech = vad_filter(audio)
                if len(speech) < SR * MIN_SPEECH_S:
                    print(f"  [{window_idx:03d}] skipped (silence)                            ")
                    window_idx += 1
                    continue

                dominant = dominant_speaker_filter(speech)
                if len(dominant) < SR * MIN_SPEECH_S:
                    dominant = speech   # fallback: use VAD output if filter too aggressive

                # ── Run all 4 models IN PARALLEL ──────────────────────
                t0 = time.time()

                f_phoneme   = executor.submit(run_phoneme,   dominant)
                f_urgency   = executor.submit(run_urgency,   dominant)
                f_rep       = executor.submit(run_repetition, speech)
                f_stage     = executor.submit(run_stage,      speech, state)

                phoneme_prob            = f_phoneme.result()
                urgency_score, urg_norm = f_urgency.result()
                rep_prob                = f_rep.result()
                stage_risk, stage_num, intent, sim_score, transcript = f_stage.result()

                elapsed = time.time() - t0

                # ── Keyword hit tracking (repetition density) ─────────
                if rep_prob >= REP_THRESHOLD:
                    state.keyword_hits += 1
                state.processed += 1

                # ── Raw fusion score ──────────────────────────────────
                raw_risk = (
                    W_STAGE   * stage_risk   +
                    W_REP     * rep_prob     +
                    W_PHONEME * phoneme_prob +
                    W_URGENCY * urg_norm
                )

                # ── Running risk (smoothed) ───────────────────────────
                running_risk = state.update_running_risk(raw_risk)

                # ── Verdict ───────────────────────────────────────────
                verdict = "🚨 SCAM" if running_risk >= threshold else "✅ safe"

                if running_risk < 0.15:   risk_label = "✅ safe"
                elif running_risk < 0.35: risk_label = "🟡 low risk"
                elif running_risk < 0.55: risk_label = "🟠 moderate"
                elif running_risk < 0.80: risk_label = "🔴 HIGH RISK"
                else:                     risk_label = "🚨 SCAM ALERT"

                # ── Print window result ───────────────────────────────
                t_display = f'"{transcript[:50]}"' if transcript else "(no transcript)"
                print(
                    f"  [{window_idx:03d}] {t_display}\n"
                    f"         Phoneme: {phoneme_prob:.3f}  "
                    f"Urgency: {urgency_score}/3({urg_norm:.1f})  "
                    f"Repeat: {rep_prob:.3f}  "
                    f"Stage: {STAGE_LABELS[stage_num]}({stage_risk:.2f})\n"
                    f"         Raw: {raw_risk:.3f}  "
                    f"Running: {running_risk:.3f}  "
                    f"→ {risk_label}  [{elapsed:.1f}s]"
                )

                if running_risk >= 0.80:
                    print(f"\n  {'!'*60}")
                    print(f"  ⚠  SCAM ALERT — Running risk: {running_risk:.3f}")
                    print(f"     Stage: {STAGE_LABELS[stage_num]}  |  Intent: {intent}  |  Sim: {sim_score:.3f}")
                    print(f"  {'!'*60}\n")

                window_idx += 1

        except KeyboardInterrupt:
            density = state.keyword_hits / state.processed if state.processed > 0 else 0.0
            print(f"\n\n{'='*70}")
            print(f"  SESSION ENDED")
            print(f"  Windows processed  : {state.processed}")
            print(f"  Final stage        : {STAGE_LABELS[state.current_stage]}")
            print(f"  Final running risk : {state.running_risk:.4f}")
            print(f"  Keyword density    : {density:.4f} ({state.keyword_hits}/{state.processed} hits)")
            print(f"  Stage history      :")
            for intent, sc in state.stage_history:
                print(f"    {intent:<16} sim={sc:.3f}")
            print(f"{'='*70}\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window",    type=int,   default=WINDOW_SEC)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    run_realtime(window_sec=args.window, threshold=args.threshold)