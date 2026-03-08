"""
Real-Time Scam Conversation Detector
=====================================
Pipeline per 4-sec window:
  mic capture → VAD → save temp WAV → whisper-cli → transcript
  → sentence embedding → cosine similarity → stage detection
  → scam risk score

Usage:
    python realtime_whisper_scam.py
    python realtime_whisper_scam.py --window 4 --whisper-model base

Requirements:
    pip install sounddevice scipy sentence-transformers scikit-learn numpy
    whisper.cpp installed locally (whisper-cli or main in PATH or specified)
"""

import sys
import os
import subprocess
import tempfile
import argparse
import warnings
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import logging

warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Import VAD from your existing pipeline ────────────────────────────────────
from scam_detection.config import SR, MIN_SPEECH_S
from scam_detection.audio_pipeline import vad_filter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

WINDOW_SEC      = 4          # seconds captured per window
WHISPER_CLI     = r"C:\Users\RadhaKrishna\whisper.cpp\build\bin\whisper-cli.exe"   # change to full path if not in PATH
                                   # e.g. r"C:\whisper.cpp\build\bin\whisper-cli.exe"
WHISPER_MODEL   = "tiny"     # tiny / base / small / medium
WHISPER_LANG    = "en"     # auto-detect, or "en", "hi"
MODEL_PATH = r"C:\Users\RadhaKrishna\whisper.cpp\models\ggml-tiny.en.bin"
SIMILARITY_GATE = 0.55       # minimum cosine similarity to count as a stage match


# ─────────────────────────────────────────────
# INTENT DICTIONARY
# ─────────────────────────────────────────────
INTENTS = {
    "greeting": [
        "hello sir how are you", "hello maam how are you",
        "good morning sir", "good afternoon maam", "good evening sir",
        "am I speaking with the account holder",
        "is this the owner of this number",
        "can you hear me clearly",
        "this is a service call regarding your account",
        "this call is for account verification",
        "namaste sir", "namaste maam",
        "hello this is customer service",
        "hello I am calling regarding your bank account",
        "hello we are calling from bank support",
        "hello this is an official call",
        "hello this is customer care calling",
    ],
    "authority": [
        "I am calling from SBI bank", "I am calling from HDFC bank",
        "I am calling from ICICI bank", "I am calling from Axis bank",
        "I am calling from bank verification team",
        "I am calling from card security department",
        "I am calling from KYC verification department",
        "I am calling from bank security team",
        "This is the bank verification team",
        "This is your bank customer care",
        "This is UPI support team",
        "I am calling from payment verification team",
        "main bank se bol raha hoon",
        "main SBI bank se bol raha hoon",
        "main bank verification department se bol raha hoon",
        "main customer support se bol raha hoon",
        "ye bank verification call hai",
    ],
    "problem": [
        "your account has suspicious activity",
        "we detected unusual transactions",
        "your ATM card has been temporarily blocked",
        "your bank account is under review",
        "your card has been restricted",
        "someone attempted login to your account",
        "your account has security issues",
        "your KYC is incomplete", "your KYC update is pending",
        "your account is temporarily suspended",
        "your account has been locked",
        "your bank services may stop",
        "your account security is compromised",
        "aapka account block ho sakta hai",
        "aapka ATM card block ho gaya hai",
        "aapka KYC pending hai",
        "aapke account mein suspicious activity hai",
        "aapka account temporarily block ho gaya hai",
    ],
    "urgency": [
        "you must verify immediately", "this is very urgent",
        "immediate action is required", "please verify now",
        "your account will be blocked today",
        "your account will be permanently blocked",
        "this needs to be done right now",
        "you need to confirm immediately",
        "your banking service will stop",
        "your card will be blocked today",
        "your account will be frozen",
        "this is time sensitive",
        "verification must be done now",
        "this cannot be delayed",
        "abhi verify karna hoga", "turant verify karna padega",
        "warna account block ho jayega", "jaldi karna hoga",
    ],
    "data_request": [
        "tell me the OTP sent to your phone",
        "share the OTP you received",
        "read the verification code",
        "tell me the six digit code",
        "read the SMS you received",
        "confirm the number sent to your phone",
        "share the verification code with me",
        "tell me the security code",
        "share the OTP quickly",
        "tell me the code for verification",
        "confirm the SMS code",
        "read the message from the bank",
        "OTP bata dijiye", "OTP bol dijiye",
        "jo OTP aya hai wo bataiye",
        "SMS mein jo code aya hai wo boliye",
        "verification code bata dijiye",
        "mobile pe jo code aya hai wo boliye",
        "OTP share kar dijiye",
        "OTP confirm kar dijiye",
    ],
}

STAGE_MAP = {
    "greeting":     1,
    "authority":    2,
    "problem":      3,
    "urgency":      4,
    "data_request": 5,
}

STAGE_RISK = {
    0: 0.05,
    1: 0.10,
    2: 0.25,
    3: 0.50,
    4: 0.70,
    5: 0.95,
}

STAGE_LABELS = {
    0: "no match",
    1: "GREETING",
    2: "AUTHORITY",
    3: "PROBLEM",
    4: "URGENCY",
    5: "DATA REQUEST",
}


# ─────────────────────────────────────────────
# LOAD EMBEDDING MODEL
# ─────────────────────────────────────────────
print("Loading sentence embedding model...")
embedder        = SentenceTransformer("all-MiniLM-L6-v2")
intent_embeddings = {
    intent: embedder.encode(sentences)
    for intent, sentences in INTENTS.items()
}
print("Embedding model ready.\n")


# ─────────────────────────────────────────────
# WHISPER TRANSCRIPTION
# Saves audio to a temp WAV, calls whisper-cli,
# reads stdout for transcript.
# ─────────────────────────────────────────────
def transcribe(audio: np.ndarray, whisper_model: str = WHISPER_MODEL) -> str:
    """
    Write audio to a temp WAV file and run whisper-cli on it.
    Returns the transcript string (empty string on failure).
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    try:
        # sounddevice gives float32 — whisper-cli needs int16 WAV
        audio_int16 = (audio * 32767).astype(np.int16)
        wav.write(tmp_path, SR, audio_int16)

        result = subprocess.run(
            [
                WHISPER_CLI,
                "-m", MODEL_PATH,
                "-f", tmp_path,
                "-l", WHISPER_LANG,
                "--no-timestamps"
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # whisper-cli prints transcript to stdout
        transcript = result.stdout.strip()

        # Strip common whisper artifacts
        for artifact in ["[BLANK_AUDIO]", "(music)", "[Music]", "(Music)"]:
            transcript = transcript.replace(artifact, "").strip()

        return transcript

    except subprocess.TimeoutExpired:
        log.warning("Whisper timed out")
        return ""
    except FileNotFoundError:
        log.error(
            f"whisper-cli not found at '{WHISPER_CLI}'.\n"
            f"Set WHISPER_CLI at the top of this file to the full path."
        )
        return ""
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────
# INTENT + STAGE DETECTION
# ─────────────────────────────────────────────
def detect_intents(text: str) -> dict:
    """Cosine similarity of text against all intent example sentences."""
    text_emb = embedder.encode([text.lower().strip()])
    return {
        intent: float(np.max(cosine_similarity(text_emb, embs)))
        for intent, embs in intent_embeddings.items()
    }


def detect_stage(intent_scores: dict) -> tuple:
    """Return (stage_number, best_intent, best_score)."""
    best_intent = max(intent_scores, key=intent_scores.get)
    best_score  = intent_scores[best_intent]
    if best_score < SIMILARITY_GATE:
        return 0, best_intent, best_score
    return STAGE_MAP[best_intent], best_intent, best_score


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
class SessionState:
    def __init__(self):
        self.current_stage  = 0
        self.stage_history  = []   # list of (window_idx, stage, intent, score)
        self.max_stage_seen = 0

    def update(self, stage: int, intent: str, score: float, window_idx: int):
        if stage > 0:
            self.stage_history.append((window_idx, stage, intent, score))
            if stage > self.max_stage_seen:
                self.max_stage_seen = stage
            # Stage can only advance or stay — never go backward
            if stage > self.current_stage:
                self.current_stage = stage

    def risk(self) -> float:
        return STAGE_RISK.get(self.current_stage, 0.05)


# ─────────────────────────────────────────────
# MAIN REAL-TIME LOOP
# ─────────────────────────────────────────────
def run_realtime(window_sec: int = WINDOW_SEC,
                 whisper_model: str = WHISPER_MODEL) -> None:

    window_samples = int(window_sec * SR)
    state          = SessionState()
    window_idx     = 0

    print(f"\n{'='*65}")
    print(f"REAL-TIME SCAM CONVERSATION DETECTOR")
    print(f"Window: {window_sec}s  |  SR: {SR}Hz  |  Whisper: {whisper_model}")
    print(f"Stages: GREETING → AUTHORITY → PROBLEM → URGENCY → DATA_REQUEST")
    print(f"Press Ctrl+C to stop.")
    print(f"{'='*65}\n")

    try:
        while True:
            # ── Capture ───────────────────────────────────────────────
            print(f"  [{window_idx:03d}] Listening ({window_sec}s)...", end="\r")
            audio = sd.rec(
                frames     = window_samples,
                samplerate = SR,
                channels   = 1,
                dtype      = "float32"
            )
            sd.wait()
            audio = audio.flatten()

            # ── VAD — skip silent windows ─────────────────────────────
            speech = vad_filter(audio)
            if len(speech) < SR * MIN_SPEECH_S:
                print(f"  [{window_idx:03d}] skipped (silence)                      ")
                window_idx += 1
                continue

            # ── Whisper transcription ─────────────────────────────────
            print(f"  [{window_idx:03d}] Transcribing...          ", end="\r")
            transcript = transcribe(speech, whisper_model)

            if not transcript:
                print(f"  [{window_idx:03d}] no transcript                          ")
                window_idx += 1
                continue

            # ── Intent + stage detection ──────────────────────────────
            intent_scores       = detect_intents(transcript)
            stage, intent, score = detect_stage(intent_scores)
            state.update(stage, intent, score, window_idx)
            risk = state.risk()

            # ── Risk label ────────────────────────────────────────────
            if risk < 0.15:
                risk_label = "✅ safe"
            elif risk < 0.35:
                risk_label = "🟡 low risk"
            elif risk < 0.55:
                risk_label = "🟠 moderate"
            elif risk < 0.80:
                risk_label = "🔴 HIGH RISK"
            else:
                risk_label = "🚨 SCAM ALERT"

            stage_label = STAGE_LABELS.get(state.current_stage, "unknown")

            print(
                f"  [{window_idx:03d}] \"{transcript[:60]}\"\n"
                f"         Intent: {intent:<14}  Score: {score:.3f}  "
                f"Stage: {stage_label:<14}  "
                f"Risk: {risk:.2f}  {risk_label}"
            )

            if risk >= 0.80:
                print(f"\n  {'!'*55}")
                print(f"  ⚠  SCAM ALERT — Stage {state.current_stage}: {stage_label}")
                print(f"  {'!'*55}\n")

            window_idx += 1

    except KeyboardInterrupt:
        print(f"\n\n{'='*65}")
        print(f"  SESSION ENDED")
        print(f"  Windows processed : {window_idx}")
        print(f"  Final stage       : {STAGE_LABELS.get(state.current_stage, 'none')}")
        print(f"  Final risk        : {state.risk():.2f}")
        print(f"  Stage history     :")
        for w, s, i, sc in state.stage_history:
            print(f"    Window {w:03d} → {STAGE_LABELS[s]:<14} ({i}, sim={sc:.3f})")
        print(f"{'='*65}\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window",        type=int, default=WINDOW_SEC,
                        help="Capture window in seconds (default: 4)")
    parser.add_argument("--whisper-model", type=str, default=WHISPER_MODEL,
                        help="Whisper model size: tiny/base/small (default: base)")
    parser.add_argument("--whisper-path",  type=str, default=WHISPER_CLI,
                        help="Full path to whisper-cli executable")
    args = parser.parse_args()

    # Allow overriding whisper path from CLI
    if args.whisper_path != WHISPER_CLI:
        WHISPER_CLI = args.whisper_path

    run_realtime(window_sec=args.window, whisper_model=args.whisper_model)