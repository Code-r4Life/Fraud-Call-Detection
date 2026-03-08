import librosa
import numpy as np
import webrtcvad
from scam_detection.config import SR, WINDOW_SEC, OVERLAP, VAD_MODE, VAD_FRAME_MS, MIN_SEG_S, HOP_LENGTH

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