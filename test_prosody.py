import numpy as np
import librosa
import sounddevice as sd
import time

# =============================
# Configuration
# =============================

sr = 16000
chunk_duration = 0.5

# =============================
# Feature Extraction
# =============================

def extract_features(audio, sr):

    # Pitch
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)

    pitch_vals = []
    for i in range(pitches.shape[1]):
        idx = magnitudes[:, i].argmax()
        pitch = pitches[idx, i]
        if pitch > 0:
            pitch_vals.append(pitch)

    pitch_vals = np.array(pitch_vals)

    pitch_mean = np.mean(pitch_vals) if len(pitch_vals) > 0 else 0
    pitch_std = np.std(pitch_vals) if len(pitch_vals) > 0 else 0

    # Energy
    rms = librosa.feature.rms(y=audio)[0]
    energy_mean = np.mean(rms)
    energy_var = np.var(rms)

    # Spectral centroid
    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(y=audio, sr=sr)
    )

    # Speech rate approximation
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)

    peaks = librosa.util.peak_pick(
        onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=5
    )

    duration = len(audio) / sr
    speech_rate = len(peaks) / duration if duration > 0 else 0

    return {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "energy_mean": energy_mean,
        "energy_var": energy_var,
        "spectral_centroid": spectral_centroid,
        "speech_rate": speech_rate
    }



# =============================
# Urgency Detection
# =============================

def compute_urgency(features, baseline):
    score = 0

    # Hard silence gate — absolute floor, not relative
    if features["energy_mean"] < 0.005:
        return 0, False

    # Relative gate — must be meaningfully louder than baseline
    if baseline["energy_mean"] > 0.001:
        if features["energy_mean"] < baseline["energy_mean"] * 1.3:
            return 0, False

    if features["energy_mean"] > max(baseline["energy_mean"] * 1.8, 0.002):
        score += 1                                     # was 1.5 / 0.0005

    if features["pitch_mean"] > baseline["pitch_mean"] * 1.3:
        score += 1                                     # was 1.2

    if features["pitch_std"] > baseline["pitch_std"] * 1.5:
        score += 1                                     # was 1.3

    if features["speech_rate"] > baseline["speech_rate"] * 1.5:
        score += 1                                     # was 1.3

    if features["spectral_centroid"] > baseline["spectral_centroid"] * 1.25:
        score += 1                                     # was 1.15

    urgent = score >= 4                                # was 2

    return score, urgent


# =============================
# Main Real-Time Loop
# =============================

print("Listening continuously... Press Ctrl+C to stop")

try:
    while True:

        audio = sd.rec(int(chunk_duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()

        audio = audio.flatten()

        features = extract_features(audio, sr)

        # Averages after many calculations for Mobile devices in silence
        baseline = {
            "pitch_mean": 372.61,
            "pitch_std": 0.0,
            "energy_mean": 3.23e-6,
            "energy_var": 1.33e-9,
            "spectral_centroid": 49257.50,
            "speech_rate": 12.5
        }

        score, urgent = compute_urgency(features, baseline)

        if urgent:
            print(f"⚠️ URGENT SPEECH DETECTED! (Score: {score})")
            print(f"   Pitch: {features['pitch_mean']:.1f} | Energy: {features['energy_mean']:.4f} | Rate: {features['speech_rate']:.1f}")
        elif score > 0:
            print(f"[Score: {score}] Pitch: {features['pitch_mean']:.1f} | Energy: {features['energy_mean']:.4f} | Rate: {features['speech_rate']:.1f}")

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopped listening.")