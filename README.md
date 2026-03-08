# 🔐 Real-Time Scam Call Detection System

### **Team: AUDIO**

**Members:**

- [Shinjan Saha](https://github.com/Code-r4Life/) - AI/ML Development
- [Arya Gupta](https://github.com/CyberKnight-cmd) - All rounder (Team Lead + System Design + Contribution in all other aspects)
- [Srijan Sarkar](https://github.com/Nameless-Seeker) - Android App Development
- [Pritam Paul](https://github.com/Pritam27112004) - AI/ML Developement

This project was developed during a **national-level hackathon hosted at Jadavpur University** (6–8 March).

Out of **2000+ registrations**, only **30 teams** were selected for the offline hackathon.
Our team ranked **Top 4 during the first round (PPT submission)** and progressed to the main hackathon stage.

Although we did not reach the final pitching round, the project successfully demonstrated a **fully functional real-time scam call detection pipeline** during the live mentor demo.

The system focuses on **protecting vulnerable users (especially elderly people)** from scam calls by detecting fraudulent patterns **during live conversations** using AI and speech analysis.

---

# 📌 1. Project Overview

This repository contains a **real-time AI system capable of detecting scam calls during live conversations**.

The system analyzes audio streams in **4-second chunks**, extracting multiple layers of features and combining the results of **four machine learning models** to produce a final **risk score**.

The architecture combines:

* Speech signal processing
* Acoustic feature analysis
* Prosody detection
* Scam keyword detection
* Speech-to-text semantic analysis

The final system outputs a **real-time risk score** indicating whether a call is safe or potentially fraudulent.

## Youtube Pitch
[![Watch the video](https://img.youtube.com/vi/TFx9VcgPPic/0.jpg)](https://youtu.be/TFx9VcgPPic)

## Repository Scope

This repository focuses on the **Machine Learning components** of the complete Scam Call Detection System.

It includes:

- Training pipelines for the ML models
- Dataset preprocessing and feature extraction
- Model experimentation notebooks
- Inference scripts for testing models
- Evaluation utilities

The **full end-to-end system** also includes:

- Android VoIP application for capturing call audio
- FastAPI backend server for real-time inference
- WebSocket streaming infrastructure

Those components are maintained in the **main system repository**, while this repository focuses specifically on the **ML research and development layer**.

🔗 **Full System Repository:**   [Spam Call Detection System](https://github.com/CyberKnight-cmd/spam-call-detection)

---

# ⚙️ 2. Key Features

✔️ **Real-time scam detection during live calls**

✔️ **Multi-model AI ensemble**

✔️ **Speech signal processing using MFCC features**

✔️ **Urgency detection using prosody analysis**

✔️ **Scam keyword detection using wakeword-style models**

✔️ **Conversation stage tracking using Whisper transcription**

✔️ **FastAPI backend with WebSocket streaming**

✔️ **Android VoIP integration for live audio streaming**

---

# 🧠 3. System Architecture

## Architecture

```
┌─────────────────┐         WebSocket          ┌──────────────────┐
│  Android App    │ ◄────────────────────────► │  Python Backend  │
│  (Kotlin)       │    4-sec audio chunks      │  (FastAPI)       │
└─────────────────┘                            └──────────────────┘
        │                                               │
        │                                               ▼
        │                                      ┌────────────────┐
        │                                      │  4 AI Models   │
        │                                      │  - Phoneme CNN │
        │                                      │  - Urgency XGB │
        │                                      │  - Repetition  │
        │                                      │  - Stage Track │
        │                                      └────────────────┘
        │                                               │
        │                                               ▼
        │                                      ┌────────────────┐
        │                                      │  Risk Score    │
        │ ◄────────────────────────────────────│  (0.0 - 1.0)   │
        │         Risk Assessment              └────────────────┘
        ▼
┌─────────────────┐
│  User Alert     │
│  ✅ SAFE        │
│  🟡 LOW RISK    │
│  🟠 MODERATE    │
│  🔴 HIGH RISK   │
│  🚨 SCAM ALERT  │
└─────────────────┘
```

---

# 🧩 4. Machine Learning Architecture

The detection system is composed of **four independent AI layers**, each focusing on a different aspect of scam behavior.

---

## 4.1 Phoneme Layer (MFCC + CNN)

This model analyzes **phonetic patterns in speech**.

Input:

* 4-second audio chunk
* 120 MFCC features

Model:

* CNN-based binary classifier

Output:

* Probability of scam speech patterns.

---

## 4.2 Prosody Layer (Urgency Detection)

Scammers often speak with **high urgency and pressure**.

This layer analyzes:

* Pitch variation
* Speech energy
* Speech rate

Output score example:

```
1 → Normal speech
2 → Slight urgency
3 → High urgency (potential scam)
```

---

## 4.3 Repetition Detection Layer

This layer detects **scam-related keywords** using a wakeword-style detection model.

Example scam keywords:

```
OTP
UPI
Refund
Lottery
Reward
Emergency
Code
```

Since no public dataset existed for scam wakewords, we **created our own dataset** using recordings from friends and family.

Dataset statistics:

* 15 keyword classes
* 30+ recordings per class
* ~1200 positive samples
* Large negative dataset (Google Speech Commands + noise)

Model performance:

* **Validation Accuracy:** 98.21%
* **Test Accuracy:** 98.44%

---

## 4.4 Semantic Layer (Speech-to-Text Analysis)

To reduce false positives, we implemented a **speech transcription stage**.

Using **Whisper / Whisper.cpp**, the system transcribes each audio chunk and analyzes the text for **scam conversation patterns**.

Example scam phrases:

```
Hello sir, I am calling from SBI bank
Sir I have sent you an OTP
You have won a lottery
Can you confirm your card details?
```

The system tracks **conversation stages**, such as:

1. Greeting
2. Authority claim
3. Problem creation
4. Urgency
5. Data request

Each stage increases the **risk score**.

---

# ⚡ 5. Risk Fusion Engine

Outputs from all four models are combined using a **weighted linear fusion**.

```
Final Risk Score =
w1 × semantic_score +
w2 × repetition_score +
w3 × phoneme_score +
w4 × urgency_score
```

The result is mapped into risk categories:

```
✅ SAFE
🟡 LOW RISK
🟠 MODERATE RISK
🔴 HIGH RISK
🚨 SCAM ALERT
```

---

# 📡 6. Real-Time Audio Pipeline

The backend processes audio using the following pipeline:

1. Receive 4-second audio chunk
2. Apply **WebRTC Voice Activity Detection**
3. Remove silence
4. Extract audio features
5. Run models **in parallel**
6. Combine results
7. Send risk prediction back to mobile app

Running models asynchronously **reduces latency and improves real-time performance**.

---

# 📁 7. Repository Structure

```
FRAUD-CALL-DETECTION/
│
├── features/                         # Raw audio features
│   ├── NORMAL_CALLS/                 # Normal call features
│   └── SCAM_CALLS/                   # Scam call features
│
├── mfcc_labels.csv                   # MFCC feature labels
├── prosody_labels.csv                # Prosody feature labels
│
├── models/                           # Trained ML models
│   ├── best_phoneme_model.keras
│   ├── best_prosody_xgb_model.pkl
│   └── best_repetition_model.keras
│
├── notebooks/                        # Training notebooks
│   ├── phoneme_layer.ipynb
│   ├── prosody_layer.ipynb
│   ├── repetition_layer.ipynb
│   └── scam_detection_whisper_pipeline.ipynb
│
├── rep_features/                  # Repetition detection features
├── rep_features/                  # Repetition detection features
│   ├── train/                     # Train features
│   ├── val/                       # Validation features
│   └── test/                      # Test features
│
├── train_labels.csv              # Train labels
├── val_labels.csv                # Validation labels
├── test_labels.csv               # Test labels
│
├── scam_detection/                   # Core inference pipeline
│   ├── __init__.py
│   ├── audio_pipeline.py
│   ├── config.py
│   ├── feature_extraction.py
│   └── main.py
│
├── convert.py                        # Dataset conversion utilities
├── preprocessing.py                  # Dataset preprocessing
├── repetition_preprocessing.py       # Repetition dataset preprocessing
├── realtime_whisper.py               # Whisper-based transcription pipeline
├── final_inference.py                # End-to-end inference testing
│
├── ASA_Meeting_Spam_Detection_in_Voice.pdf  # Research reference
│
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

---

# 📊 8. Technologies Used

### Backend

* FastAPI
* TensorFlow / Keras
* XGBoost
* Librosa
* WebRTC VAD
* Whisper.cpp
* Sentence Transformers
* NumPy / Pandas

⚠️ Android components are part of the full system repository.

---

# 🔮 9. Future Improvements

* Multi-language scam detection (Hindi + regional languages)
* On-device inference with TensorFlow Lite
* Larger scam speech dataset
* Sentiment-based conversation analysis
* Integration with telecom spam detection systems

---

# 📬 Interested in a Similar Project?

I build smart, ML-integrated applications and responsive web platforms. Let’s build something powerful together!

📧 shinjansaha00@gmail.com

🔗 [LinkedIn Profile](https://www.linkedin.com/in/shinjan-saha-1bb744319/)