# EchoStream ⚡
> **Ultra-Low Latency (<500ms) Conversational Voice AI with Sonic Pulse Orb Visualizer**  
> Built with a **100% Free, Open-Source Stack**: Silero VAD, Faster-Whisper, Groq LPU (`qwen/qwen3.8-27b`), Edge-TTS, FastAPI, and Web Audio API.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Groq LPU](https://img.shields.io/badge/Groq-LPU%20Inference-f55036.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌟 Highlights

- **Sub-500ms End-to-End Latency**: Human-like conversational response speed achieved through sentence-level streaming buffers.
- **Interactive Sonic Pulse Orb**: Real-time canvas visualizer with vibrating shockwave ripples and ambient halos that dynamically react to speech frequency and conversation states.
- **Zero-Feedback Headphone Audio**: Physically isolated Web Audio graph that prevents microphone monitoring loopback while providing crystal-clear playback.
- **Gapless Timeline Audio Engine**: Hardware-clocked ahead-of-time PCM scheduling with jitter pre-buffering that completely eliminates clicks, pops, and cracked audio artifacts.
- **Dual Runtime Modes**: Run either as a full-featured **Interactive Web Interface** on `http://localhost:8000` or as a **Terminal CLI Agent**.
- **Instant Barge-In & Interruption**: Full-duplex interruption filter that flushes pending audio the moment sustained speech is detected.

---

## 🏛 Architecture & Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                ECHOSTREAM PIPELINE                               │
│                                                                                  │
│   Browser Microphone                                    Browser Speaker          │
│          │                                                     ▲                 │
│          ▼ (16kHz PCM chunks)                                  │ (Gapless PCM)   │
│   [WebSocket /ws]                                       [WebSocket /ws]          │
│          │                                                     ▲                 │
│          ▼                                                     │                 │
│   [Silero VAD] (ONNX Runtime, ~1.1ms)                          │                 │
│          │                                                     │                 │
│   ┌──────┴──────────────┐                                      │                 │
│   │ [Speech End]        │ [Barge-In]                           │                 │
│   ▼                     ▼                                      │                 │
│ [Faster-Whisper]    KILL-SWITCH                                │                 │
│ (tiny.en, int8)     (Flush LLM & TTS)                          │                 │
│   │                                                            │                 │
│   ▼                                                            │                 │
│ [Groq LPU] ──(stream)──► [Sentence Buffer] ──► [Edge-TTS] ─────┘                 │
│ (<150ms TTFT)             (Punctuation split)   (PyAV Resampler)                 │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Measured Latency Breakdown (Target: <500ms)
| Pipeline Component | Technology | Latency |
|---|---|---|
| **Voice Activity Detection** | Silero VAD (ONNX) | ~1.1 ms per frame |
| **Speech-to-Text (STT)** | Faster-Whisper `tiny.en` (int8) | ~120 ms |
| **LLM Inference (TTFT)** | Groq `qwen/qwen3.8-27b` | ~106 - 150 ms |
| **Text-to-Speech (TTFB)** | Microsoft Edge-TTS + PyAV | ~110 - 140 ms |
| **End-to-End Turnaround** | Full Pipeline | **340 - 450 ms** |

---

## 📦 Project Structure

```
EchoStream/
├── .env.example              # Environment variables template
├── requirements.txt          # Production dependencies
├── server.py                 # FastAPI & WebSocket server for the Web UI
├── main.py                   # Terminal CLI voice agent
├── diagnose_audio.py         # Audio device diagnostics utility
├── frontend/
│   ├── voice.html            # Live full-duplex interactive voice interface
│   └── index.html            # Root web entrypoint
├── core/
│   ├── __init__.py
│   └── pipeline.py           # Full-duplex coordinator & barge-in logic
├── services/
│   ├── __init__.py
│   ├── vad.py                # Silero VAD speech state tracking
│   ├── stt.py                # Faster-Whisper local transcription
│   ├── llm.py                # Groq client with streaming & voice prompts
│   └── tts.py                # Edge-TTS synthesizer & PyAV PCM stream
├── utils/
│   ├── __init__.py
│   └── stream_buffers.py     # Sentence boundary punctuation chunker
└── tests/
    ├── test_stream_buffers.py# Buffer unit tests
    └── test_pipeline_e2e.py  # Latency benchmarks & E2E verification
```

---

## ⚡ Quickstart Guide

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Rayyan2005-git/EchoStream.git
cd EchoStream

# Create a Python 3.11 virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` and paste your free Groq API key:
```env
# Get a free key at https://console.groq.com/keys
GROQ_API_KEY=gsk_your_groq_api_key_here

# Audio & VAD Settings
SAMPLE_RATE=16000
CHUNK_SIZE=512
VAD_CONFIDENCE_THRESHOLD=0.5
VAD_SILENCE_DURATION_MS=400

# Whisper STT Settings
WHISPER_MODEL_SIZE=tiny.en
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

# Groq LLM Settings
LLM_MODEL=qwen/qwen3.8-27b

# Neural TTS Voice
TTS_VOICE=en-US-JennyNeural

# Barge-In Protection
ENABLE_BARGE_IN=false
```

---

## 🚀 Running EchoStream

### Option A: Interactive Web Interface (Recommended)

Launch the FastAPI & WebSocket server:
```bash
python server.py
```

Then open your browser at **[http://localhost:8000](http://localhost:8000)**:
1. Click **"Start Voice Session"** (or tap the glowing orb).
2. Allow microphone access.
3. Put on your headphones and speak naturally!
4. Watch the glowing Sonic Pulse orb vibrate to your voice and reply in real-time.

### Option B: Terminal CLI Mode

```bash
python main.py
```
Speaks directly through your system's default microphone and speakers via PyAudio.

---

## 🛠 Diagnostics & Verification

Test your audio hardware setup:
```bash
python diagnose_audio.py
```

Run latency verification benchmarks:
```bash
python tests/test_pipeline_e2e.py
```

---

## 📜 License

MIT License. Free to use, modify, and distribute.
