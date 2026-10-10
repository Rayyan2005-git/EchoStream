"""EchoStream Entry Point & Session Initialization.

Initializes the ultra-low latency voice conversational pipeline, loads configuration
from .env, verifies audio peripherals, and runs the asynchronous full-duplex session.
"""

from __future__ import annotations

import asyncio
import os
import sys

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env
load_dotenv()

from core.pipeline import EchoStreamPipeline


def print_banner() -> None:
    """Displays EchoStream startup banner and system configuration."""
    banner = """
====================================================================
  _____      _            _____ _                               
 | ____|___ | |__   ___  / ____| |_ _ __ ___  __ _ _ __ ___    
 |  _| / __|| '_ \ / _ \ \___ \| __| '__/ _ \/ _` | '_ ` _ \   
 | |___| (__| | | | (_) | ___) | |_| | |  __/ (_| | | | | | |  
 |_____|\___|_| |_|\___/ |____/ \__|_|  \___|\__,_|_| |_| |_|  
====================================================================
 Target Latency  : < 500ms End-to-End
 Pipeline Stack  : Free & Open-Source (Silero + Whisper + Groq + Edge-TTS)
 Duplex Mode     : Full-Duplex with Instant Barge-In Kill Switch
 Streaming Mode  : Non-Blocking Async Sentence-Level Punctuation Buffer
====================================================================
"""
    print(banner)


async def main() -> None:
    """Initializes and runs the real-time conversational agent."""
    print_banner()

    # Verify Groq API Key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key.startswith("gsk_your_groq_api_key"):
        logger.error(
            "GROQ_API_KEY is not configured! Please add your free key to .env file."
        )
        logger.info("Get a free key instantly at: https://console.groq.com/keys")
        print("\nPress Ctrl+C to exit and configure .env.")
        # Proceeding still allows hardware/VAD/TTS inspection

    # Load configuration options
    sample_rate = int(os.getenv("SAMPLE_RATE", "16000"))
    chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
    vad_threshold = float(os.getenv("VAD_CONFIDENCE_THRESHOLD", "0.5"))
    vad_silence_ms = int(os.getenv("VAD_SILENCE_DURATION_MS", "350"))
    whisper_model = os.getenv("WHISPER_MODEL_SIZE", "tiny.en")
    llm_model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    tts_voice = os.getenv("TTS_VOICE", "en-US-JennyNeural")

    logger.info(f"Configuring pipeline: VAD Silence={vad_silence_ms}ms, STT={whisper_model}, LLM={llm_model}, TTS={tts_voice}")

    pipeline = EchoStreamPipeline(
        sample_rate=sample_rate,
        chunk_size=chunk_size,
        vad_threshold=vad_threshold,
        vad_silence_duration_ms=vad_silence_ms,
        whisper_model=whisper_model,
        llm_model=llm_model,
        tts_voice=tts_voice,
    )

    try:
        await pipeline.initialize()
        await pipeline.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupted by user. Exiting gracefully...")
    except Exception as e:
        logger.exception(f"Fatal error in pipeline session: {e}")
    finally:
        await pipeline.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)

