"""Faster-Whisper Speech-to-Text (STT) Service.

Ingests raw 16kHz PCM audio buffers and performs ultra-fast local transcription
using CTranslate2 / Faster-Whisper. Inference is offloaded to a background thread
to ensure the main asyncio audio event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional
from loguru import logger
import numpy as np


class FasterWhisperSTTService:
    """Ultra-low-latency local transcription service using Faster-Whisper."""

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,  # Greedy search for lowest latency
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size

        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads Faster-Whisper model into memory."""
        logger.info(
            f"Loading Faster-Whisper ({self.model_size}, device={self.device}, compute={self.compute_type})..."
        )
        from faster_whisper import WhisperModel

        start_time = time.perf_counter()
        # Automatically fall back to float32 if int8 is not supported on certain hardware
        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as e:
            logger.warning(
                f"Failed to load with compute_type={self.compute_type}: {e}. Retrying with float32..."
            )
            self.compute_type = "float32"
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="float32",
            )

        load_elapsed = (time.perf_counter() - start_time) * 1000
        logger.success(f"Faster-Whisper model loaded in {load_elapsed:.1f}ms.")

    def _transcribe_sync(self, audio_float32: np.ndarray) -> str:
        """Synchronous transcription execution on float32 numpy buffer."""
        if len(audio_float32) == 0:
            return ""

        # Faster-whisper accepts normalized float32 numpy array directly
        segments, _ = self.model.transcribe(
            audio_float32,
            beam_size=self.beam_size,
            language="en",
            task="transcribe",
            vad_filter=False,  # Already filtered by upstream Silero VAD
            condition_on_previous_text=False,
            temperature=0.0,
        )

        text_pieces = [segment.text.strip() for segment in segments]
        transcription = " ".join(filter(None, text_pieces)).strip()
        return transcription

    async def transcribe(self, pcm_bytes: bytes) -> str:
        """Asynchronously transcribes 16kHz 16-bit mono PCM bytes without blocking

        the event loop.
        """
        if not pcm_bytes or len(pcm_bytes) < 1600:  # Ignore tiny noise blips (<50ms)
            return ""

        start_time = time.perf_counter()

        # Convert raw int16 PCM bytes to float32 numpy array
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        # Execute CTranslate2 C++ inference in a worker thread
        text = await asyncio.to_thread(self._transcribe_sync, audio_float32)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if text:
            logger.info(f"STT Transcribed in {elapsed_ms:.1f}ms: \"{text}\"")

        return text

