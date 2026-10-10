"""Voice Activity Detection (VAD) Service using Silero VAD.

Provides real-time, low-latency voice activity detection for 16kHz PCM audio streams.
Detects speech onset, continuous speech, and speech termination (silence threshold),
triggering barge-in events and speech buffers for STT.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Deque, Literal, Optional
from loguru import logger
import numpy as np


@dataclass
class VADResult:
    """Result of processing a single audio frame through VAD."""

    probability: float
    is_speech: bool
    state: Literal["silence", "speech_start", "speaking", "speech_end"]
    audio_frame: bytes


class SileroVADService:
    """Manages Silero VAD model inference and stateful speech tracking."""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 400,
        speech_pad_ms: int = 128,
        chunk_size: int = 512,  # 512 samples = 32ms at 16kHz
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.chunk_size = chunk_size

        # Calculate frame counts
        # At 16kHz, 512 samples = 32ms
        self.ms_per_frame = (self.chunk_size / self.sample_rate) * 1000.0
        self.silence_frames_threshold = int(self.min_silence_duration_ms / self.ms_per_frame)
        self.pad_frames_count = max(1, int(self.speech_pad_ms / self.ms_per_frame))

        # Model instance
        self.model = None
        self._is_onnx = False

        # Speech state
        self._is_speaking = False
        self._silence_counter = 0
        self._pre_speech_ring_buffer: Deque[bytes] = collections.deque(
            maxlen=self.pad_frames_count
        )
        self._current_speech_chunks: list[bytes] = []

        self._load_model()

    def _load_model(self) -> None:
        """Loads Silero VAD via silero_vad or torch hub."""
        logger.info("Initializing Silero VAD model...")
        try:
            from silero_vad import load_silero_vad
            try:
                # Prefer ONNX for minimum CPU latency and no torch overhead
                self.model = load_silero_vad(onnx=True)
                self._is_onnx = True
                logger.success("Silero VAD loaded successfully (ONNX runtime).")
            except Exception as e:
                logger.warning(f"ONNX Silero load failed, falling back to Torch: {e}")
                self.model = load_silero_vad(onnx=False)
                self._is_onnx = False
                logger.success("Silero VAD loaded successfully (Torch).")
        except ImportError:
            # Fallback to torch hub
            import torch
            logger.info("silero_vad package not found, loading from torch.hub...")
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,
            )
            self.model = model
            self._is_onnx = True
            logger.success("Silero VAD loaded from torch.hub.")

    def reset(self) -> None:
        """Resets internal state of VAD model and speech buffers."""
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()
        self._is_speaking = False
        self._silence_counter = 0
        self._pre_speech_ring_buffer.clear()
        self._current_speech_chunks.clear()

    def process_frame(self, pcm_bytes: bytes) -> VADResult:
        """Processes a single raw PCM frame (16-bit signed, mono, 16000Hz).

        Expects `chunk_size` samples (e.g. 512 samples = 1024 bytes).
        """
        # Convert bytes to normalized float32 numpy array
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        if len(audio_float32) != self.chunk_size:
            # Pad or trim if needed
            if len(audio_float32) < self.chunk_size:
                audio_float32 = np.pad(
                    audio_float32, (0, self.chunk_size - len(audio_float32))
                )
            else:
                audio_float32 = audio_float32[: self.chunk_size]

        # Model inference
        prob = self._infer(audio_float32)
        is_speech = prob >= self.threshold

        state: Literal["silence", "speech_start", "speaking", "speech_end"] = "silence"

        if is_speech:
            self._silence_counter = 0
            if not self._is_speaking:
                self._is_speaking = True
                state = "speech_start"
                # Include pre-speech buffer so early consonants are not clipped
                self._current_speech_chunks = list(self._pre_speech_ring_buffer)
                self._current_speech_chunks.append(pcm_bytes)
            else:
                state = "speaking"
                self._current_speech_chunks.append(pcm_bytes)
        else:
            if self._is_speaking:
                self._silence_counter += 1
                self._current_speech_chunks.append(pcm_bytes)
                if self._silence_counter >= self.silence_frames_threshold:
                    self._is_speaking = False
                    state = "speech_end"
                else:
                    state = "speaking"
            else:
                self._pre_speech_ring_buffer.append(pcm_bytes)
                state = "silence"

        return VADResult(
            probability=prob,
            is_speech=is_speech,
            state=state,
            audio_frame=pcm_bytes,
        )

    def get_accumulated_speech_audio(self) -> bytes:
        """Returns the full accumulated audio bytes for the completed speech segment

        and clears the internal speech buffer.
        """
        collected = b"".join(self._current_speech_chunks)
        self._current_speech_chunks.clear()
        return collected

    def _infer(self, audio_float32: np.ndarray) -> float:
        """Run single chunk inference."""
        import torch

        tensor_data = torch.from_numpy(audio_float32)
        # silero-vad expects tensor of shape [1, N] or [N] depending on version
        with torch.no_grad():
            output = self.model(tensor_data, self.sample_rate)
            if isinstance(output, torch.Tensor):
                return float(output.item())
            return float(output)

