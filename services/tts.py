"""Edge-TTS Text-to-Speech (TTS) Service.

Asynchronously streams synthetic speech using Microsoft Edge's neural TTS engine.
Converts generated audio to PCM in memory and streams chunks immediately to the
playback queue, minimizing latency. Supports instantaneous cancellation on barge-in.
"""

from __future__ import annotations

import asyncio
import io
import time
from typing import AsyncGenerator, Optional
from loguru import logger
import edge_tts


class EdgeTTSService:
    """Ultra-low latency streaming TTS service using Edge-TTS."""

    def __init__(
        self,
        voice: str = "en-US-JennyNeural",
        output_sample_rate: int = 16000,
        rate_adjustment: str = "+10%",  # Slightly faster speech reduces conversational latency
    ):
        self.voice = voice
        self.output_sample_rate = output_sample_rate
        self.rate_adjustment = rate_adjustment

    def _decode_mp3_to_pcm(self, mp3_bytes: bytes) -> bytes:
        """Decodes MP3 bytes into 16-bit mono PCM bytes at output_sample_rate using PyAV."""
        if not mp3_bytes:
            return b""

        import av

        input_buffer = io.BytesIO(mp3_bytes)
        container = av.open(input_buffer)
        resampler = av.AudioResampler(
            format="s16", layout="mono", rate=self.output_sample_rate
        )

        pcm_frames = []
        for frame in container.decode(audio=0):
            resampled = resampler.resample(frame)
            for rf in resampled:
                pcm_frames.append(rf.to_ndarray().tobytes())

        return b"".join(pcm_frames)

    async def synthesize_sentence_stream(
        self,
        sentence: str,
    ) -> AsyncGenerator[bytes, None]:
        """Synthesizes a single sentence and yields decoded PCM audio chunks on-the-fly as they arrive.

        Measures Time-To-First-Audio-Byte (TTFB) for latency profiling.
        """
        if not sentence or not sentence.strip():
            return

        clean_text = sentence.strip()
        start_time = time.perf_counter()
        first_chunk_received = False

        try:
            import av

            codec = av.CodecContext.create("mp3", "r")
            resampler = av.AudioResampler(
                format="s16", layout="mono", rate=self.output_sample_rate
            )

            communicate = edge_tts.Communicate(
                text=clean_text,
                voice=self.voice,
                rate=self.rate_adjustment,
            )

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    packets = codec.parse(chunk["data"])
                    for packet in packets:
                        for frame in codec.decode(packet):
                            for rf in resampler.resample(frame):
                                pcm_chunk = rf.to_ndarray().tobytes()
                                if pcm_chunk:
                                    if not first_chunk_received:
                                        ttfb_ms = (time.perf_counter() - start_time) * 1000
                                        logger.info(
                                            f"TTS TTFB (Time To First Audio Byte): {ttfb_ms:.1f}ms for: \"{clean_text[:30]}...\""
                                        )
                                        first_chunk_received = True
                                    yield pcm_chunk

            # Flush any remaining frames from decoder
            for frame in codec.decode():
                for rf in resampler.resample(frame):
                    pcm_chunk = rf.to_ndarray().tobytes()
                    if pcm_chunk:
                        yield pcm_chunk

        except asyncio.CancelledError:
            logger.info("TTS synthesis cancelled (Barge-in / Kill Switch).")
            raise
        except Exception as e:
            logger.error(f"Edge-TTS synthesis error: {e}")
            raise

    async def synthesize_to_pcm(self, text: str) -> bytes:
        """Helper to synthesize complete text to PCM bytes directly."""
        chunks = []
        async for chunk in self.synthesize_sentence_stream(text):
            chunks.append(chunk)
        return b"".join(chunks)

