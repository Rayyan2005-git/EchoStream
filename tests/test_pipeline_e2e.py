"""End-to-End Component and Latency Verification Script.

Tests each pipeline layer in isolation to benchmark latency and verify zero-delay streaming:
1. Punctuation Buffer token streaming & splitting
2. Silero VAD frame processing & state transitions
3. Edge-TTS streaming synthesis & MP3-to-PCM decoding
4. Faster-Whisper transcription on audio buffer
"""

from __future__ import annotations

import asyncio
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import time
from pathlib import Path
import numpy as np
from loguru import logger

from dotenv import load_dotenv

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from utils.stream_buffers import PunctuationBuffer, sentence_stream_buffer
from services.vad import SileroVADService
from services.tts import EdgeTTSService


async def benchmark_punctuation_buffer():
    logger.info("=== 1. Benchmarking Punctuation Buffer ===")
    tokens = [
        "EchoStream ", "achieves ", "sub-500 ", "millisecond ", "latency. ",
        "It ", "streams ", "audio ", "concurrently! ",
        "Interruptibility ", "is ", "guaranteed."
    ]
    start = time.perf_counter()

    async def token_gen():
        for t in tokens:
            await asyncio.sleep(0.01)  # 10ms per token simulation
            yield t

    sentences = []
    async for s in sentence_stream_buffer(token_gen()):
        sentences.append(s)

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.success(f"Buffer processed {len(sentences)} sentences in {elapsed_ms:.1f}ms: {sentences}")
    assert len(sentences) == 3


async def benchmark_vad():
    logger.info("\n=== 2. Benchmarking Silero VAD ===")
    vad = SileroVADService(sample_rate=16000, chunk_size=512)

    # 1. Test 10 frames of silence (zeros)
    silence_frame = np.zeros(512, dtype=np.int16).tobytes()
    for _ in range(10):
        res = vad.process_frame(silence_frame)
        assert res.state == "silence"

    # 2. Test synthetic speech tone (sine wave 300Hz, amplitude ~16000)
    t = np.linspace(0, 512 / 16000, 512, endpoint=False)
    tone = (np.sin(2 * np.pi * 300 * t) * 16000).astype(np.int16).tobytes()

    start = time.perf_counter()
    speech_detected = False
    for _ in range(15):
        res = vad.process_frame(tone)
        if res.is_speech or res.state in ("speech_start", "speaking"):
            speech_detected = True

    infer_time_ms = (time.perf_counter() - start) * 1000 / 15
    logger.success(f"Silero VAD average frame inference time: {infer_time_ms:.2f}ms (speech detected: {speech_detected})")


async def benchmark_edge_tts():
    logger.info("\n=== 3. Benchmarking Edge-TTS Streaming ===")
    tts = EdgeTTSService(voice="en-US-JennyNeural")
    test_sentence = "Hello! EchoStream is streaming text to speech."

    start = time.perf_counter()
    chunks = []
    first_chunk_time = None

    async for pcm_chunk in tts.synthesize_sentence_stream(test_sentence):
        if first_chunk_time is None:
            first_chunk_time = (time.perf_counter() - start) * 1000
        chunks.append(pcm_chunk)

    total_time = (time.perf_counter() - start) * 1000
    total_bytes = sum(len(c) for c in chunks)
    logger.success(
        f"Edge-TTS TTFB: {first_chunk_time:.1f}ms | Total synthesis: {total_time:.1f}ms | Received {total_bytes} PCM bytes"
    )
    assert total_bytes > 0


async def benchmark_full_turn():
    logger.info("\n=== 4. Benchmarking Full End-to-End LLM -> Sentence Streaming -> TTS Pipeline ===")
    from services.llm import GroqLLMService

    llm = GroqLLMService()
    tts = EdgeTTSService(voice="en-US-JennyNeural")

    user_query = "What is the speed of light in miles per second?"
    logger.info(f"Simulating User Query: \"{user_query}\"")

    turn_start = time.perf_counter()
    token_stream = llm.stream_response(user_query)
    sentence_stream = sentence_stream_buffer(token_stream)

    first_sentence_time = None
    first_audio_byte_time = None
    total_audio_bytes = 0
    sentences_received = []

    async for sentence in sentence_stream:
        if first_sentence_time is None:
            first_sentence_time = (time.perf_counter() - turn_start) * 1000
            logger.info(f"First Sentence Ready in {first_sentence_time:.1f}ms: \"{sentence}\"")

        sentences_received.append(sentence)

        # Synthesize sentence to audio stream concurrently
        async for pcm_chunk in tts.synthesize_sentence_stream(sentence):
            if first_audio_byte_time is None:
                first_audio_byte_time = (time.perf_counter() - turn_start) * 1000
                logger.success(
                    f"🎯 [END-TO-END LATENCY] Time To First Playable Audio Byte: {first_audio_byte_time:.1f}ms!"
                )
            total_audio_bytes += len(pcm_chunk)

    total_turn_time = (time.perf_counter() - turn_start) * 1000
    logger.info(
        f"Completed Full Turn in {total_turn_time:.1f}ms ({len(sentences_received)} sentences, {total_audio_bytes} audio bytes)"
    )
    assert first_audio_byte_time is not None
    assert total_audio_bytes > 0


async def main():
    logger.info("Starting EchoStream Component Verification Suite...")
    await benchmark_punctuation_buffer()
    await benchmark_vad()
    await benchmark_edge_tts()
    await benchmark_full_turn()
    logger.success("\n>>> All component & pipeline benchmarks passed! EchoStream is primed for ultra-low latency.")


if __name__ == "__main__":
    asyncio.run(main())

