"""Core Asynchronous Conversational Pipeline for EchoStream.

Orchestrates the full-duplex conversational voice loop:
1. Non-blocking audio capture (PyAudio callback -> asyncio.Queue)
2. Real-time Silero VAD frame evaluation
3. Instantaneous Barge-In Kill Switch (interrupts LLM/TTS and flushes queues if user speaks)
4. Offloaded Faster-Whisper local transcription (<120ms)
5. Groq Llama-3 token streaming (<150ms TTFT)
6. Asynchronous punctuation buffer chunking (exact millisecond sentence dispatch)
7. Edge-TTS streaming synthesis (<100ms TTFB)
8. Concurrent audio output playback
Target end-to-end response latency: <500ms
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Optional
from loguru import logger
import pyaudio

from services.vad import SileroVADService
from services.stt import FasterWhisperSTTService
from services.llm import GroqLLMService
from services.tts import EdgeTTSService
from utils.stream_buffers import sentence_stream_buffer, PunctuationBuffer


class EchoStreamPipeline:
    """High-velocity asynchronous streaming pipeline."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 512,  # 32ms per frame
        vad_threshold: float = 0.5,
        vad_silence_duration_ms: int = 350,
        whisper_model: str = "tiny.en",
        llm_model: str = "llama-3.1-8b-instant",
        tts_voice: str = "en-US-JennyNeural",
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.vad_threshold = vad_threshold
        self.vad_silence_duration_ms = vad_silence_duration_ms
        self.whisper_model = whisper_model
        self.llm_model = llm_model
        self.tts_voice = tts_voice

        # Core Services
        self.vad_service: Optional[SileroVADService] = None
        self.stt_service: Optional[FasterWhisperSTTService] = None
        self.llm_service: Optional[GroqLLMService] = None
        self.tts_service: Optional[EdgeTTSService] = None

        # Audio I/O Infrastructure
        self.py_audio: Optional[pyaudio.PyAudio] = None
        self.in_stream: Optional[pyaudio.Stream] = None
        self.out_stream: Optional[pyaudio.Stream] = None

        # Asynchronous Communication Channels
        self.input_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.output_queue: asyncio.Queue[bytes] = asyncio.Queue()

        # Concurrency & State Management
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.is_running: bool = False
        self.is_assistant_speaking: bool = False
        self.is_assistant_generating: bool = False

        # Active background processing tasks
        self._current_response_task: Optional[asyncio.Task] = None
        self._current_tts_task: Optional[asyncio.Task] = None
        self._playback_worker_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initializes all models, services, and PyAudio audio streams."""
        self.loop = asyncio.get_running_loop()
        logger.info("Initializing EchoStream sub-services...")

        # 1. Silero VAD
        self.vad_service = SileroVADService(
            sample_rate=self.sample_rate,
            threshold=self.vad_threshold,
            min_silence_duration_ms=self.vad_silence_duration_ms,
            chunk_size=self.chunk_size,
        )

        # 2. Faster-Whisper Local STT
        self.stt_service = FasterWhisperSTTService(
            model_size=self.whisper_model,
            device="cpu",
            compute_type="int8",
        )

        # 3. Groq Streaming LLM
        self.llm_service = GroqLLMService(
            model=self.llm_model,
        )

        # 4. Edge-TTS Engine
        self.tts_service = EdgeTTSService(
            voice=self.tts_voice,
            output_sample_rate=self.sample_rate,
        )

        # 5. Initialize PyAudio Hardware
        logger.info("Configuring PortAudio microphone & speaker streams...")
        self.py_audio = pyaudio.PyAudio()

        def _mic_callback(in_data, frame_count, time_info, status):
            if self.is_running and self.loop and not self.loop.is_closed():
                self.loop.call_soon_threadsafe(self.input_queue.put_nowait, in_data)
            return (None, pyaudio.paContinue)

        self.in_stream = self.py_audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=_mic_callback,
        )

        self.out_stream = self.py_audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size,
        )

        logger.success("EchoStream initialization complete. Ready for real-time voice.")

    def trigger_barge_in_kill_switch(self) -> None:
        """Instantaneously cancels ongoing LLM generation, halts TTS synthesis,

        and flushes the audio playback buffer when the user interrupts.
        """
        if not (self.is_assistant_speaking or self.is_assistant_generating):
            return

        logger.warning("⚡ [BARGE-IN KILL-SWITCH] User interrupted. Flushing pipeline...")

        # 1. Cancel active generation & TTS tasks
        if self._current_response_task and not self._current_response_task.done():
            self._current_response_task.cancel()

        if self._current_tts_task and not self._current_tts_task.done():
            self._current_tts_task.cancel()

        # 2. Flush audio output queue
        flushed_chunks = 0
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
                self.output_queue.task_done()
                flushed_chunks += 1
            except (asyncio.QueueEmpty, ValueError):
                break

        # 3. Reset pipeline state
        self.is_assistant_speaking = False
        self.is_assistant_generating = False
        logger.info(f"⚡ [KILL-SWITCH COMPLETE] Flushed {flushed_chunks} audio chunks.")

    async def _playback_worker(self) -> None:
        """Continuously pulls PCM audio frames from the output queue and sends them

        to PortAudio for real-time speaker playback.
        """
        while self.is_running:
            try:
                pcm_chunk = await self.output_queue.get()
                if pcm_chunk is None:
                    break

                self.is_assistant_speaking = True
                # Non-blocking write to speaker stream via background thread
                await asyncio.to_thread(self.out_stream.write, pcm_chunk)
                self.output_queue.task_done()

                if self.output_queue.empty() and not self.is_assistant_generating:
                    self.is_assistant_speaking = False

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Playback worker error: {e}")

    async def _process_user_turn(self, speech_bytes: bytes, speech_end_timestamp: float) -> None:
        """Coordinates STT -> Groq LLM -> Sentence Streaming -> Edge-TTS for a single turn."""
        try:
            self.is_assistant_generating = True

            # Step 1: Speech-to-Text
            stt_start = time.perf_counter()
            transcript = await self.stt_service.transcribe(speech_bytes)
            stt_latency = (time.perf_counter() - stt_start) * 1000

            if not transcript or not transcript.strip():
                logger.debug("Empty transcription, ignoring noise.")
                self.is_assistant_generating = False
                return

            print(f"\n👤 User: {transcript}")

            # Step 2: Groq Token Stream + Async Punctuation Buffering
            llm_start = time.perf_counter()
            token_gen = self.llm_service.stream_response(transcript)
            sentence_gen = sentence_stream_buffer(token_gen)

            first_sentence = True
            print("🤖 Assistant: ", end="", flush=True)

            # Step 3: Sentence-Level Streaming to TTS
            async for sentence in sentence_gen:
                if not sentence.strip():
                    continue

                print(f"{sentence} ", end="", flush=True)

                if first_sentence:
                    sentence_time = (time.perf_counter() - llm_start) * 1000
                    logger.debug(f"First sentence ready in {sentence_time:.1f}ms: \"{sentence}\"")

                # Dispatch sentence to Edge-TTS stream worker
                tts_start = time.perf_counter()
                async for pcm_chunk in self.tts_service.synthesize_sentence_stream(sentence):
                    if first_sentence:
                        # Log comprehensive sub-500ms latency benchmark
                        total_latency = (time.perf_counter() - speech_end_timestamp) * 1000
                        tts_ttfb = (time.perf_counter() - tts_start) * 1000
                        logger.success(
                            f"\n🎯 [LATENCY BREAKDOWN] Total: {total_latency:.1f}ms "
                            f"(STT: {stt_latency:.1f}ms | First Sentence: {sentence_time:.1f}ms | TTS TTFB: {tts_ttfb:.1f}ms)"
                        )
                        first_sentence = False

                    await self.output_queue.put(pcm_chunk)

            print()  # Newline after assistant speech

        except asyncio.CancelledError:
            logger.info("Turn generation pipeline cancelled.")
        except Exception as e:
            logger.error(f"Error during turn processing: {e}")
        finally:
            self.is_assistant_generating = False

    async def run(self) -> None:
        """Main non-blocking audio event loop."""
        self.is_running = True
        self.in_stream.start_stream()
        self.out_stream.start_stream()

        # Start playback worker
        self._playback_worker_task = asyncio.create_task(self._playback_worker())

        logger.info("🎤 EchoStream live. Listening for voice input...")

        try:
            while self.is_running:
                # Read mic chunk from asynchronous callback queue
                pcm_chunk = await self.input_queue.get()
                self.input_queue.task_done()

                # Process chunk through Silero VAD
                vad_result = self.vad_service.process_frame(pcm_chunk)

                if vad_result.state == "speech_start":
                    # Check for Barge-in interrupt
                    if self.is_assistant_speaking or self.is_assistant_generating:
                        self.trigger_barge_in_kill_switch()

                elif vad_result.state == "speech_end":
                    speech_end_time = time.perf_counter()
                    speech_audio = self.vad_service.get_accumulated_speech_audio()

                    # Launch turn processing task
                    self._current_response_task = asyncio.create_task(
                        self._process_user_turn(speech_audio, speech_end_time)
                    )

        except asyncio.CancelledError:
            logger.info("Pipeline event loop stopped.")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Cleanly tears down audio streams and background tasks."""
        logger.info("Shutting down EchoStream pipeline...")
        self.is_running = False

        if self._current_response_task and not self._current_response_task.done():
            self._current_response_task.cancel()

        if self._playback_worker_task and not self._playback_worker_task.done():
            self._playback_worker_task.cancel()

        if self.in_stream:
            self.in_stream.stop_stream()
            self.in_stream.close()

        if self.out_stream:
            self.out_stream.stop_stream()
            self.out_stream.close()

        if self.py_audio:
            self.py_audio.terminate()

        logger.info("Shutdown complete.")

