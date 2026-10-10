"""EchoStream FastAPI & WebSocket Server.

Bridges the browser frontend with the real-time Python pipeline:
- Serves the frontend web app at http://localhost:8000
- WebSocket /ws for full-duplex binary audio streaming:
  - Browser mic PCM in -> Silero VAD -> Faster-Whisper STT -> Groq LLM -> Edge-TTS -> Browser speaker PCM out
- Real-time state synchronization (idle, listening, thinking, speaking) & latency profiling
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from services.vad import SileroVADService
from services.stt import FasterWhisperSTTService
from services.llm import GroqLLMService
from services.tts import EdgeTTSService
from utils.stream_buffers import sentence_stream_buffer

app = FastAPI(title="EchoStream AI Voice Agent")

# Mount frontend directory for static assets
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Shared cached models for fast connection handling
vad_service: Optional[SileroVADService] = None
stt_service: Optional[FasterWhisperSTTService] = None


@app.on_event("startup")
async def startup_event():
    global vad_service, stt_service
    logger.info("Initializing EchoStream VAD & STT models for Web Server...")
    vad_service = SileroVADService(
        sample_rate=16000,
        threshold=float(os.getenv("VAD_CONFIDENCE_THRESHOLD", "0.5")),
        min_silence_duration_ms=int(os.getenv("VAD_SILENCE_DURATION_MS", "350")),
        chunk_size=512,
    )
    stt_service = FasterWhisperSTTService(
        model_size=os.getenv("WHISPER_MODEL_SIZE", "tiny.en"),
        device=os.getenv("WHISPER_DEVICE", "cpu"),
        compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
    )
    logger.success("EchoStream models loaded and ready for WebSockets.")


@app.get("/")
async def serve_index():
    """Serves the main interactive EchoStream web app."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.websocket("/ws")
async def websocket_voice_endpoint(websocket: WebSocket):
    """Full-Duplex Voice WebSocket Connection.

    Receives raw 16kHz PCM audio buffers from browser mic.
    Streams back synthesized PCM audio chunks, transcripts, and state events.
    """
    await websocket.accept()
    logger.info(f"WebSocket client connected from {websocket.client}")

    # Session configuration
    voice = os.getenv("TTS_VOICE", "en-US-JennyNeural")
    llm_model = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")

    llm_service = GroqLLMService(model=llm_model)
    tts_service = EdgeTTSService(voice=voice, output_sample_rate=16000)

    # Session state
    is_assistant_generating = False
    is_assistant_speaking = False
    current_response_task: Optional[asyncio.Task] = None

    # VAD local instance for this session
    session_vad = SileroVADService(
        sample_rate=16000,
        threshold=float(os.getenv("VAD_CONFIDENCE_THRESHOLD", "0.5")),
        min_silence_duration_ms=int(os.getenv("VAD_SILENCE_DURATION_MS", "350")),
        chunk_size=512,
    )

    async def send_json(data: dict):
        try:
            await websocket.send_text(json.dumps(data))
        except Exception:
            pass

    async def handle_turn(speech_audio: bytes, speech_end_time: float):
        nonlocal is_assistant_generating, is_assistant_speaking
        try:
            is_assistant_generating = True
            await send_json({"type": "state", "state": "thinking"})

            # 1. Transcribe speech
            stt_start = time.perf_counter()
            transcript = await stt_service.transcribe(speech_audio)
            stt_latency = (time.perf_counter() - stt_start) * 1000

            if not transcript or not transcript.strip():
                await send_json({"type": "state", "state": "idle"})
                return

            await send_json({
                "type": "transcript",
                "role": "user",
                "text": transcript,
                "stt_ms": round(stt_latency, 1),
            })

            # 2. Stream tokens from Groq
            llm_start = time.perf_counter()
            token_gen = llm_service.stream_response(transcript)
            sentence_gen = sentence_stream_buffer(token_gen)

            first_sentence = True
            first_sentence_logged = False
            sentence_time = 0.0

            async for sentence in sentence_gen:
                if not sentence.strip():
                    continue

                if first_sentence:
                    sentence_time = (time.perf_counter() - llm_start) * 1000
                    first_sentence = False

                # Inform client assistant is speaking and stream transcript
                is_assistant_speaking = True
                await send_json({
                    "type": "transcript_chunk",
                    "role": "assistant",
                    "text": sentence,
                })
                await send_json({"type": "state", "state": "speaking"})

                # 3. Stream synthetic PCM audio chunks to browser
                tts_start = time.perf_counter()
                async for pcm_chunk in tts_service.synthesize_sentence_stream(sentence):
                    if not first_sentence_logged:
                        total_latency = (time.perf_counter() - speech_end_time) * 1000
                        tts_ttfb = (time.perf_counter() - tts_start) * 1000
                        await send_json({
                            "type": "latency_stats",
                            "total_ms": round(total_latency, 1),
                            "stt_ms": round(stt_latency, 1),
                            "llm_ms": round(sentence_time, 1),
                            "tts_ms": round(tts_ttfb, 1),
                        })
                        first_sentence_logged = True

                    # Send binary audio chunk to browser
                    await websocket.send_bytes(pcm_chunk)

            await send_json({"type": "turn_complete"})

        except asyncio.CancelledError:
            logger.info("Web turn cancelled (Barge-in / User Interruption).")
            await send_json({"type": "interrupted"})
        except Exception as e:
            logger.error(f"Error handling web turn: {e}")
            await send_json({"type": "error", "message": str(e)})
        finally:
            is_assistant_generating = False

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                try:
                    payload = json.loads(message["text"])
                    if payload.get("type") == "config":
                        new_voice = payload.get("voice")
                        if new_voice:
                            voice = new_voice
                            tts_service = EdgeTTSService(voice=voice, output_sample_rate=16000)
                            logger.info(f"Switched voice to: {voice}")
                            await send_json({"type": "config_updated", "voice": voice})
                    elif payload.get("type") == "interrupt":
                        # Client-side user interrupt
                        if current_response_task and not current_response_task.done():
                            current_response_task.cancel()
                        is_assistant_speaking = False
                        is_assistant_generating = False
                        session_vad.reset()
                        await send_json({"type": "state", "state": "idle"})
                    elif payload.get("type") == "playback_ended":
                        is_assistant_speaking = False
                        if not is_assistant_generating:
                            await send_json({"type": "state", "state": "idle"})
                except json.JSONDecodeError:
                    pass

            elif "bytes" in message:
                pcm_chunk = message["bytes"]

                # Process microphone chunk through Silero VAD
                vad_result = session_vad.process_frame(pcm_chunk)

                if vad_result.state == "speech_start":
                    # If assistant is speaking or generating, barge-in interrupt
                    if is_assistant_speaking or is_assistant_generating:
                        if current_response_task and not current_response_task.done():
                            current_response_task.cancel()
                        is_assistant_speaking = False
                        is_assistant_generating = False
                        await send_json({"type": "interrupted"})

                    await send_json({"type": "state", "state": "listening"})

                elif vad_result.state == "speech_end":
                    speech_end_time = time.perf_counter()
                    speech_audio = session_vad.get_accumulated_speech_audio()

                    if speech_audio and len(speech_audio) >= 3200:  # At least 100ms
                        current_response_task = asyncio.create_task(
                            handle_turn(speech_audio, speech_end_time)
                        )
                    else:
                        await send_json({"type": "state", "state": "idle"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if current_response_task and not current_response_task.done():
            current_response_task.cancel()


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting EchoStream Web Server on http://localhost:8000...")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
