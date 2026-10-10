"""Groq API Key Verification & Latency Benchmark Script."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

from services.llm import GroqLLMService
from utils.stream_buffers import sentence_stream_buffer


async def test_groq_streaming():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        logger.error("GROQ_API_KEY is missing or empty in .env!")
        sys.exit(1)

    masked_key = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    logger.info(f"Testing Groq authentication with key: {masked_key}")

    llm = GroqLLMService()

    test_prompt = "Say a one-sentence warm greeting to confirm you are online and ready."
    print(f"\nUser Prompt: \"{test_prompt}\"")
    print("Streaming Tokens: ", end="", flush=True)

    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    full_response = []

    try:
        async for token in llm.stream_response(test_prompt):
            if first_token_time is None:
                first_token_time = (time.perf_counter() - start_time) * 1000
            print(token, end="", flush=True)
            full_response.append(token)
            token_count += 1
        print("\n")
    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        sys.exit(1)

    total_time_ms = (time.perf_counter() - start_time) * 1000

    print("================ GROQ BENCHMARK RESULTS ================")
    print(f"Time To First Token (TTFT) : {first_token_time:.1f} ms")
    print(f"Total Generation Time       : {total_time_ms:.1f} ms")
    print(f"Token Count                 : {token_count} tokens")
    if total_time_ms > 0:
        tokens_per_sec = (token_count / (total_time_ms / 1000.0))
        print(f"Throughput                  : {tokens_per_sec:.1f} tokens/sec")
    print("Status                      : [SUCCESS] Groq API Key is valid and active!")
    print("=========================================================\n")


if __name__ == "__main__":
    asyncio.run(test_groq_streaming())
