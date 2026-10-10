"""Unit and Latency Tests for EchoStream Stream Buffers."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.stream_buffers import PunctuationBuffer, sentence_stream_buffer


async def test_punctuation_buffer_sentence_splitting():
    """Verify that sentences are yielded the exact millisecond punctuation arrives."""
    buffer = PunctuationBuffer()

    tokens = ["Hello", " world", "!", " How", " are", " you", " doing", "?", " I", " am", " fine", "."]
    sentences_received = []

    for token in tokens:
        sentences = buffer.append(token)
        for s in sentences:
            sentences_received.append(s)

    # Remaining
    final = buffer.flush()
    if final:
        sentences_received.append(final)

    assert sentences_received == [
        "Hello world!",
        "How are you doing?",
        "I am fine.",
    ]


async def test_abbreviations_and_decimals():
    """Verify that abbreviations (Dr., e.g.) and decimal numbers (3.14) do not trigger premature splits."""
    buffer = PunctuationBuffer()

    # Feeding text with abbreviations and decimal
    tokens = ["Dr.", " Smith", " calculated", " the", " value", " as", " 3.14", " today", "."]
    sentences_received = []

    for token in tokens:
        sentences = buffer.append(token)
        sentences_received.extend(sentences)

    final = buffer.flush()
    if final:
        sentences_received.append(final)

    # Should not split at Dr. or 3.14
    assert len(sentences_received) == 1
    assert "Dr. Smith calculated the value as 3.14 today." in sentences_received[0]


async def test_sentence_stream_buffer_generator():
    """Test the async generator wrapper."""
    async def mock_llm_stream():
        chunks = ["EchoStream ", "is ", "running. ", "End-to-end ", "latency ", "is ", "low!"]
        for c in chunks:
            await asyncio.sleep(0.005)
            yield c

    results = []
    async for sentence in sentence_stream_buffer(mock_llm_stream()):
        results.append(sentence)

    assert len(results) == 2
    assert results[0] == "EchoStream is running."
    assert results[1] == "End-to-end latency is low!"


if __name__ == "__main__":
    asyncio.run(test_punctuation_buffer_sentence_splitting())
    asyncio.run(test_abbreviations_and_decimals())
    asyncio.run(test_sentence_stream_buffer_generator())
    print("[SUCCESS] All stream buffer tests passed successfully!")

