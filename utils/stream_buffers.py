"""Asynchronous Stream Buffers for Sentence-Level Chunking.

Mandatory Architecture:
Sentence-Level Streaming: Dispatches complete sentences immediately to the TTS engine
the exact millisecond a sentence terminator (. ? !) is yielded by the LLM, without
waiting for the entire LLM generation to complete.
"""

from __future__ import annotations

import re
from typing import AsyncGenerator, List, Optional


class PunctuationBuffer:
    """Asynchronous punctuation buffer that slices raw LLM streaming tokens

    into clean, natural sentence chunks suitable for immediate TTS synthesis.
    """

    # Common abbreviations to prevent premature splitting
    ABBREVIATIONS = {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "sr.",
        "jr.",
        "vs.",
        "etc.",
        "e.g.",
        "i.e.",
        "inc.",
        "ltd.",
        "co.",
        "approx.",
        "dept.",
        "fig.",
    }

    # Matches sentence-terminating punctuation followed by whitespace or end of string
    SENTENCE_SPLIT_REGEX = re.compile(r"([.?!]+(?:\s+|\n+|$))")

    def __init__(self, min_chunk_words: int = 3, max_clause_words: int = 25):
        """Args:

        min_chunk_words: Minimum words before splitting on weaker punctuation
        like commas max_clause_words: Maximum words without sentence
        punctuation before forcing a clause break
        """
        self._buffer: str = ""
        self._min_chunk_words = min_chunk_words
        self._max_clause_words = max_clause_words

    def append(self, token: str) -> List[str]:
        """Appends a new token to the internal buffer and extracts any complete

        sentences.

        Returns a list of complete sentence strings ready for TTS.
        """
        self._buffer += token
        return self._extract_sentences()

    def _is_abbreviation(self, text_segment: str) -> bool:
        """Checks if the end of text_segment is a recognized abbreviation."""
        tokens = text_segment.strip().split()
        if not tokens:
            return False
        last_word = tokens[-1].lower()
        return last_word in self.ABBREVIATIONS

    def _extract_sentences(self) -> List[str]:
        ready_sentences: List[str] = []

        while True:
            # Check for punctuation marks
            match = self.SENTENCE_SPLIT_REGEX.search(self._buffer)
            if not match:
                # Fallback: if buffer is getting too long (run-on sentence), split on comma or semicolon
                words = self._buffer.split()
                if len(words) >= self._max_clause_words:
                    clause_match = re.search(r"([,;:\n]+(?:\s+|$))", self._buffer)
                    if clause_match:
                        end_idx = clause_match.end()
                        sentence = self._buffer[:end_idx].strip()
                        self._buffer = self._buffer[end_idx:]
                        if sentence:
                            ready_sentences.append(sentence)
                            continue
                break

            end_idx = match.end()
            candidate = self._buffer[:end_idx]

            # Check if this split is an abbreviation or decimal number (e.g. 3.14)
            # Avoid splitting on decimals: digit followed by dot followed by digit
            if re.search(r"\d+\.\d+$", candidate.strip()):
                # Likely a decimal number; wait for more text unless buffer ends
                break

            if self._is_abbreviation(candidate):
                # Look past this abbreviation in the next iteration or wait
                break

            sentence = candidate.strip()
            self._buffer = self._buffer[end_idx:]

            if sentence:
                ready_sentences.append(sentence)

        return ready_sentences

    def flush(self) -> Optional[str]:
        """Flushes whatever text remains in the buffer at the end of the stream."""
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None

    def clear(self) -> None:
        """Immediately discards all buffered text (used during Barge-In / Kill Switch)."""
        self._buffer = ""


async def sentence_stream_buffer(
    token_generator: AsyncGenerator[str, None],
    min_chunk_words: int = 3,
) -> AsyncGenerator[str, None]:
    """Asynchronous generator that consumes tokens from an LLM stream and yields

    complete sentences the millisecond they are terminated.
    """
    buffer = PunctuationBuffer(min_chunk_words=min_chunk_words)

    async for token in token_generator:
        sentences = buffer.append(token)
        for sentence in sentences:
            if sentence:
                yield sentence

    # Flush remaining text after token generation finishes
    final_sentence = buffer.flush()
    if final_sentence:
        yield final_sentence

