"""Groq LLM Service for Ultra-Low Latency Conversational Voice.

Streams tokens using Groq's high-throughput LPU inference running Llama-3.
Includes voice-optimized system prompting and async streaming token generators.
"""

from __future__ import annotations

import asyncio
import os
import time
from loguru import logger
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()


DEFAULT_VOICE_SYSTEM_PROMPT = """You are EchoStream, an ultra-fast, conversational AI voice assistant.
Follow these rules strictly for natural voice interactions:
1. Be concise, direct, and conversational. Respond in 1 to 2 short sentences unless the user explicitly requests more detail.
2. Never use markdown formatting (no asterisks, bold, bullet points, headers, or emojis) because your output will be read aloud by a Text-to-Speech engine.
3. Speak naturally, as if having a fluid real-time phone conversation.
4. Keep punctuation clean (. ? !) to enable rapid sentence streaming.
"""


class GroqLLMService:
    """Manages conversational dialogue and token streaming using Groq's API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: str = DEFAULT_VOICE_SYSTEM_PROMPT,
        temperature: float = 0.6,
        max_tokens: int = 250,
        max_history_turns: int = 6,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.warning(
                "GROQ_API_KEY is not set. Please provide it in your .env or constructor."
            )

        self.model = model or os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_history_turns = max_history_turns

        self.client = AsyncGroq(api_key=self.api_key) if self.api_key else None
        self.conversation_history: List[Dict[str, str]] = []

    def reset_conversation(self) -> None:
        """Clears conversational history."""
        self.conversation_history.clear()

    def add_user_message(self, text: str) -> None:
        """Appends user message to history."""
        self.conversation_history.append({"role": "user", "content": text})
        self._trim_history()

    def add_assistant_message(self, text: str) -> None:
        """Appends assistant response to history."""
        self.conversation_history.append({"role": "assistant", "content": text})
        self._trim_history()

    def _trim_history(self) -> None:
        """Keeps history within max_history_turns (paired user/assistant turns)."""
        max_messages = self.max_history_turns * 2
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]

    def _build_messages(self) -> List[Dict[str, str]]:
        """Constructs prompt payload with system message and recent dialogue."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        return messages

    async def stream_response(
        self,
        user_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Asynchronously streams generated tokens from Groq Llama-3.

        Measures Time-To-First-Token (TTFT) and logs streaming throughput.
        """
        if not self.client:
            self.client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

        if user_prompt:
            self.add_user_message(user_prompt)

        messages = self._build_messages()

        start_time = time.perf_counter()
        first_token_received = False
        full_response_parts: List[str] = []

        try:
            try:
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                )
            except Exception as model_err:
                # If model was deprecated/decommissioned on Groq, automatically fall back to active models
                fallback_models = ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
                stream = None
                for fb in fallback_models:
                    if fb != self.model:
                        logger.warning(
                            f"Model '{self.model}' unavailable ({model_err}). Falling back to '{fb}'..."
                        )
                        self.model = fb
                        stream = await self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            stream=True,
                        )
                        break
                if stream is None:
                    raise model_err

            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    if not first_token_received:
                        ttft_ms = (time.perf_counter() - start_time) * 1000
                        logger.info(f"LLM TTFT (Time To First Token): {ttft_ms:.1f}ms")
                        first_token_received = True

                    full_response_parts.append(content)
                    yield content

            # Once fully streamed, record assistant message in history
            complete_response = "".join(full_response_parts).strip()
            if complete_response:
                self.add_assistant_message(complete_response)

        except asyncio.CancelledError:
            logger.info("LLM generation stream cancelled (Barge-in).")
            # If interrupted midway, still record what was uttered so context isn't lost
            interrupted_text = "".join(full_response_parts).strip()
            if interrupted_text:
                self.add_assistant_message(interrupted_text + "...")
            raise
        except Exception as e:
            logger.error(f"Groq LLM streaming error: {e}")
            raise

