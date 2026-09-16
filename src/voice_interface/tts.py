"""Text-to-speech interface. Real streaming TTS needs a vendor account (ElevenLabs, Deepgram,
etc.) this repo doesn't have, so MockTTS stands in — it produces placeholder audio sized and timed
like real 8kHz mulaw (Twilio's wire format) instead of doing real synthesis, so downstream latency
numbers and payload sizes are meaningful rather than instant/empty."""

import asyncio
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORDS_PER_MINUTE = 150
_SYNTHESIS_LATENCY_S = 0.12  # simulated per-chunk time-to-first-byte
_MULAW_SAMPLE_RATE_HZ = 8000  # Twilio Media Streams' wire format: 8kHz mulaw, 1 byte/sample


@dataclass
class AudioChunk:
    text: str  # the sentence this chunk represents — stand-in for real audio bytes
    audio: bytes  # placeholder payload, sized to match real Twilio audio duration
    duration_s: float


class TextToSpeech(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        """Yield AudioChunks for the given reply text, in playback order."""


class MockTTS(TextToSpeech):
    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        sentences = [s for s in _SENTENCE_SPLIT.split(text.strip()) if s]

        for sentence in sentences:
            await asyncio.sleep(_SYNTHESIS_LATENCY_S)
            word_count = max(1, len(sentence.split()))
            duration_s = word_count / _WORDS_PER_MINUTE * 60
            audio = b"\x00" * int(duration_s * _MULAW_SAMPLE_RATE_HZ)
            yield AudioChunk(text=sentence, audio=audio, duration_s=duration_s)
