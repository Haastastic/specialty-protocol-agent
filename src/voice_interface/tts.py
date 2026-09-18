"""Text-to-speech interface. Real streaming TTS needs a vendor account (ElevenLabs, Deepgram,
etc.) this repo doesn't have, so MockTTS stands in — it produces placeholder audio sized and timed
like real 8kHz mulaw (Twilio's wire format) instead of doing real synthesis, so downstream latency
numbers and payload sizes are meaningful rather than instant/empty."""

import asyncio
import os
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

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


_ELEVENLABS_STREAM_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"


class ElevenLabsTTS(TextToSpeech):
    """Real streaming TTS via ElevenLabs' REST streaming endpoint
    (https://elevenlabs.io/docs/api-reference/text-to-speech/stream), requested directly in
    Twilio's wire format (output_format=ulaw_8000) so audio flows straight into
    media_stream_server.py's outbound relay with no transcoding step.

    One HTTP request per sentence, same granularity as MockTTS — each request is buffered in full
    before yielding (ElevenLabs streams the HTTP response in chunks, but those chunk boundaries
    don't line up with anything meaningful downstream) so a mid-reply barge-in still only wastes at
    most one sentence of synthesis instead of the whole turn.
    """

    def __init__(
        self,
        voice_id: str,
        api_key: str | None = None,
        model_id: str = "eleven_turbo_v2_5",
    ):
        self.voice_id = voice_id
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not set.")
        self.model_id = model_id

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        sentences = [s for s in _SENTENCE_SPLIT.split(text.strip()) if s]
        url = _ELEVENLABS_STREAM_URL.format(voice_id=self.voice_id)

        async with httpx.AsyncClient() as client:
            for sentence in sentences:
                audio = bytearray()
                async with client.stream(
                    "POST",
                    url,
                    params={"output_format": "ulaw_8000"},
                    headers={"xi-api-key": self.api_key},
                    json={"text": sentence, "model_id": self.model_id},
                    timeout=30.0,
                ) as response:
                    response.raise_for_status()
                    async for byte_chunk in response.aiter_bytes():
                        audio.extend(byte_chunk)

                duration_s = len(audio) / _MULAW_SAMPLE_RATE_HZ
                yield AudioChunk(text=sentence, audio=bytes(audio), duration_s=duration_s)
