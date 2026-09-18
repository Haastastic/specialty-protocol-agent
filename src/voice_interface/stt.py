"""Speech-to-text interface. Real streaming ASR needs a vendor account (Deepgram, AssemblyAI,
etc.) this repo doesn't have, so MockSTT stands in — it doesn't decode audio, it's driven by a
pre-scripted list of caller lines (same idea as ProtocolAgent.run_conversation's patient_turns)
so the rest of the pipeline still exercises a real partial->final recognition cadence. Real audio
bytes still flow through media_stream_server.py unchanged; only the transcription is scripted."""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import websockets


@dataclass
class TranscriptEvent:
    text: str
    is_final: bool


class SpeechToText(ABC):
    """One instance per call. feed_audio() is called with each inbound audio frame as it arrives
    off the wire, in order."""

    @abstractmethod
    async def feed_audio(self, audio_chunk: bytes) -> TranscriptEvent | None:
        """Consume one inbound audio frame. Returns a TranscriptEvent for a partial or final
        recognition update, or None if this frame didn't produce one."""

    @abstractmethod
    def reset(self) -> None:
        """Called once a final transcript has been consumed, before the next utterance starts."""


class MockSTT(SpeechToText):
    """frames_per_utterance controls how many feed_audio() calls one scripted line "costs" before
    going final — the caller decides how audio frames map to that (media_stream_server.py uses
    real Twilio frame timing; simulate_call.py just calls feed_audio() that many times directly).
    """

    def __init__(
        self,
        scripted_lines: list[str],
        frames_per_utterance: int = 3,
        recognition_delay_s: float = 0.05,
    ):
        self.scripted_lines = list(scripted_lines)
        self.frames_per_utterance = max(1, frames_per_utterance)
        self.recognition_delay_s = recognition_delay_s
        self._cursor = 0
        self._frame_count = 0

    async def feed_audio(self, audio_chunk: bytes) -> TranscriptEvent | None:
        if self.exhausted:
            return None

        self._frame_count += 1
        line = self.scripted_lines[self._cursor]
        words = line.split()

        if self._frame_count < self.frames_per_utterance:
            portion = max(1, round(len(words) * self._frame_count / self.frames_per_utterance))
            return TranscriptEvent(text=" ".join(words[:portion]), is_final=False)

        await asyncio.sleep(self.recognition_delay_s)
        return TranscriptEvent(text=line, is_final=True)

    def reset(self) -> None:
        self._cursor += 1
        self._frame_count = 0

    @property
    def exhausted(self) -> bool:
        return self._cursor >= len(self.scripted_lines)


_DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"


class DeepgramSTT(SpeechToText):
    """Real streaming STT via Deepgram's WebSocket API
    (https://developers.deepgram.com/reference/speech-to-text-api/listen-streaming), talked to
    directly over its documented wire protocol (query-param config + JSON transcript messages)
    rather than through deepgram-sdk — that SDK's client surface has changed across major versions
    and this repo has no live Deepgram account to test against, so the wire contract is the more
    stable thing to depend on.

    feed_audio() is fire-and-forget from the caller's perspective (send this frame, immediately
    check for any transcript that has already arrived) since Deepgram delivers transcripts
    asynchronously and not necessarily one-per-frame — a queue absorbs the mismatch between "audio
    frame in" and "transcript event out" that MockSTT's synchronous scripted-lines model doesn't
    have to deal with.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "nova-3",
        language: str = "en-US",
    ):
        self.api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPGRAM_API_KEY not set.")
        self.model = model
        self.language = language
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._recv_task: asyncio.Task | None = None
        self._events: asyncio.Queue[TranscriptEvent] = asyncio.Queue()

    async def _ensure_connected(self) -> None:
        if self._ws is not None:
            return
        params = (
            "encoding=mulaw&sample_rate=8000&channels=1"
            f"&model={self.model}&language={self.language}"
            "&interim_results=true&punctuate=true"
        )
        # extra_headers is the websockets<14 API this repo pins against (see requirements.txt);
        # a later websockets upgrade renamed this to additional_headers.
        self._ws = await websockets.connect(
            f"{_DEEPGRAM_WS_URL}?{params}",
            extra_headers={"Authorization": f"Token {self.api_key}"},
        )
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        async for raw in self._ws:
            message = json.loads(raw)
            alternatives = message.get("channel", {}).get("alternatives")
            if not alternatives:
                continue
            transcript = alternatives[0].get("transcript", "")
            if not transcript:
                continue
            await self._events.put(
                TranscriptEvent(text=transcript, is_final=bool(message.get("is_final")))
            )

    async def feed_audio(self, audio_chunk: bytes) -> TranscriptEvent | None:
        await self._ensure_connected()
        assert self._ws is not None
        await self._ws.send(audio_chunk)
        try:
            return self._events.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def reset(self) -> None:
        pass  # Deepgram's connection is continuous across utterances; nothing to clear per-turn

    async def close(self) -> None:
        """Not part of the SpeechToText ABC (MockSTT has no connection to tear down) — callers
        that hold a real connection, like media_stream_server.py, should call this via
        getattr(stt, "close", None) when the call ends."""
        if self._recv_task is not None:
            self._recv_task.cancel()
        if self._ws is not None:
            await self._ws.close()
