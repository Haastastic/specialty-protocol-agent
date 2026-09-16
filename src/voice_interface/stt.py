"""Speech-to-text interface. Real streaming ASR needs a vendor account (Deepgram, AssemblyAI,
etc.) this repo doesn't have, so MockSTT stands in — it doesn't decode audio, it's driven by a
pre-scripted list of caller lines (same idea as ProtocolAgent.run_conversation's patient_turns)
so the rest of the pipeline still exercises a real partial->final recognition cadence. Real audio
bytes still flow through media_stream_server.py unchanged; only the transcription is scripted."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass


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
