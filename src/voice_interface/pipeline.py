"""VoiceCallPipeline: wires STT -> ProtocolAgent.respond() -> TTS for one call. This is the seam
CLAUDE.md architecture principle 4 describes — ProtocolAgent is used unchanged (via respond(),
called off-thread since it's a blocking anthropic SDK call and this pipeline is async); every
voice-specific concern (latency, barge-in, audio buffering) lives here instead.

Turn processing runs as a background task rather than inline in handle_audio_frame() so inbound
audio keeps flowing while the agent is "thinking" and TTS is playing — that's what makes barge-in
possible: a new final transcript arriving mid-turn immediately emits a "clear" event (Twilio's own
barge-in signal) and stops that turn's audio from being spoken.

Important limitation: respond() mutates the shared `session` dict in place and is a plain blocking
call with no cooperative cancellation, so a barged-in turn's agent.respond() call is left to run to
completion in the background rather than aborted — an interrupted respond() call whose thread kept
mutating `session["messages"]` concurrently with the next turn's respond() call produced a genuine
data race here during testing (session["messages"] getting corrupted, next turn's reply visibly
addressing both utterances at once). Fixed by serializing respond() calls through _turn_lock, so
only one is ever in flight against `session` at a time — a superseded turn still finishes and its
Q&A still lands in session["messages"], it just never reaches the caller's ears. A production
system would want a genuinely cancellable client call here instead."""

import asyncio
from dataclasses import dataclass
from typing import Protocol

from .latency import LatencyTracker
from .stt import SpeechToText
from .tts import AudioChunk, TextToSpeech


class RespondingAgent(Protocol):
    def respond(self, session: dict, user_utterance: str) -> str: ...


@dataclass
class OutboundEvent:
    """What the pipeline hands the transport layer (media_stream_server.py or a script)."""

    kind: str  # "audio" | "clear" | "turn_done"
    chunk: AudioChunk | None = None


class VoiceCallPipeline:
    def __init__(self, agent: RespondingAgent, stt: SpeechToText, tts: TextToSpeech):
        self.agent = agent
        self.stt = stt
        self.tts = tts
        self.session: dict = {"specialty": None, "messages": []}
        self.latency = LatencyTracker()
        self.outbound: asyncio.Queue[OutboundEvent] = asyncio.Queue()
        self._turn_task: asyncio.Task | None = None
        self._turn_lock = asyncio.Lock()
        self._active_turn_id = 0

    async def handle_audio_frame(self, audio_chunk: bytes) -> None:
        """Feed one inbound audio frame through STT. Non-blocking: a final transcript starts a
        background turn task and returns immediately, so the transport's read loop keeps up with
        incoming audio instead of stalling behind the agent call and TTS playback."""
        event = await self.stt.feed_audio(audio_chunk)
        if event is None or not event.is_final:
            return

        self._active_turn_id += 1
        turn_id = self._active_turn_id
        if self._turn_task is not None and not self._turn_task.done():
            await self.outbound.put(OutboundEvent(kind="clear"))

        self.stt.reset()
        self._turn_task = asyncio.create_task(self._run_turn(event.text, turn_id))

    async def _run_turn(self, utterance: str, turn_id: int) -> None:
        turn = self.latency.start_turn()
        self.latency.mark(turn, "stt_final")

        async with self._turn_lock:  # only one respond() call ever touches `session` at a time
            reply = await asyncio.to_thread(self.agent.respond, self.session, utterance)

        if turn_id != self._active_turn_id:
            return  # superseded by a barge-in while this turn's respond() call was in flight
        self.latency.mark(turn, "agent_response_ready")

        first_chunk = True
        async for chunk in self.tts.synthesize(reply):
            if turn_id != self._active_turn_id:
                return
            if first_chunk:
                self.latency.mark(turn, "tts_first_chunk")
                first_chunk = False
            await self.outbound.put(OutboundEvent(kind="audio", chunk=chunk))

        await self.outbound.put(OutboundEvent(kind="turn_done"))
