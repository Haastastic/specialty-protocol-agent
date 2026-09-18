"""Offline tests for the voice interface — no network calls, same convention as
tests/test_retrieval.py."""

import asyncio
import time

import pytest

from src.voice_interface.latency import STAGE_BUDGETS_MS, TOTAL_BUDGET_MS, LatencyTracker
from src.voice_interface.pipeline import VoiceCallPipeline
from src.voice_interface.stt import MockSTT
from src.voice_interface.tts import MockTTS
from src.voice_interface.twiml import stream_twiml


class StubAgent:
    """Records calls instead of hitting the real Anthropic API."""

    def __init__(self, reply: str = "Sure, here's your answer.", delay_s: float = 0.0):
        self.calls = []
        self.reply = reply
        self.delay_s = delay_s

    def respond(self, session: dict, user_utterance: str) -> str:
        self.calls.append((session["specialty"], user_utterance))
        if self.delay_s:
            time.sleep(self.delay_s)
        session["messages"].append({"role": "user", "content": user_utterance})
        return self.reply


def test_stream_twiml_wraps_url():
    xml = stream_twiml("wss://example.com/media-stream")
    assert '<Connect><Stream url="wss://example.com/media-stream" /></Connect>' in xml
    assert xml.startswith("<?xml")


def test_mock_stt_partial_then_final():
    stt = MockSTT(["hello there friend"], frames_per_utterance=3, recognition_delay_s=0.0)

    async def run():
        return [await stt.feed_audio(b"") for _ in range(3)]

    events = asyncio.run(run())
    assert [e.is_final for e in events] == [False, False, True]
    assert events[2].text == "hello there friend"


def test_mock_stt_exhausts_after_reset():
    stt = MockSTT(["one"], frames_per_utterance=1, recognition_delay_s=0.0)
    asyncio.run(stt.feed_audio(b""))
    assert not stt.exhausted
    stt.reset()
    assert stt.exhausted
    assert asyncio.run(stt.feed_audio(b"")) is None


def test_mock_tts_chunks_by_sentence():
    tts = MockTTS()

    async def run():
        return [chunk async for chunk in tts.synthesize("Hi there. How can I help you today?")]

    chunks = asyncio.run(run())
    assert [c.text for c in chunks] == ["Hi there.", "How can I help you today?"]
    assert all(c.duration_s > 0 and len(c.audio) > 0 for c in chunks)


def test_latency_tracker_stage_math_and_budgets():
    tracker = LatencyTracker()
    turn = tracker.start_turn()
    turn.turn_start = 0.0
    turn.stt_final = 0.1
    turn.agent_response_ready = 0.9
    turn.tts_first_chunk = 1.0

    stages = turn.stage_ms()
    assert stages["stt_final"] == pytest.approx(100)
    assert stages["agent_response"] == pytest.approx(800)
    assert stages["tts_first_chunk"] == pytest.approx(100)
    assert turn.total_ms() == pytest.approx(1000)
    assert stages["agent_response"] < STAGE_BUDGETS_MS["agent_response"]
    assert turn.total_ms() < TOTAL_BUDGET_MS

    tracker.render()  # must not raise


def test_pipeline_runs_a_turn_and_updates_session():
    agent = StubAgent(reply="Sounds good, see you then.")
    stt = MockSTT(["book me an appointment"], frames_per_utterance=1, recognition_delay_s=0.0)
    pipeline = VoiceCallPipeline(agent=agent, stt=stt, tts=MockTTS())

    async def run():
        await pipeline.handle_audio_frame(b"")
        events = []
        while True:
            event = await pipeline.outbound.get()
            events.append(event)
            if event.kind == "turn_done":
                break
        return events

    events = asyncio.run(run())

    assert agent.calls == [(None, "book me an appointment")]
    assert [e.kind for e in events] == ["audio", "turn_done"]
    assert events[0].chunk.text == "Sounds good, see you then."
    assert pipeline.latency.turns[0].total_ms() is not None


class _FailingTTS(MockTTS):
    async def synthesize(self, text: str):
        raise RuntimeError("402 Payment Required")
        yield  # unreachable; makes this an async generator like the real TTS


def test_pipeline_tts_failure_is_logged_and_ends_the_turn(caplog):
    stt = MockSTT(["book me an appointment"], frames_per_utterance=1, recognition_delay_s=0.0)
    pipeline = VoiceCallPipeline(agent=StubAgent(), stt=stt, tts=_FailingTTS())

    async def run():
        await pipeline.handle_audio_frame(b"")
        return await asyncio.wait_for(pipeline.outbound.get(), timeout=2)

    event = asyncio.run(run())

    assert event.kind == "turn_done"
    assert "Turn 1 failed" in caplog.text
    assert "402 Payment Required" in caplog.text


def test_pipeline_agent_failure_is_logged_and_ends_the_turn(caplog):
    class BoomAgent:
        def respond(self, session, user_utterance):
            raise RuntimeError("anthropic down")

    stt = MockSTT(["hi"], frames_per_utterance=1, recognition_delay_s=0.0)
    pipeline = VoiceCallPipeline(agent=BoomAgent(), stt=stt, tts=MockTTS())

    async def run():
        await pipeline.handle_audio_frame(b"")
        return await asyncio.wait_for(pipeline.outbound.get(), timeout=2)

    assert asyncio.run(run()).kind == "turn_done"
    assert "anthropic down" in caplog.text


class _SequencedStubAgent:
    """Appends user/assistant messages the same way ProtocolAgent.respond() does (user turn
    before the "network" delay, assistant turn after) so a serialization race shows up as
    interleaved session["messages"] instead of the clean alternating order asserted below."""

    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s

    def respond(self, session: dict, user_utterance: str) -> str:
        session["messages"].append({"role": "user", "content": user_utterance})
        if self.delay_s:
            time.sleep(self.delay_s)
        reply = f"reply to: {user_utterance}"
        session["messages"].append({"role": "assistant", "content": reply})
        return reply


def test_pipeline_barge_in_discards_superseded_reply_and_serializes_session():
    agent = _SequencedStubAgent(delay_s=0.3)
    stt = MockSTT(["first line", "second line"], frames_per_utterance=1, recognition_delay_s=0.0)
    pipeline = VoiceCallPipeline(agent=agent, stt=stt, tts=MockTTS())

    async def run():
        await pipeline.handle_audio_frame(b"")  # starts the slow first turn in the background
        await asyncio.sleep(0.05)  # well before the 0.3s stub agent call returns
        await pipeline.handle_audio_frame(b"")  # barges in before the first turn finishes

        events = []
        while len(events) < 3:  # clear, audio (second turn only), turn_done
            events.append(await pipeline.outbound.get())
        return events

    events = asyncio.run(run())

    assert [e.kind for e in events] == ["clear", "audio", "turn_done"]
    # the barged-in first turn's reply is never spoken — only the second turn's is
    assert events[1].chunk.text == "reply to: second line"
    # respond() calls are fully serialized: no interleaving even though the first turn's
    # underlying call was still running (unstoppable) when the second turn started
    assert pipeline.session["messages"] == [
        {"role": "user", "content": "first line"},
        {"role": "assistant", "content": "reply to: first line"},
        {"role": "user", "content": "second line"},
        {"role": "assistant", "content": "reply to: second line"},
    ]
