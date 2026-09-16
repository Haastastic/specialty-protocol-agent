"""FastAPI app implementing Twilio's Media Streams wire protocol and routing audio to/from a
VoiceCallPipeline. This is the one file in this package that speaks Twilio's actual message
schema (https://www.twilio.com/docs/voice/media-streams/websocket-messages) — pipeline.py, stt.py
and tts.py are all transport-agnostic.

_stt_factory() intentionally raises: MockSTT is driven by a pre-scripted list of caller lines,
which only makes sense for the local simulate_call.py harness — there's no way to script what a
real caller on a live Twilio number will say. Point this at a real streaming-STT adapter
(implementing SpeechToText from stt.py) before running this server against a live call. MockTTS
is left as the default since a missing TTS just means the caller hears silence rather than the
server being unable to function at all; swap in a real TextToSpeech adapter the same way.
"""

import asyncio
import base64
import json

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from src.protocol_agent.agent import ProtocolAgent

from .pipeline import RespondingAgent, VoiceCallPipeline
from .stt import SpeechToText
from .tts import MockTTS, TextToSpeech
from .twiml import stream_twiml

app = FastAPI()

_agent: ProtocolAgent | None = None


def _agent_singleton() -> RespondingAgent:
    global _agent
    if _agent is None:
        _agent = ProtocolAgent()
    return _agent


def _stt_factory() -> SpeechToText:
    raise NotImplementedError(
        "No real speech-to-text adapter is wired up. See README 'Voice interface' section: "
        "implement SpeechToText (stt.py) for your vendor and replace this factory before running "
        "against a live Twilio number."
    )


def _tts_factory() -> TextToSpeech:
    return MockTTS()


@app.post("/twiml")
async def twiml(request: Request) -> Response:
    ws_url = f"wss://{request.url.netloc}/media-stream"
    return Response(content=stream_twiml(ws_url), media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    stream_sid: str | None = None
    pipeline: VoiceCallPipeline | None = None
    outbound_task: asyncio.Task | None = None

    try:
        while True:
            raw = await websocket.receive_text()
            message = json.loads(raw)
            event = message.get("event")

            if event == "connected":
                continue

            if event == "start":
                stream_sid = message["streamSid"]
                pipeline = VoiceCallPipeline(
                    agent=_agent_singleton(), stt=_stt_factory(), tts=_tts_factory()
                )
                outbound_task = asyncio.create_task(
                    _drain_outbound(websocket, pipeline, stream_sid)
                )
                continue

            if event == "media" and pipeline is not None:
                payload = base64.b64decode(message["media"]["payload"])
                await pipeline.handle_audio_frame(payload)
                continue

            if event == "stop":
                break

    except WebSocketDisconnect:
        pass
    finally:
        if outbound_task is not None:
            outbound_task.cancel()


async def _drain_outbound(websocket: WebSocket, pipeline: VoiceCallPipeline, stream_sid: str) -> None:
    """Relays pipeline.outbound events to Twilio's outbound wire format until the connection
    closes. Runs as a background task so inbound audio can keep being read concurrently."""
    mark_count = 0
    while True:
        event = await pipeline.outbound.get()

        if event.kind == "audio" and event.chunk is not None:
            payload = base64.b64encode(event.chunk.audio).decode("ascii")
            await websocket.send_text(
                json.dumps({"event": "media", "streamSid": stream_sid, "media": {"payload": payload}})
            )
            mark_count += 1
            await websocket.send_text(
                json.dumps(
                    {"event": "mark", "streamSid": stream_sid, "mark": {"name": f"chunk-{mark_count}"}}
                )
            )
        elif event.kind == "clear":
            await websocket.send_text(json.dumps({"event": "clear", "streamSid": stream_sid}))
        # "turn_done" is an internal completion signal only — nothing to relay to Twilio for it.
