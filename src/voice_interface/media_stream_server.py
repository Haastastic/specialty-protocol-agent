"""FastAPI app implementing Twilio's Media Streams wire protocol and routing audio to/from a
VoiceCallPipeline. This is the one file in this package that speaks Twilio's actual message
schema (https://www.twilio.com/docs/voice/media-streams/websocket-messages) — pipeline.py, stt.py
and tts.py are all transport-agnostic.

_stt_factory()/_tts_factory() default to the real DeepgramSTT/ElevenLabsTTS adapters, which need
DEEPGRAM_API_KEY/ELEVENLABS_API_KEY (and ELEVENLABS_VOICE_ID) set — see .env.example. Swap either
factory back to MockSTT/MockTTS for local testing without vendor accounts (simulate_call.py does
this already).

/twiml and /media-stream verify Twilio's X-Twilio-Signature using TWILIO_AUTH_TOKEN and reject
anything unsigned. Set TWILIO_SKIP_SIGNATURE_CHECK=1 to bypass that for local testing only.
"""

import asyncio
import base64
import json
import logging
import os

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from src.protocol_agent.agent import ProtocolAgent

from .pipeline import RespondingAgent, VoiceCallPipeline
from .stt import DeepgramSTT, SpeechToText
from .tts import ElevenLabsTTS, TextToSpeech
from .twilio_auth import is_valid_signature, signature_check_enabled
from .twiml import stream_twiml

logger = logging.getLogger(__name__)

app = FastAPI()

_agent: ProtocolAgent | None = None


def _agent_singleton() -> RespondingAgent:
    global _agent
    if _agent is None:
        _agent = ProtocolAgent()
    return _agent


def _stt_factory() -> SpeechToText:
    return DeepgramSTT()


def _tts_factory() -> TextToSpeech:
    return ElevenLabsTTS(voice_id=os.environ["ELEVENLABS_VOICE_ID"])


def _signature_ok(headers, host: str, path: str, params: dict[str, str] | None = None) -> bool:
    if not signature_check_enabled():
        return True
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    if not auth_token:
        logger.error("TWILIO_AUTH_TOKEN is not set; rejecting request (or set TWILIO_SKIP_SIGNATURE_CHECK=1)")
        return False
    urls = [f"https://{host}{path}", f"wss://{host}{path}"]
    return is_valid_signature(auth_token, headers.get("x-twilio-signature"), urls, params)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/twiml")
async def twiml(request: Request) -> Response:
    form = {k: str(v) for k, v in (await request.form()).items()}
    if not _signature_ok(request.headers, request.url.netloc, request.url.path, form):
        return Response(status_code=403, content="Invalid Twilio signature")
    ws_url = f"wss://{request.url.netloc}/media-stream"
    return Response(content=stream_twiml(ws_url), media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket) -> None:
    if not _signature_ok(websocket.headers, websocket.url.netloc, websocket.url.path):
        await websocket.close(code=1008)
        return
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
        if pipeline is not None:
            stt_close = getattr(pipeline.stt, "close", None)
            if stt_close is not None:
                await stt_close()


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
