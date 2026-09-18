"""Wire-protocol tests for media_stream_server.py using FastAPI's TestClient — no real socket, no
Twilio account, no Anthropic/Deepgram/ElevenLabs API calls (agent, STT, and TTS factories are all
monkeypatched to stubs/mocks)."""

import base64
import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.voice_interface import media_stream_server
from src.voice_interface.stt import MockSTT
from src.voice_interface.tts import MockTTS
from src.voice_interface.twilio_auth import compute_signature

AUTH_TOKEN = "test-token"


@pytest.fixture
def skip_signature_check(monkeypatch):
    monkeypatch.setenv("TWILIO_SKIP_SIGNATURE_CHECK", "1")


@pytest.fixture
def enforce_signature_check(monkeypatch):
    monkeypatch.delenv("TWILIO_SKIP_SIGNATURE_CHECK", raising=False)
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", AUTH_TOKEN)


class _StubAgent:
    def respond(self, session: dict, user_utterance: str) -> str:
        session["messages"].append({"role": "user", "content": user_utterance})
        return "Got it."


def test_twiml_endpoint_returns_stream_url(skip_signature_check):
    client = TestClient(media_stream_server.app)
    response = client.post("/twiml")
    assert response.status_code == 200
    assert "wss://" in response.text
    assert "/media-stream" in response.text


def test_media_stream_round_trip(monkeypatch, skip_signature_check):
    monkeypatch.setattr(media_stream_server, "_agent_singleton", lambda: _StubAgent())
    monkeypatch.setattr(
        media_stream_server,
        "_stt_factory",
        lambda: MockSTT(["hello"], frames_per_utterance=1, recognition_delay_s=0.0),
    )
    monkeypatch.setattr(media_stream_server, "_tts_factory", lambda: MockTTS())

    client = TestClient(media_stream_server.app)
    with client.websocket_connect("/media-stream") as ws:
        ws.send_text(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        ws.send_text(
            json.dumps(
                {
                    "event": "start",
                    "streamSid": "MZ123",
                    "start": {
                        "streamSid": "MZ123",
                        "callSid": "CA123",
                        "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
                    },
                }
            )
        )
        payload = base64.b64encode(b"\x00" * 160).decode("ascii")
        ws.send_text(
            json.dumps(
                {
                    "event": "media",
                    "streamSid": "MZ123",
                    "media": {"track": "inbound", "chunk": "1", "timestamp": "20", "payload": payload},
                }
            )
        )

        media_msg = json.loads(ws.receive_text())
        mark_msg = json.loads(ws.receive_text())

        assert media_msg["event"] == "media"
        assert media_msg["streamSid"] == "MZ123"
        assert len(base64.b64decode(media_msg["media"]["payload"])) > 0
        assert mark_msg["event"] == "mark"
        assert mark_msg["streamSid"] == "MZ123"

        ws.send_text(json.dumps({"event": "stop", "streamSid": "MZ123", "stop": {"callSid": "CA123"}}))


def test_twiml_rejects_unsigned_request(enforce_signature_check):
    client = TestClient(media_stream_server.app)
    assert client.post("/twiml", data={"CallSid": "CA123"}).status_code == 403


def test_twiml_rejects_bad_signature(enforce_signature_check):
    client = TestClient(media_stream_server.app)
    response = client.post("/twiml", data={"CallSid": "CA123"}, headers={"X-Twilio-Signature": "bogus"})
    assert response.status_code == 403


def test_twiml_accepts_valid_signature(enforce_signature_check):
    client = TestClient(media_stream_server.app)
    params = {"CallSid": "CA123", "From": "+15550100"}
    signature = compute_signature(AUTH_TOKEN, "https://testserver/twiml", params)
    response = client.post("/twiml", data=params, headers={"X-Twilio-Signature": signature})
    assert response.status_code == 200
    assert "wss://testserver/media-stream" in response.text


def test_twiml_fails_closed_without_auth_token(monkeypatch):
    monkeypatch.delenv("TWILIO_SKIP_SIGNATURE_CHECK", raising=False)
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    client = TestClient(media_stream_server.app)
    assert client.post("/twiml").status_code == 403


def test_media_stream_rejects_unsigned_handshake(enforce_signature_check):
    client = TestClient(media_stream_server.app)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/media-stream"):
            pass


def test_media_stream_accepts_signed_handshake(enforce_signature_check):
    client = TestClient(media_stream_server.app)
    signature = compute_signature(AUTH_TOKEN, "wss://testserver/media-stream")
    with client.websocket_connect("/media-stream", headers={"X-Twilio-Signature": signature}) as ws:
        ws.send_text(json.dumps({"event": "stop", "streamSid": "MZ123"}))


def test_healthz_needs_no_signature(enforce_signature_check):
    assert TestClient(media_stream_server.app).get("/healthz").json() == {"status": "ok"}
