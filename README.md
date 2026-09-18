# Specialty Protocol Agent

Reproduces the core capability behind a specialty-aware patient scheduling voice/chat agent —
the kind of product a patient-access AI platform offers — without a proprietary corpus of
millions of real patient interactions.

## Thesis

Specialty depth doesn't have to come from training on scale. It can come from a curated knowledge
layer (protocol packs: scheduling rules, payer logic, FAQ, escalation triggers) retrieved at call
time, combined with synthetic data generation to bootstrap edge-case coverage before real traffic
exists. This repo builds both halves as one system.

## Components

- **`protocols/`** — specialty knowledge as YAML config, not model weights. Six specialties
  (dermatology, cardiology, pediatrics, orthopedics, oncology, primary care) included; adding a
  seventh requires zero code changes.
- **`src/protocol_agent/`** — TF-IDF retrieval over protocol chunks + a Claude agent with tool use
  (mock scheduling). Escalation triggers always surface regardless of retrieval ranking — a
  deliberate safety-over-relevance choice.
- **`src/synth_data/`** — generates synthetic call scenarios per specialty/intent, then scores and
  distills them into a curated few-shot library. Output schema matches `ai-agent-eval-harness`'s
  scenario format exactly.
- **`src/voice_interface/`** — wraps `ProtocolAgent.respond()`, unchanged, with a real Twilio
  Media Streams call-handling layer, pluggable (mocked by default) streaming STT/TTS, per-stage
  latency budget tracking, and basic barge-in. See "Voice interface" below.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY

# Generate synthetic scenarios for a specialty
python scripts/generate_synthetic_data.py --specialty dermatology --count-per-intent 2

# Distill into a few-shot library
python scripts/build_few_shot_library.py

# Run the retrieval tests (no API key needed)
python -m pytest tests/ -v
```

To exercise the agent directly:

```python
from src.protocol_agent.agent import ProtocolAgent

agent = ProtocolAgent()
transcript = agent.run_conversation([
    "I have a mole that's been changing color and growing",
    "Yes it's on my arm, been about 3 weeks"
])
```

## Composing with the eval harness

This agent exposes the same `run_conversation(patient_turns) -> transcript` interface as
`ai-agent-eval-harness`'s `SchedulingAgent`. To regression-test this agent with that repo's rule and
LLM-judge evaluators, swap the agent import in that repo's `runner.py`:

```python
# in ai-agent-eval-harness/src/agent_harness/runner.py
from specialty_protocol_agent.src.protocol_agent.agent import ProtocolAgent as SchedulingAgent
```

Generated scenarios from `scripts/generate_synthetic_data.py` use the exact same YAML shape as that
repo's `scenarios/*.yaml` — they can be copied directly into that folder.

## Voice interface (Phase 3)

`src/voice_interface/` puts a real-time call layer around `ProtocolAgent.respond()` without
touching it — same `respond(session, user_utterance) -> str` contract `run_conversation()` already
uses above, just called per recognized utterance instead of per scripted string.

**What's real vs. mocked.** Speech-to-text and text-to-speech are pluggable (`SpeechToText`/
`TextToSpeech` in `stt.py`/`tts.py`):

- `media_stream_server.py` implements Twilio's actual [Media Streams wire
  protocol](https://www.twilio.com/docs/voice/media-streams/websocket-messages) (`start`/`media`/
  `stop` inbound, `media`/`mark`/`clear` outbound) over a FastAPI WebSocket, and defaults to real
  vendor adapters: `DeepgramSTT` (streaming transcription over Deepgram's WebSocket API) and
  `ElevenLabsTTS` (streaming synthesis over ElevenLabs' REST endpoint, requested directly in
  Twilio's 8kHz mulaw wire format). These need `DEEPGRAM_API_KEY`, `ELEVENLABS_API_KEY`, and
  `ELEVENLABS_VOICE_ID` set (see `.env.example`).
- `scripts/simulate_call.py` uses `MockSTT`/`MockTTS` instead — `MockSTT` is driven by a
  pre-scripted list of caller lines rather than real audio decoding, `MockTTS` produces placeholder
  audio sized/timed like real 8kHz mulaw rather than doing real synthesis. This keeps the local
  harness runnable and testable with zero external services beyond Anthropic. Swap either factory
  in `media_stream_server.py` back to the mocks for testing the Twilio wiring itself without vendor
  accounts.

**Run it end to end locally** (uses the real `ProtocolAgent`, same `ANTHROPIC_API_KEY` as above,
mocked STT/TTS):

```bash
python scripts/simulate_call.py
```

Prints a scripted call transcript, including a deliberate barge-in demonstration, plus a per-stage
latency report checked against budget (`src/voice_interface/latency.py`).

**Pointing a real Twilio number at it:** set `DEEPGRAM_API_KEY`, `ELEVENLABS_API_KEY`, and
`ELEVENLABS_VOICE_ID` in `.env`, run `uvicorn src.voice_interface.media_stream_server:app`
(tunnel it with e.g. `ngrok http 8000` for local testing, since Twilio needs a public HTTPS/WSS
endpoint), then point the number's Voice webhook at `POST https://<your-host>/twiml`. Requests are
signature-checked, so also set `TWILIO_AUTH_TOKEN` (or `TWILIO_SKIP_SIGNATURE_CHECK=1` for local
testing only). To host it instead of tunneling, see "Deploying to Cloud Run" below.

## Deploying to Cloud Run

Cloud Run gives a public HTTPS/WSS URL with no tunnel, scales to zero between calls, and a demo's
traffic sits inside its free tier. The `Dockerfile` ships only `src/`, `protocols/`, and the
few-shot library; `.env` is excluded from the image, so keys go in as secrets.

**Signature validation.** `/twiml` and `/media-stream` reject any request without a valid
`X-Twilio-Signature` (checked against `TWILIO_AUTH_TOKEN`), so strangers can't spend your
Deepgram/ElevenLabs/Anthropic credits. It fails closed: a missing token rejects everything. For
local-only testing, `TWILIO_SKIP_SIGNATURE_CHECK=1` disables the check — never set it on a public
deployment.

```bash
# one-time: create secrets from your .env values
for name in ANTHROPIC_API_KEY DEEPGRAM_API_KEY ELEVENLABS_API_KEY ELEVENLABS_VOICE_ID TWILIO_AUTH_TOKEN; do
  printf "%s" "<value>" | gcloud secrets create $name --data-file=-
done

gcloud run deploy specialty-protocol-agent \
  --source . --region us-central1 --allow-unauthenticated \
  --timeout 3600 --max-instances 2 --concurrency 10 --memory 512Mi \
  --set-secrets ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest,DEEPGRAM_API_KEY=DEEPGRAM_API_KEY:latest,ELEVENLABS_API_KEY=ELEVENLABS_API_KEY:latest,ELEVENLABS_VOICE_ID=ELEVENLABS_VOICE_ID:latest,TWILIO_AUTH_TOKEN=TWILIO_AUTH_TOKEN:latest
```

Notes:

- `--allow-unauthenticated` is required because Twilio can't present Google credentials; the
  signature check above is what protects the service.
- `--timeout 3600` lets a call's WebSocket stay open up to an hour. `--max-instances` and
  `--concurrency` cap simultaneous calls, and so your worst-case spend.
- Cold start after idle takes a few seconds; add `--min-instances 1` (a few dollars a month) for
  a demo where the first call must connect instantly.
- Grant the service's runtime service account `roles/secretmanager.secretAccessor`.
- **Fresh project?** Before the first deploy, enable the Compute Engine API
  (`gcloud services enable compute.googleapis.com`). It creates the default compute service
  account, which both the secret binding and Cloud Run's source build depend on; without it you get
  `Service account ...-compute@developer.gserviceaccount.com does not exist` and a
  `PERMISSION_DENIED` build failure. Then grant that account (`<project-number>-compute@developer.gserviceaccount.com`)
  `roles/secretmanager.secretAccessor`, `roles/run.builder`, `roles/logging.logWriter`,
  `roles/artifactregistry.writer`, and `roles/storage.objectViewer`.
- Point the Twilio number's Voice webhook at `POST https://<service-url>/twiml`. `GET /health`
  is unauthenticated and does no work.

### Try it

Call the live demo line: **+1 (816) 704-6267**. It's a portfolio demo on a small budget, so it may be
offline at any time, and it only handles the synthetic scenarios and mock scheduling
backend (see the disclaimer below). Don't share real personal or health information on the call.

## Disclaimer

Portfolio demo using synthetic scenarios and a mock scheduling backend. Not connected to real
patient data or any production system.
