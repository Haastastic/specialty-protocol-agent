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

**What's real vs. mocked.** This repo has no Twilio account and no STT/TTS vendor keys, only
`ANTHROPIC_API_KEY`. So:

- `media_stream_server.py` implements Twilio's actual [Media Streams wire
  protocol](https://www.twilio.com/docs/voice/media-streams/websocket-messages) (`start`/`media`/
  `stop` inbound, `media`/`mark`/`clear` outbound) over a FastAPI WebSocket — this part is
  protocol-correct and swap-in-ready for a live Twilio number.
- Speech-to-text and text-to-speech are pluggable (`SpeechToText`/`TextToSpeech` in `stt.py`/
  `tts.py`) and default to mocks — `MockSTT` is driven by a pre-scripted list of caller lines
  instead of doing real audio decoding, `MockTTS` produces placeholder audio sized/timed like real
  8kHz mulaw instead of doing real synthesis. This keeps the whole pipeline runnable and testable
  with zero external services beyond Anthropic. Swap in a real vendor adapter by implementing
  either interface — nothing else changes.

**Run it end to end** (uses the real `ProtocolAgent`, same `ANTHROPIC_API_KEY` as above):

```bash
python scripts/simulate_call.py
```

Prints a scripted call transcript, including a deliberate barge-in demonstration, plus a per-stage
latency report checked against budget (`src/voice_interface/latency.py`).

**Pointing a real Twilio number at it:** run `uvicorn src.voice_interface.media_stream_server:app`,
point the number's voice webhook at `POST /twiml`, and replace `_stt_factory()` in
`media_stream_server.py` with a real `SpeechToText` adapter first — it raises `NotImplementedError`
by default since `MockSTT`'s scripted-lines approach can't transcribe a real caller.

## Disclaimer

Portfolio demo using synthetic scenarios and a mock scheduling backend. Not connected to real
patient data or any production system.
