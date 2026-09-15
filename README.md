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

- **`protocols/`** — specialty knowledge as YAML config, not model weights. Two specialties
  (dermatology, cardiology) included; adding a third requires zero code changes.
- **`src/protocol_agent/`** — TF-IDF retrieval over protocol chunks + a Claude agent with tool use
  (mock scheduling). Escalation triggers always surface regardless of retrieval ranking — a
  deliberate safety-over-relevance choice.
- **`src/synth_data/`** — generates synthetic call scenarios per specialty/intent, then scores and
  distills them into a curated few-shot library. Output schema matches `ai-agent-eval-harness`'s
  scenario format exactly.

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

## Roadmap

See `CLAUDE.md` for full build-phase plan. Phase 3 adds a `src/voice_interface/` package (Twilio
Media Streams, streaming STT/TTS) that calls `ProtocolAgent.respond()` unchanged — the text-based
interface here was deliberately designed so voice is a swap-in layer, not a rewrite.

## Disclaimer

Portfolio demo using synthetic scenarios and a mock scheduling backend. Not connected to real
patient data or any production system.
