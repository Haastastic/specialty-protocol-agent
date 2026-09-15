# CLAUDE.md — Specialty Protocol Agent

Orientation file for Claude Code sessions on this repo. Read before making changes.

## Purpose

Portfolio project demonstrating how a specialty-aware voice/chat patient agent — the category of
product a patient-access AI platform offers — can be reproduced without a proprietary corpus of
millions of real patient interactions. The thesis: specialty depth comes from a curated **knowledge layer** (protocol packs)
retrieved at call time, not from training a model on scale. Scale is then generated synthetically
to seed and validate that knowledge layer.

This repo combines two things that are meant to be read as one system:

1. **Protocol-pack RAG agent** (`src/protocol_agent/`) — a general-purpose Claude agent whose
   specialty behavior comes entirely from retrieved config, not fine-tuning
2. **Synthetic data generation** (`src/synth_data/`) — generates call scenarios and transcripts to
   bootstrap both the protocol packs' edge-case coverage and a few-shot example library, standing in
   for the "compounding dataset" a real company accumulates from live traffic

## Architecture principles

1. **Specialty knowledge is data, not weights.** Every specialty lives in `protocols/<name>.yaml`
   with four sections: `scheduling_rules`, `payer_logic`, `faq`, `escalation_triggers`. Adding a
   14th specialty should never require touching agent code — this mirrors how a real patient-access
   platform scales across dozens of specialties far more honestly than fine-tuning would.

2. **Retrieval, not full-context stuffing.** The agent retrieves only the relevant protocol sections
   per turn (`retrieval.py`, TF-IDF over sections) rather than dumping every specialty's full config
   into the system prompt. This keeps token cost flat as specialty count grows — directly answers
   the "inexpensive" half of the brief.

3. **Synthetic data has a real purpose, not just volume.** Generated transcripts feed two things:
   - new eval scenarios (same YAML schema as `ai-agent-eval-harness`, so generated scenarios can be
     dropped straight into that repo's `scenarios/` folder)
   - a distilled few-shot library (`data/few_shot_library.yaml`) used to steer tone/format, curated
     down from many generated examples rather than used in bulk

4. **Agent core is interface-agnostic — this is the seam for Option 3.**
   `ProtocolAgent.respond(session, user_utterance) -> str` is the entire contract. `run_conversation()`
   in this repo drives it with a scripted list of strings (text-in/text-out, for eval purposes). A
   future voice interface (streaming STT -> `respond()` -> streaming TTS) would call the *same*
   method per turn. **Do not let voice-specific concerns (latency, barge-in, audio buffering) leak
   into `agent.py` or `retrieval.py`.** Those belong in a future `src/voice_interface/` package that
   depends on this one, never the reverse.

5. **Composability with `ai-agent-eval-harness`.** This repo's `ProtocolAgent` exposes the same
   `run_conversation(patient_turns) -> transcript` shape as that repo's `SchedulingAgent`. To
   regression-test this agent, point the harness's runner at this class instead of its built-in one
   rather than duplicating the eval logic here.

## Tech stack

- Python 3.11+
- `anthropic` SDK for both the agent and the synthetic data generator
- `scikit-learn` (TfidfVectorizer) for lightweight retrieval — no vector DB needed at this scale
- `pyyaml` for protocol packs and generated scenarios
- No database — protocol packs and generated data are files, this is a portfolio-scale demo

## Build phases

- **Phase 1 (this scaffold):** 2 protocol packs (dermatology, cardiology), TF-IDF retrieval, agent
  with tool use, synthetic scenario generator, distillation into few-shot library
- **Phase 2:** expand to 4-5 protocol packs, generate 50+ synthetic scenarios per specialty, wire
  generated scenarios into `ai-agent-eval-harness` to regression-test this agent
- **Phase 3 (Option 3 integration):** add `src/voice_interface/` — Twilio Media Streams for call
  handling, streaming STT, streaming TTS, latency budget measurement per pipeline stage. Reuses
  `ProtocolAgent.respond()` unchanged.

## Conventions

- Every protocol pack YAML must include all four top-level keys even if a section is empty
- Retrieval always returns chunk provenance (which specialty, which section) alongside text, so the
  agent's system prompt can be inspected/debugged
- No hardcoded API keys — read from `.env` via `python-dotenv`

## Orientation prompts (for future Claude Code sessions)

- "Add a specialty" → new YAML in `protocols/`, no code changes
- "Generate synthetic scenarios for X" → `python scripts/generate_synthetic_data.py --specialty X --count N`
- "Build the few-shot library" → `python scripts/build_few_shot_library.py`
- "Test this agent with the eval harness" → see README "Composing with the eval harness" section
- "Start the voice interface" → this is Phase 3, not yet scaffolded; read architecture principle 4 first
