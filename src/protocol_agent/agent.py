"""ProtocolAgent: general-purpose Claude agent whose specialty behavior comes entirely from
retrieved protocol-pack config, not from model training. This is the core deliverable — the thing
that answers 'how do you get specialty depth without millions of interactions.'

Interface contract (see CLAUDE.md, architecture principle 4): respond() is the seam a future
voice interface would call per-turn. Do not add voice/latency/audio concerns here.
"""

import os

import anthropic

from .mock_tools import TOOL_IMPLS, TOOL_SCHEMAS
from .retrieval import ProtocolIndex

BASE_SYSTEM_PROMPT = """You are a patient scheduling assistant for a healthcare provider
network. You help patients book, reschedule, or cancel appointments, answer FAQs, and route
insurance and clinical questions appropriately.

Rules you must always follow:
- Never give medical advice, diagnose symptoms, or recommend treatment.
- If a patient's message matches an escalation trigger below, immediately tell them to call 911 or
  go to the nearest emergency room, and stop the scheduling flow.
- Always confirm specialty, date/time, and patient name back to the patient before booking.
- If unsure, ask a clarifying question rather than guessing.
- Be warm and direct. No corporate filler.

Below is retrieved specialty-specific guidance relevant to this patient's message. Treat it as
authoritative for this conversation. If nothing relevant was retrieved, rely on the general rules
above and ask clarifying questions.

RETRIEVED GUIDANCE:
{retrieved_context}
"""


def _format_chunks(chunks) -> str:
    if not chunks:
        return "(none retrieved for this turn)"
    lines = []
    for c in chunks:
        lines.append(f"[{c.specialty} / {c.section}] {c.text}")
    return "\n".join(lines)


class ProtocolAgent:
    def __init__(self, protocol_dir: str = "protocols", model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.index = ProtocolIndex(protocol_dir)

    def run_conversation(self, patient_turns: list[str]) -> list[dict]:
        """Same shape as ai-agent-eval-harness's SchedulingAgent.run_conversation — this is what
        makes the two repos composable for eval purposes."""
        session = {"specialty": None, "messages": []}
        transcript = []

        for patient_msg in patient_turns:
            transcript.append({"role": "patient", "content": patient_msg})
            reply = self.respond(session, patient_msg)
            transcript.append({"role": "agent", "content": reply})

        return transcript

    def respond(self, session: dict, user_utterance: str) -> str:
        """The interface-agnostic core. Text in, text out, one turn at a time.
        A future voice interface calls this exact method per recognized utterance."""

        if session["specialty"] is None:
            detected = self.index.detect_specialty(user_utterance)
            if detected:
                session["specialty"] = detected

        chunks = self.index.retrieve(user_utterance, specialty=session["specialty"], top_k=4)
        system_prompt = BASE_SYSTEM_PROMPT.format(retrieved_context=_format_chunks(chunks))

        session["messages"].append({"role": "user", "content": user_utterance})
        working_messages = list(session["messages"])

        for _ in range(5):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                tools=TOOL_SCHEMAS,
                messages=working_messages,
            )

            if response.stop_reason != "tool_use":
                reply = "".join(b.text for b in response.content if b.type == "text")
                session["messages"].append({"role": "assistant", "content": reply})
                return reply

            working_messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                impl = TOOL_IMPLS.get(block.name)
                result = impl(**block.input) if impl else {"error": "unknown_tool"}
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": str(result)}
                )
            working_messages.append({"role": "user", "content": tool_results})

        return "[agent exceeded tool-call budget without a final reply]"
