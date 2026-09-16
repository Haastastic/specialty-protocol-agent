"""Per-turn, per-stage latency measurement and budget checking — a voice-specific concern kept
out of src/protocol_agent/ per CLAUDE.md architecture principle 4.

Budgets are standard conversational-AI targets: caller-stops-talking to caller-hears-reply under
~1.8s reads as a natural conversation. Splitting the budget per stage means a regression can be
attributed to STT, the agent call, or TTS instead of just "the turn was slow.\""""

import time
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

STAGE_BUDGETS_MS = {
    "stt_final": 300,
    "agent_response": 1200,
    "tts_first_chunk": 300,
}
TOTAL_BUDGET_MS = 1800


@dataclass
class TurnLatency:
    turn_start: float
    stt_final: float | None = None
    agent_response_ready: float | None = None
    tts_first_chunk: float | None = None

    def stage_ms(self) -> dict[str, float]:
        stages = {}
        if self.stt_final is not None:
            stages["stt_final"] = (self.stt_final - self.turn_start) * 1000
        if self.agent_response_ready is not None and self.stt_final is not None:
            stages["agent_response"] = (self.agent_response_ready - self.stt_final) * 1000
        if self.tts_first_chunk is not None and self.agent_response_ready is not None:
            stages["tts_first_chunk"] = (self.tts_first_chunk - self.agent_response_ready) * 1000
        return stages

    def total_ms(self) -> float | None:
        if self.tts_first_chunk is None:
            return None
        return (self.tts_first_chunk - self.turn_start) * 1000


class LatencyTracker:
    """One instance per call. start_turn()/mark() record stage timestamps; render() prints a rich
    table with per-stage and total budget pass/fail."""

    def __init__(self):
        self.turns: list[TurnLatency] = []

    def start_turn(self) -> TurnLatency:
        turn = TurnLatency(turn_start=time.monotonic())
        self.turns.append(turn)
        return turn

    def mark(self, turn: TurnLatency, stage: str) -> None:
        setattr(turn, stage, time.monotonic())

    def render(self, console: Console | None = None) -> None:
        console = console or Console()
        table = Table(title="Voice pipeline latency (per turn)")
        table.add_column("Turn")
        for stage in STAGE_BUDGETS_MS:
            table.add_column(stage)
        table.add_column("total")

        for i, turn in enumerate(self.turns, start=1):
            stages = turn.stage_ms()
            row = [str(i)]
            for stage, budget in STAGE_BUDGETS_MS.items():
                row.append(_format_cell(stages.get(stage), budget))
            row.append(_format_cell(turn.total_ms(), TOTAL_BUDGET_MS))
            table.add_row(*row)

        console.print(table)


def _format_cell(value_ms: float | None, budget_ms: float) -> str:
    if value_ms is None:
        return "-"
    style = "green" if value_ms <= budget_ms else "red"
    return f"[{style}]{value_ms:.0f}ms[/{style}]"
