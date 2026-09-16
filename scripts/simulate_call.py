"""CLI: run a simulated call through VoiceCallPipeline end-to-end using the real ProtocolAgent
with mocked STT/TTS, printing the transcript plus a per-stage latency report. This is what
actually demonstrates Phase 3 working — no Twilio account or STT/TTS vendor needed."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from rich.console import Console

from src.protocol_agent.agent import ProtocolAgent
from src.voice_interface.pipeline import VoiceCallPipeline
from src.voice_interface.stt import MockSTT
from src.voice_interface.tts import MockTTS

CALLER_TURNS = [
    "Hi, I have a mole that's been changing color and growing",
    "Yes it's on my arm, been about 3 weeks",
    "Sure, book me for the first available slot, my name is Alex Rivera",
]
BARGE_IN_LINE = "Actually wait, can you check cardiology instead"

console = Console()


async def _speak_turn(pipeline: VoiceCallPipeline, line: str) -> None:
    console.print(f"[bold green]caller:[/bold green] {line}")
    for _ in range(pipeline.stt.frames_per_utterance):
        await pipeline.handle_audio_frame(b"\x00" * 160)  # 20ms of silence-shaped payload


async def _print_reply(pipeline: VoiceCallPipeline) -> None:
    """Drains pipeline.outbound until the current turn resolves, printing the reply as it's
    synthesized — or a barge-in notice if the turn gets cleared instead."""
    reply_parts = []
    while True:
        event = await pipeline.outbound.get()
        if event.kind == "audio" and event.chunk is not None:
            reply_parts.append(event.chunk.text)
        elif event.kind == "clear":
            console.print("[yellow]  (barge-in — clearing in-flight reply)[/yellow]")
            return
        elif event.kind == "turn_done":
            console.print(f"[bold cyan]agent:[/bold cyan] {' '.join(reply_parts)}\n")
            return


async def main():
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.[/red]")
        return

    stt = MockSTT([*CALLER_TURNS[:-1], BARGE_IN_LINE, CALLER_TURNS[-1]])
    pipeline = VoiceCallPipeline(agent=ProtocolAgent(), stt=stt, tts=MockTTS())

    for line in CALLER_TURNS[:-1]:
        await _speak_turn(pipeline, line)
        await _print_reply(pipeline)

    console.print("[dim]-- demonstrating barge-in: caller interrupts before hearing the reply --[/dim]")
    await _speak_turn(pipeline, BARGE_IN_LINE)
    await asyncio.sleep(0.15)  # let the agent call start before the caller talks over it
    await _speak_turn(pipeline, CALLER_TURNS[-1])
    await _print_reply(pipeline)  # the "clear" from the interrupted turn
    await _print_reply(pipeline)  # the real reply to the final line

    pipeline.latency.render(console)


if __name__ == "__main__":
    asyncio.run(main())
