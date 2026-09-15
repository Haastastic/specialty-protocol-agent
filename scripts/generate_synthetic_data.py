"""CLI: generate synthetic scenarios for a specialty, write to data/synthetic_transcripts/."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from rich.console import Console

from src.synth_data.generator import generate_batch

console = Console()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specialty", required=True)
    parser.add_argument("--count-per-intent", type=int, default=2)
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.[/red]")
        return

    console.print(f"Generating synthetic scenarios for [bold]{args.specialty}[/bold]...")
    scenarios = generate_batch(args.specialty, args.count_per_intent)

    out_dir = Path("data/synthetic_transcripts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.specialty}_batch.json"
    with open(out_path, "w") as f:
        json.dump(scenarios, f, indent=2)

    console.print(f"[green]Wrote {len(scenarios)} scenarios to {out_path}[/green]")


if __name__ == "__main__":
    main()
