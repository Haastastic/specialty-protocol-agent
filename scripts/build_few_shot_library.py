"""CLI: score generated scenarios and distill them into data/few_shot_library.yaml."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from rich.console import Console

from src.synth_data.distill import distill, write_library

console = Console()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/synthetic_transcripts")
    parser.add_argument("--top-n", type=int, default=2)
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.[/red]")
        return

    all_scenarios = []
    for path in Path(args.input_dir).glob("*.json"):
        with open(path) as f:
            all_scenarios.extend(json.load(f))

    if not all_scenarios:
        console.print(f"[red]No generated scenarios found in {args.input_dir}. Run generate_synthetic_data.py first.[/red]")
        return

    console.print(f"Scoring and distilling {len(all_scenarios)} scenarios...")
    library = distill(all_scenarios, top_n_per_category=args.top_n)
    write_library(library)
    console.print("[green]Wrote few-shot library to data/few_shot_library.yaml[/green]")


if __name__ == "__main__":
    main()
