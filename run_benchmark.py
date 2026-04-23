from __future__ import annotations
import json
from pathlib import Path
import typer
from rich import print
from rich.progress import track
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl
app = typer.Typer(add_completion=False)

@app.command()
def main(
    dataset: str = "data/hotpot_real.json",
    out_dir: str = "outputs/real_run",
    reflexion_attempts: int = 3,
) -> None:
    examples = load_dataset(dataset)
    print(f"[cyan]Loaded {len(examples)} examples from {dataset}[/cyan]")

    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts, adaptive=True)

    print("[yellow]Running ReAct agent...[/yellow]")
    react_records = []
    for example in track(examples, description="ReAct"):
        react_records.append(react.run(example))

    print("[yellow]Running Reflexion agent...[/yellow]")
    reflexion_records = []
    for example in track(examples, description="Reflexion"):
        reflexion_records.append(reflexion.run(example))

    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)

    report = build_report(all_records, dataset_name=Path(dataset).name, mode="real")
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
