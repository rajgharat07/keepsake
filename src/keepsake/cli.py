from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Allow `python -m keepsake` from project root with src layout
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from keepsake import agent, db, memory
from keepsake.config import get_settings
from keepsake.tracing import init_tracing

app = typer.Typer(
    name="keepsake",
    help="Offline AI memory companion for one grandparent.",
    add_completion=False,
)
console = Console()


@app.command()
def init() -> None:
    """Create Tiger schema (pgvector + tsvector + hypertable)."""
    init_tracing()
    db.init_schema()
    info = db.healthcheck()
    console.print(
        Panel.fit(
            f"[green]Schema ready[/green]\n"
            f"{info['postgres']}\nextensions~{info['extensions_seen']}\n"
            f"memories: {info['memories']}",
            title="Keepsake / Tiger Data",
        )
    )


@app.command()
def seed(
    path: Path = typer.Option(
        Path("data/sample_memories"),
        "--path",
        "-p",
        help="Folder of sample memory markdown files",
    ),
) -> None:
    """Load sample grandma stories so the demo works immediately."""
    init_tracing()
    db.init_schema()
    if not path.exists():
        console.print(f"[red]Missing seed folder:[/red] {path}")
        raise typer.Exit(1)
    files = sorted(path.glob("*.md"))
    total = 0
    for f in files:
        ids = memory.ingest_file(f)
        total += len(ids)
        console.print(f"  ingested {f.name} -> {len(ids)} chunk(s)")

    # Historical rows so onthisday is demoable immediately
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc)
    for years_ago, title, content in [
        (
            12,
            "First monsoon after the move",
            "Dadi hung mango leaves on the door and said a new house needs old blessings.",
        ),
        (
            40,
            "Letter from the village post",
            "Dadaji's letter arrived late. Dadi read it twice before she sat down.",
        ),
    ]:
        created = today.replace(year=today.year - years_ago)
        ids = memory.ingest_text(
            content,
            kind="event",
            title=title,
            people=["Dadi"],
            tags=["onthisday", "archive"],
            created_at=created,
        )
        total += len(ids)
        console.print(f"  onthisday archive -> {title}")

    console.print(
        f"[bold green]Seeded {total} memory chunks[/bold green] "
        f"(total rows: {db.count_memories()})"
    )


@app.command()
def ingest(
    file: Path = typer.Argument(..., exists=True, readable=True),
    kind: str = typer.Option("story", "--kind", "-k"),
) -> None:
    """Ingest a real memory file from the family."""
    init_tracing()
    db.init_schema()
    ids = memory.ingest_file(file, kind=kind)
    console.print(f"[green]Stored {len(ids)} chunk(s)[/green] ids={ids}")


@app.command()
def chat(
    question: Optional[str] = typer.Argument(None),
    no_store: bool = typer.Option(False, "--no-store", help="Do not grow memory from this turn"),
) -> None:
    """Ask Keepsake something about your loved one."""
    init_tracing()
    settings = get_settings()
    if question is None:
        question = typer.prompt(f"Ask about {settings.keepsake_person_name}")
    with console.status("Remembering…"):
        result = agent.ask(question, store=not no_store)

    console.print(
        Panel(
            result.reply,
            title=f"Keepsake · {settings.keepsake_person_name}",
            border_style="cyan",
        )
    )
    if result.memories:
        table = Table(title="Retrieved memories (hybrid search)", show_lines=False)
        table.add_column("kind", style="magenta")
        table.add_column("title")
        table.add_column("rrf", justify="right")
        for m in result.memories:
            table.add_row(m.kind, (m.title or "")[:40], f"{m.score:.4f}")
        console.print(table)
    tok = result.tokens
    console.print(
        f"[dim]tokens prompt={tok.get('prompt_tokens')} "
        f"completion={tok.get('completion_tokens')} "
        f"stored={result.stored}[/dim]"
    )


@app.command("onthisday")
def onthisday() -> None:
    """Surface memories from this calendar day across years."""
    init_tracing()
    blurb = agent.on_this_day_blurb()
    if not blurb:
        console.print("[yellow]No memories landed on this calendar day yet.[/yellow]")
        console.print("Tip: seed sample data, or ingest stories with historical dates.")
        raise typer.Exit(0)
    console.print(Panel(blurb, title="On this day", border_style="yellow"))


@app.command()
def status() -> None:
    """Quick health check for Tiger + local models config."""
    settings = get_settings()
    console.print(f"person: {settings.keepsake_person_name} ({settings.keepsake_person_relation})")
    console.print(f"chat model: {settings.ollama_chat_model}")
    console.print(f"embed model: {settings.ollama_embed_model}")
    console.print(f"sentry: {'on' if settings.sentry_dsn else 'off'}")
    try:
        info = db.healthcheck()
        console.print(f"[green]tiger ok[/green] · {info}")
    except Exception as exc:
        console.print(f"[red]tiger error:[/red] {exc}")


if __name__ == "__main__":
    app()
