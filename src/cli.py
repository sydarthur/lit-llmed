"""Typer-based CLI entrypoint for lit-llmed."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from src.core.log import configure_logging, get_logger
from src.core.models import Journal
from src.core.store import ConfigStore
from src.jobs.fetch import FetchJob
from src.jobs.scheduler import LiteratureScheduler, SchedulerConfig

app = typer.Typer(help="Fetch and enrich journal metadata from Crossref.")
LOGGER = get_logger(__name__)


def _config_store(config_dir: Optional[str]) -> ConfigStore:
    return ConfigStore() if not config_dir else ConfigStore(Path(config_dir))


@app.callback()
def common(
    ctx: typer.Context,
    log_level: str = typer.Option("INFO", help="Logging level"),
    config_dir: Optional[str] = typer.Option(None, help="Override configuration directory"),
) -> None:
    """Configure logging and attach shared objects to context."""
    configure_logging(log_level)
    ctx.obj = {
        "config_store": _config_store(config_dir),
    }


@app.command("fetch-single")
def fetch_single(
    ctx: typer.Context,
    issn: str = typer.Argument(..., help="Journal ISSN"),
    email: str = typer.Option(..., help="Contact email for API usage"),
) -> None:
    """Fetch articles for a single journal configured in journals.json."""
    config_store: ConfigStore = ctx.obj["config_store"]
    journals = config_store.load_journals()
    journal = next((journal for journal in journals if journal.issn == issn), None)
    if not journal:
        raise typer.BadParameter(f"ISSN {issn} not found in configuration")
    job = FetchJob(email, config_store=config_store)
    articles = job.fetch_single(journal)
    typer.echo(f"Fetched {len(articles)} articles for {journal.name}")


@app.command("fetch-all")
def fetch_all(
    ctx: typer.Context,
    email: str = typer.Option(..., help="Contact email for API usage"),
    csv: bool = typer.Option(True, help="Export combined CSV"),
    ris: bool = typer.Option(True, help="Export combined RIS"),
    zotero: bool = typer.Option(False, help="Push articles directly to Zotero"),
) -> None:
    """Fetch all active journals and produce exports."""
    config_store: ConfigStore = ctx.obj["config_store"]
    job = FetchJob(email, config_store=config_store)
    summary = job.run_full_fetch(export_csv=csv, export_ris=ris, export_zotero=zotero)
    typer.echo(f"Fetched {summary['total_articles']} articles across {summary['journals_fetched']} journals")
    if csv_path := summary.get("csv_file"):
        typer.echo(f"CSV export: {csv_path}")
    if ris_path := summary.get("ris_file"):
        typer.echo(f"RIS export: {ris_path}")
    if zotero_status := summary.get("zotero_import"):
        typer.echo(f"Zotero: {zotero_status}")


@app.command("schedule")
def schedule_fetch(
    ctx: typer.Context,
    email: str = typer.Option(..., help="Contact email for API usage"),
    interval_hours: int = typer.Option(24, help="Run frequency in hours"),
    run_once: bool = typer.Option(False, help="Run a single fetch and exit"),
) -> None:
    """Start the lightweight scheduler or execute a single scheduled run."""

    config_store: ConfigStore = ctx.obj["config_store"]
    scheduler = LiteratureScheduler(
        email,
        config_store=config_store,
        job_config=SchedulerConfig(fetch_interval_hours=interval_hours),
    )
    if run_once:
        scheduler.run_once()
        typer.echo("Scheduled fetch executed once")
        return
    typer.echo("Starting scheduler. Press Ctrl+C to stop.")
    try:
        scheduler.schedule(interval_hours)
        scheduler.start(run_immediately=True)
    except KeyboardInterrupt:  # pragma: no cover - interactive path
        scheduler.stop()
        typer.echo("Scheduler stopped")


journals_app = typer.Typer(help="Manage journal configuration entries.")
app.add_typer(journals_app, name="journals")


@journals_app.command("list")
def list_journals(ctx: typer.Context, show_inactive: bool = typer.Option(False, help="Show inactive journals")) -> None:
    config_store: ConfigStore = ctx.obj["config_store"]
    journals = config_store.load_journals()
    for journal in journals:
        if not show_inactive and not journal.active:
            continue
        status = "active" if journal.active else "inactive"
        typer.echo(f"{journal.name} ({journal.issn}) - {status}")


@journals_app.command("add")
def add_journal(
    ctx: typer.Context,
    name: str = typer.Option(..., prompt="Journal name"),
    issn: str = typer.Option(..., prompt="ISSN"),
    publisher: Optional[str] = typer.Option(None),
    subject_area: Optional[str] = typer.Option(None),
    collection: Optional[str] = typer.Option(None, help="Zotero collection name"),
) -> None:
    config_store: ConfigStore = ctx.obj["config_store"]
    journals = config_store.load_journals()
    if any(journal.issn == issn for journal in journals):
        raise typer.BadParameter(f"ISSN {issn} already exists")
    journals.append(
        Journal(
            name=name,
            issn=issn,
            publisher=publisher,
            subject_area=subject_area,
            zotero_collection=collection,
        )
    )
    config_store.save_journals(journals)
    typer.echo(f"Journal {name} added")


@journals_app.command("remove")
def remove_journal(ctx: typer.Context, issn: str = typer.Argument(...)) -> None:
    config_store: ConfigStore = ctx.obj["config_store"]
    journals = config_store.load_journals()
    remaining = [journal for journal in journals if journal.issn != issn]
    if len(remaining) == len(journals):
        raise typer.BadParameter(f"ISSN {issn} not found")
    config_store.save_journals(remaining)
    typer.echo(f"Removed journal {issn}")


@journals_app.command("toggle")
def toggle_journal(ctx: typer.Context, issn: str = typer.Argument(...)) -> None:
    config_store: ConfigStore = ctx.obj["config_store"]
    journals = config_store.load_journals()
    found = False
    for journal in journals:
        if journal.issn == issn:
            journal.active = not journal.active
            found = True
            break
    if not found:
        raise typer.BadParameter(f"ISSN {issn} not found")
    config_store.save_journals(journals)
    status = "active" if any(journal.issn == issn and journal.active for journal in journals) else "inactive"
    typer.echo(f"Journal {issn} toggled to {status}")


if __name__ == "__main__":
    app()
