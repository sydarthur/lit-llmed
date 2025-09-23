"""Default paths and helpers for enrichment pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.core.store import ConfigStore


@dataclass
class EnrichmentPaths:
    """Resolves input/output directories for enrichment workflows."""

    input_root: Path
    output_root: Path

    @property
    def notes_root(self) -> Path:
        return self.output_root / "notes"

    @property
    def debug_root(self) -> Path:
        return self.output_root / "debug"

    def ensure(self) -> None:
        for path in (self.input_root, self.notes_root, self.debug_root):
            path.mkdir(parents=True, exist_ok=True)


def resolve_paths(config_dir: Optional[Path] = None) -> EnrichmentPaths:
    """Resolve enrichment paths using settings.toml when available."""

    config_store = ConfigStore(config_dir=config_dir) if config_dir else ConfigStore()
    settings = config_store.load_settings()
    input_path = Path(settings.get("enrich", {}).get("input_root", "pdfs_input"))
    output_path = Path(settings.get("enrich", {}).get("output_root", "output/enrich"))
    paths = EnrichmentPaths(input_root=input_path, output_root=output_path)
    paths.ensure()
    return paths
