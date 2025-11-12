"""Simple scheduler built on top of the fetch job."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import schedule

from src.core.log import get_logger
from src.core.store import ConfigStore, ContentStore
from src.features.journal_fetch.fetch_job import FetchJob

LOGGER = get_logger(__name__)


@dataclass
class SchedulerConfig:
    fetch_interval_hours: int = 24
    retention_days: int = 30


class LiteratureScheduler:
    def __init__(
        self,
        email: str,
        *,
        config_store: Optional[ConfigStore] = None,
        content_store: Optional[ContentStore] = None,
        job_config: Optional[SchedulerConfig] = None,
    ):
        self.email = email
        self.config_store = config_store or ConfigStore()
        self.content_store = content_store or ContentStore()
        self.job_config = job_config or SchedulerConfig()
        self.job = FetchJob(email, config_store=self.config_store)
        self.is_running = False

    def schedule(self, interval_hours: Optional[int] = None) -> None:
        interval_hours = interval_hours or self.job_config.fetch_interval_hours
        schedule.every(interval_hours).hours.do(self._run_job)
        LOGGER.info("scheduler_registered", interval_hours=interval_hours)

    def _run_job(self) -> None:
        LOGGER.info("scheduler_job_start")
        summary = self.job.run_full_fetch()
        LOGGER.info("scheduler_job_complete", total=summary["total_articles"])
        if self.job_config.retention_days > 0:
            self._cleanup_old_files(self.job_config.retention_days)

    def start(self, run_immediately: bool = False) -> None:
        self.is_running = True
        if run_immediately:
            self._run_job()
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)

    def stop(self) -> None:
        self.is_running = False
        schedule.clear()

    def run_once(self) -> None:
        self._run_job()

    def _cleanup_old_files(self, retention_days: int) -> None:
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        for pattern in ("*.json", "*.csv", "*.ris"):
            for path in self.content_store.paths.data.glob(pattern):
                if _is_stale(path, cutoff):
                    path.unlink(missing_ok=True)
        for pattern in ("*.csv", "*.ris"):
            for directory in (self.content_store.paths.csv, self.content_store.paths.ris):
                for path in directory.glob(pattern):
                    if _is_stale(path, cutoff):
                        path.unlink(missing_ok=True)


def _is_stale(path: Path, cutoff: datetime) -> bool:
    return datetime.utcfromtimestamp(path.stat().st_mtime) < cutoff
