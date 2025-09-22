import schedule
import time
import logging
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from multi_journal_fetcher import MultiJournalFetcher

logger = logging.getLogger(__name__)

class LiteratureScheduler:
    def __init__(self, email: str, config_file: str = "scheduler_config.json"):
        self.email = email
        self.config_file = Path(config_file)
        self.fetcher = None
        self.config = self.load_scheduler_config()
        self.is_running = False
        
    def load_scheduler_config(self) -> dict:
        """Load scheduler configuration."""
        default_config = {
            "fetch_interval_hours": 24,
            "output_directory": "literature_data",
            "export_csv": True,
            "max_workers": 5,
            "rate_limit_delay": 1.0,
            "log_level": "INFO",
            "retention_days": 30,
            "email_notifications": False,
            "notification_email": None
        }
        
        if not self.config_file.exists():
            self.save_scheduler_config(default_config)
            return default_config
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            return {**default_config, **config}
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading scheduler config: {e}. Using defaults.")
            return default_config
    
    def save_scheduler_config(self, config: dict):
        """Save scheduler configuration."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def initialize_fetcher(self):
        """Initialize the multi-journal fetcher."""
        if not self.fetcher:
            self.fetcher = MultiJournalFetcher(
                email=self.email,
                rate_limit_delay=self.config["rate_limit_delay"],
                max_workers=self.config["max_workers"]
            )
    
    def fetch_job(self):
        """Job function to fetch literature."""
        try:
            logger.info("Starting scheduled literature fetch")
            self.initialize_fetcher()
            
            summary = self.fetcher.run_full_fetch(
                output_dir=self.config["output_directory"],
                export_csv=self.config["export_csv"]
            )
            
            logger.info(f"Scheduled fetch completed. Total articles: {summary['total_articles']}")
            
            # Clean up old files if retention is set
            if self.config["retention_days"] > 0:
                self.cleanup_old_files()
                
        except Exception as e:
            logger.error(f"Error in scheduled fetch: {e}")
    
    def cleanup_old_files(self):
        """Remove files older than retention period."""
        try:
            output_dir = Path(self.config["output_directory"])
            if not output_dir.exists():
                return
            
            cutoff_date = datetime.now() - timedelta(days=self.config["retention_days"])
            removed_count = 0
            
            for file_path in output_dir.glob("*.json"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
                    removed_count += 1
            
            for file_path in output_dir.glob("*.csv"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
                    removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old files")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def schedule_fetch(self, interval_hours: Optional[int] = None):
        """Schedule regular literature fetches."""
        if interval_hours is None:
            interval_hours = self.config["fetch_interval_hours"]
        
        schedule.every(interval_hours).hours.do(self.fetch_job)
        logger.info(f"Scheduled literature fetch every {interval_hours} hours")
    
    def run_once(self):
        """Run a single fetch immediately."""
        self.fetch_job()
    
    def start_scheduler(self, run_immediately: bool = False):
        """Start the scheduler."""
        self.is_running = True
        
        if run_immediately:
            logger.info("Running initial fetch...")
            self.run_once()
        
        logger.info("Starting literature scheduler...")
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def stop_scheduler(self):
        """Stop the scheduler."""
        self.is_running = False
        schedule.clear()
        logger.info("Scheduler stopped")
    
    def get_status(self) -> dict:
        """Get scheduler status."""
        next_run = None
        if schedule.jobs:
            next_run = min(job.next_run for job in schedule.jobs)
            next_run = next_run.isoformat() if next_run else None
        
        return {
            "is_running": self.is_running,
            "scheduled_jobs": len(schedule.jobs),
            "next_run": next_run,
            "config": self.config
        }

def run_scheduler_daemon(email: str, config_file: str = "scheduler_config.json"):
    """Run scheduler as a daemon process."""
    import signal
    import sys
    
    scheduler = LiteratureScheduler(email, config_file)
    
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        scheduler.stop_scheduler()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    scheduler.schedule_fetch()
    scheduler.start_scheduler(run_immediately=True)