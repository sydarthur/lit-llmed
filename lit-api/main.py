#!/usr/bin/env python3
"""
Literature Fetcher CLI
A command-line tool for fetching and managing academic literature metadata.
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

from get_latest_literature import LiteratureFetcher
from multi_journal_fetcher import MultiJournalFetcher
from journal_config import JournalConfigManager, JournalConfig
from scheduler import LiteratureScheduler, run_scheduler_daemon
from utils.config_manager import ConfigManager
from utils.ris_exporter import RISExporter

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def cmd_fetch_single(args):
    """Fetch articles from a single journal."""
    # Setup config manager and ensure directories exist
    config_manager = ConfigManager()
    config_manager.ensure_output_directories()
    paths = config_manager.get_output_paths()
    
    fetcher = LiteratureFetcher(args.email, args.rate_limit)
    
    articles = fetcher.fetch_journal_articles(
        issn=args.issn,
        rows=args.max_articles,
        days_back=args.days_back,
        fetch_abstracts=args.abstracts,
        fetch_oa_links=args.oa_links
    )
    
    if args.output:
        fetcher.save_articles_json(articles, args.output)
        
        # Also create RIS file if requested
        if args.create_ris and articles:
            ris_exporter = RISExporter()
            ris_file = args.output.replace('.json', '.ris')
            ris_exporter.export_from_json(args.output, ris_file)
            print(f"RIS file created: {ris_file}")
            
    else:
        # Save to default location
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        journal_name = args.issn.replace('-', '_')
        output_file = paths["data"] / f"{journal_name}_{timestamp}.json"
        
        fetcher.save_articles_json(articles, str(output_file))
        print(f"Saved to: {output_file}")
        
        # Create RIS file by default
        if articles:
            ris_exporter = RISExporter()
            ris_file = paths["ris"] / f"{journal_name}_{timestamp}.ris"
            ris_exporter.export_to_ris(articles, str(ris_file))
            print(f"RIS file: {ris_file}")
        
        # Print summary to console
        print(f"\nFetched {len(articles)} articles:")
        for i, article in enumerate(articles[:5], 1):
            print(f"{i}. {article.title}")
            print(f"   DOI: {article.doi}")
            print(f"   Authors: {', '.join(article.authors[:3]) if article.authors else 'No authors'}")
            print(f"   Published: {article.published_date}")
            print()

def cmd_fetch_all(args):
    """Fetch articles from all configured journals."""
    # Load Zotero config if available
    config_manager = ConfigManager()
    zotero_config = None
    
    if hasattr(args, 'zotero_api') and args.zotero_api:
        # Manual Zotero credentials provided
        api_key, user_id = args.zotero_api
        zotero_config = {
            "api_key": api_key,
            "user_id": user_id,
            "library_type": "user"
        }
    else:
        # Try to load from config file
        zotero_config = config_manager.load_zotero_config()
    
    multi_fetcher = MultiJournalFetcher(
        email=args.email,
        config_file=args.config,
        rate_limit_delay=args.rate_limit,
        max_workers=args.workers
    )
    
    summary = multi_fetcher.run_full_fetch(
        export_csv=args.csv,
        export_ris=args.ris,
        export_zotero=args.zotero,
        zotero_config=zotero_config
    )
    
    print(f"\nFetch Summary:")
    print(f"Total articles: {summary['total_articles']}")
    print(f"Journals processed: {summary['journals_fetched']}")
    
    # Show output files
    if "csv_file" in summary:
        print(f"CSV export: {summary['csv_file']}")
    if "ris_file" in summary:
        print(f"RIS export: {summary['ris_file']}")
    if "zotero_import" in summary:
        print(f"Zotero import: {summary['zotero_import']}")

def cmd_config_journals(args):
    """Manage journal configurations."""
    config_manager = JournalConfigManager(args.config)
    
    if args.list:
        journals = config_manager.get_active_journals()
        print(f"\nConfigured Journals ({len(journals)}):")
        for journal in journals:
            status = "✓" if journal.active else "✗"
            print(f"{status} {journal.name} ({journal.issn}) - {journal.subject_area}")
    
    elif args.add:
        journal = JournalConfig(
            name=args.add[0],
            issn=args.add[1],
            publisher=args.add[2] if len(args.add) > 2 else "Unknown",
            subject_area=args.add[3] if len(args.add) > 3 else "General",
            zotero_collection=args.add[4] if len(args.add) > 4 else f"{args.add[0]} Articles"
        )
        config_manager.add_journal(journal)
        print(f"Added journal: {journal.name}")
        print(f"Zotero collection: {journal.zotero_collection}")
    
    elif args.remove:
        config_manager.remove_journal(args.remove)
        print(f"Removed journal with ISSN: {args.remove}")
    
    elif args.toggle:
        journal = config_manager.get_journal(args.toggle)
        if journal:
            config_manager.update_journal(args.toggle, active=not journal.active)
            status = "activated" if not journal.active else "deactivated"
            print(f"Journal {journal.name} {status}")
        else:
            print(f"Journal with ISSN {args.toggle} not found")

def cmd_schedule(args):
    """Manage scheduled fetching."""
    scheduler = LiteratureScheduler(args.email, args.config)
    
    if args.start:
        print("Starting literature fetcher scheduler...")
        run_scheduler_daemon(args.email, args.config)
    
    elif args.run_once:
        print("Running single fetch...")
        scheduler.run_once()
        print("Fetch completed")
    
    elif args.status:
        status = scheduler.get_status()
        print(f"Scheduler Status:")
        print(f"Running: {status['is_running']}")
        print(f"Scheduled jobs: {status['scheduled_jobs']}")
        print(f"Next run: {status['next_run'] or 'Not scheduled'}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Literature Fetcher - Fetch academic literature metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--email', required=True, 
                       help='Email address for API requests (required for Unpaywall)')
    parser.add_argument('--config', default='config/journal_configs.json',
                       help='Journal configuration file')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single journal fetch command
    single_parser = subparsers.add_parser('fetch-single', help='Fetch from a single journal')
    single_parser.add_argument('issn', help='Journal ISSN')
    single_parser.add_argument('--max-articles', type=int, default=50,
                              help='Maximum articles to fetch')
    single_parser.add_argument('--days-back', type=int, default=30,
                              help='Fetch articles from last N days')
    single_parser.add_argument('--output', help='Output JSON file')
    single_parser.add_argument('--rate-limit', type=float, default=1.0,
                              help='Rate limit delay between requests')
    single_parser.add_argument('--no-abstracts', dest='abstracts', action='store_false',
                              help='Skip fetching abstracts')
    single_parser.add_argument('--no-oa-links', dest='oa_links', action='store_false',
                              help='Skip fetching open access links')
    single_parser.add_argument('--create-ris', action='store_true',
                              help='Create RIS file alongside JSON output')
    
    # Multi-journal fetch command
    multi_parser = subparsers.add_parser('fetch-all', help='Fetch from all configured journals')
    multi_parser.add_argument('--workers', type=int, default=5,
                             help='Number of parallel workers')
    multi_parser.add_argument('--rate-limit', type=float, default=1.0,
                             help='Rate limit delay between requests')
    multi_parser.add_argument('--no-csv', dest='csv', action='store_false',
                             help='Skip CSV export')
    multi_parser.add_argument('--no-ris', dest='ris', action='store_false',
                             help='Skip RIS export')
    multi_parser.add_argument('--zotero', action='store_true',
                             help='Import directly to Zotero (requires saved config)')
    multi_parser.add_argument('--zotero-api', nargs=2, metavar=('API_KEY', 'USER_ID'),
                             help='Import directly to Zotero via API (requires API key and user ID)')
    
    # Journal configuration command
    config_parser = subparsers.add_parser('config', help='Manage journal configurations')
    config_group = config_parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument('--list', action='store_true',
                             help='List configured journals')
    config_group.add_argument('--add', nargs='+', metavar=('NAME', 'ISSN', 'PUBLISHER', 'SUBJECT', 'COLLECTION'),
                             help='Add journal (name issn [publisher] [subject] [zotero_collection])')
    config_group.add_argument('--remove', metavar='ISSN',
                             help='Remove journal by ISSN')
    config_group.add_argument('--toggle', metavar='ISSN',
                             help='Toggle journal active status')
    
    # Scheduler command
    schedule_parser = subparsers.add_parser('schedule', help='Manage scheduled fetching')
    schedule_group = schedule_parser.add_mutually_exclusive_group(required=True)
    schedule_group.add_argument('--start', action='store_true',
                               help='Start scheduler daemon')
    schedule_group.add_argument('--run-once', action='store_true',
                               help='Run single scheduled fetch')
    schedule_group.add_argument('--status', action='store_true',
                               help='Show scheduler status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    setup_logging(args.log_level)
    
    try:
        if args.command == 'fetch-single':
            cmd_fetch_single(args)
        elif args.command == 'fetch-all':
            cmd_fetch_all(args)
        elif args.command == 'config':
            cmd_config_journals(args)
        elif args.command == 'schedule':
            cmd_schedule(args)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())