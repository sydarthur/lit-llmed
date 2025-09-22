import json
import time
import logging
from datetime import datetime
from typing import List, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from get_latest_literature import LiteratureFetcher, Article
from journal_config import JournalConfigManager, JournalConfig
from utils.config_manager import ConfigManager
from utils.zotero_client import ZoteroClient
from utils.ris_exporter import RISExporter

logger = logging.getLogger(__name__)

class MultiJournalFetcher:
    def __init__(self, email: str, config_file: str = "config/journal_configs.json", 
                 rate_limit_delay: float = 1.0, max_workers: int = 5):
        self.email = email
        self.fetcher = LiteratureFetcher(email, rate_limit_delay)
        self.config_manager = JournalConfigManager(config_file)
        self.sys_config_manager = ConfigManager()
        self.max_workers = max_workers
        
    def fetch_single_journal(self, journal_config: JournalConfig) -> List[Article]:
        """Fetch articles for a single journal."""
        try:
            return self.fetcher.fetch_journal_articles(
                issn=journal_config.issn,
                rows=journal_config.max_articles_per_fetch,
                days_back=journal_config.days_back,
                fetch_abstracts=journal_config.fetch_abstracts,
                fetch_oa_links=journal_config.fetch_oa_links
            )
        except Exception as e:
            logger.error(f"Error fetching articles for {journal_config.name}: {e}")
            return []
    
    def fetch_all_journals(self, max_workers: int = None) -> Dict[str, List[Article]]:
        """Fetch articles from all active journals using parallel processing."""
        if max_workers is None:
            max_workers = self.max_workers
            
        active_journals = self.config_manager.get_active_journals()
        results = {}
        
        logger.info(f"Starting fetch for {len(active_journals)} journals")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all journal fetch tasks
            future_to_journal = {
                executor.submit(self.fetch_single_journal, journal): journal
                for journal in active_journals
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_journal):
                journal = future_to_journal[future]
                try:
                    articles = future.result()
                    results[journal.issn] = articles
                    logger.info(f"Completed fetch for {journal.name}: {len(articles)} articles")
                except Exception as e:
                    logger.error(f"Error fetching {journal.name}: {e}")
                    results[journal.issn] = []
        
        return results
    
    def save_all_results(self, results: Dict[str, List[Article]], 
                        output_dir: str = "output/data"):
        """Save results from all journals to separate files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual journal results
        for issn, articles in results.items():
            journal_config = self.config_manager.get_journal(issn)
            journal_name = journal_config.name if journal_config else issn
            filename = f"{journal_name.replace(' ', '_').lower()}_{timestamp}.json"
            filepath = output_path / filename
            
            self.fetcher.save_articles_json(articles, str(filepath))
        
        # Save combined results
        combined_articles = []
        for articles in results.values():
            combined_articles.extend(articles)
        
        combined_filename = f"all_journals_{timestamp}.json"
        combined_filepath = output_path / combined_filename
        self.fetcher.save_articles_json(combined_articles, str(combined_filepath))
        
        # Save summary
        summary = {
            "fetch_timestamp": timestamp,
            "total_articles": len(combined_articles),
            "journals_fetched": len(results),
            "articles_per_journal": {
                self.config_manager.get_journal(issn).name if self.config_manager.get_journal(issn) 
                else issn: len(articles) 
                for issn, articles in results.items()
            }
        }
        
        summary_filepath = output_path / f"fetch_summary_{timestamp}.json"
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(combined_articles)} total articles to {output_dir}")
        return summary
    
    def export_to_csv(self, articles: List[Article], filename: str):
        """Export articles to CSV format."""
        import csv
        
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'title', 'doi', 'authors', 'published_date', 'journal_name', 
                'issn', 'abstract', 'open_access_url', 'url', 'volume', 'issue', 'pages'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for article in articles:
                row = {
                    'title': article.title,
                    'doi': article.doi,
                    'authors': '; '.join(article.authors),
                    'published_date': article.published_date,
                    'journal_name': article.journal_name,
                    'issn': article.issn,
                    'abstract': article.abstract or '',
                    'open_access_url': article.open_access_url or '',
                    'url': article.url or '',
                    'volume': article.volume or '',
                    'issue': article.issue or '',
                    'pages': article.pages or ''
                }
                writer.writerow(row)
        
        logger.info(f"Exported {len(articles)} articles to CSV: {filename}")
    
    def run_full_fetch(self, output_dir: str = "output/data", 
                      export_csv: bool = True, export_ris: bool = True,
                      export_zotero: bool = False, zotero_config: Dict = None) -> Dict:
        """Run a complete fetch cycle for all journals."""
        logger.info("Starting full literature fetch cycle")
        
        # Ensure output directories exist
        self.sys_config_manager.ensure_output_directories()
        paths = self.sys_config_manager.get_output_paths()
        
        # Fetch from all journals
        results = self.fetch_all_journals()
        
        # Save results to data directory
        summary = self.save_all_results(results, str(paths["data"]))
        
        # Combine all articles
        combined_articles = []
        for articles in results.values():
            combined_articles.extend(articles)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export to CSV if requested
        if export_csv and combined_articles:
            csv_filename = paths["csv"] / f"all_journals_{timestamp}.csv"
            self.export_to_csv(combined_articles, str(csv_filename))
            summary["csv_file"] = str(csv_filename)
        
        # Export to RIS if requested
        if export_ris and combined_articles:
            ris_exporter = RISExporter()
            ris_filename = paths["ris"] / f"all_journals_{timestamp}.ris"
            ris_file = ris_exporter.export_to_ris(combined_articles, str(ris_filename))
            summary["ris_file"] = ris_file
            logger.info(f"Created RIS file: {ris_file}")
        
        # Export to Zotero if requested and configured
        if export_zotero and combined_articles:
            zotero_config = zotero_config or self.sys_config_manager.load_zotero_config()
            
            if zotero_config:
                try:
                    client = ZoteroClient(**zotero_config)
                    
                    # Group articles by journal and create separate collections
                    for issn, articles in results.items():
                        if not articles:
                            continue
                            
                        journal_config = self.config_manager.get_journal(issn)
                        if not journal_config:
                            continue
                        
                        # Get collection name from journal config or use default
                        collection_name = journal_config.zotero_collection or f"{journal_config.name} Articles"
                        
                        # Create or get collection
                        collection_key = client.get_or_create_collection(collection_name)
                        
                        if collection_key:
                            # Import articles to collection
                            success = client.import_articles(articles, collection_key)
                            
                            if success:
                                logger.info(f"Imported {len(articles)} articles to '{collection_name}'")
                            else:
                                logger.error(f"Failed to import articles to '{collection_name}'")
                        else:
                            logger.error(f"Failed to create collection '{collection_name}'")
                    
                    summary["zotero_import"] = "completed"
                    
                except Exception as e:
                    logger.error(f"Zotero import failed: {e}")
                    summary["zotero_import"] = f"failed: {e}"
            else:
                logger.warning("Zotero not configured, skipping direct import")
                summary["zotero_import"] = "not configured"
        
        logger.info("Full fetch cycle completed")
        return summary