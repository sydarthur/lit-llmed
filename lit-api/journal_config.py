import json
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class JournalConfig:
    name: str
    issn: str
    publisher: str
    subject_area: str
    zotero_collection: str = ""  # Custom Zotero collection name
    fetch_abstracts: bool = True
    fetch_oa_links: bool = True
    max_articles_per_fetch: int = 50
    days_back: int = 30
    active: bool = True

class JournalConfigManager:
    def __init__(self, config_file: str = "config/journal_configs.json"):
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.journals: List[JournalConfig] = []
        self.load_config()
    
    def add_journal(self, journal: JournalConfig):
        """Add a new journal configuration."""
        self.journals.append(journal)
        self.save_config()
    
    def remove_journal(self, issn: str):
        """Remove a journal by ISSN."""
        self.journals = [j for j in self.journals if j.issn != issn]
        self.save_config()
    
    def get_journal(self, issn: str) -> Optional[JournalConfig]:
        """Get journal configuration by ISSN."""
        for journal in self.journals:
            if journal.issn == issn:
                return journal
        return None
    
    def get_active_journals(self) -> List[JournalConfig]:
        """Get all active journal configurations."""
        return [j for j in self.journals if j.active]
    
    def update_journal(self, issn: str, **kwargs):
        """Update journal configuration."""
        journal = self.get_journal(issn)
        if journal:
            for key, value in kwargs.items():
                if hasattr(journal, key):
                    setattr(journal, key, value)
            self.save_config()
    
    def save_config(self):
        """Save journal configurations to file."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            "journals": [asdict(journal) for journal in self.journals],
            "last_updated": str(datetime.now())
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def load_config(self):
        """Load journal configurations from file."""
        if not self.config_file.exists():
            self._create_default_config()
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            self.journals = []
            for journal_data in config_data.get("journals", []):
                self.journals.append(JournalConfig(**journal_data))
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading config: {e}. Creating default config.")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default journal configurations."""
        default_journals = [
            JournalConfig(
                name="Nature",
                issn="0028-0836",
                publisher="Nature Publishing Group",
                subject_area="Multidisciplinary Sciences",
                zotero_collection="Nature Articles"
            ),
            JournalConfig(
                name="Science",
                issn="0036-8075",
                publisher="American Association for the Advancement of Science",
                subject_area="Multidisciplinary Sciences",
                zotero_collection="Science Articles"
            ),
            JournalConfig(
                name="Cell",
                issn="0092-8674",
                publisher="Elsevier",
                subject_area="Cell Biology",
                zotero_collection="Cell Biology Papers"
            ),
            JournalConfig(
                name="The Lancet",
                issn="0140-6736",
                publisher="Elsevier",
                subject_area="Medicine",
                zotero_collection="Medical Research"
            ),
            JournalConfig(
                name="NEJM",
                issn="0028-4793",
                publisher="Massachusetts Medical Society",
                subject_area="Medicine",
                zotero_collection="Medical Research"
            ),
            JournalConfig(
                name="Journal of Business Logistics",
                issn="0735-3766",
                publisher="Wiley",
                subject_area="Supply Chain Management",
                zotero_collection="Supply Chain Research"
            )
        ]
        
        self.journals = default_journals
        self.save_config()

# Fix import
from datetime import datetime