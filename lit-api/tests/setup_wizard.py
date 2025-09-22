#!/usr/bin/env python3
"""
Setup wizard for Literature Fetcher configuration.
"""

import sys
import json
import getpass
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_manager import ConfigManager
from utils.zotero_client import ZoteroClient
from journal_config import JournalConfigManager, JournalConfig

class SetupWizard:
    """Interactive setup wizard for Literature Fetcher."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config_manager.ensure_output_directories()
    
    def welcome(self):
        """Display welcome message."""
        print("🚀 Literature Fetcher Setup Wizard")
        print("=" * 50)
        print("This wizard will help you configure the Literature Fetcher system.")
        print("You can skip any section by pressing Enter.\n")
    
    def setup_zotero(self) -> bool:
        """Setup Zotero API configuration."""
        print("📚 Zotero API Setup")
        print("-" * 30)
        print("To use direct Zotero import, you need:")
        print("1. API Key from https://www.zotero.org/settings/keys")
        print("2. Your User ID (shown on the same page)")
        print()
        
        setup_zotero = input("Setup Zotero integration? (y/n) [n]: ").strip().lower()
        
        if setup_zotero not in ['y', 'yes']:
            print("Skipping Zotero setup. You can still export RIS files.\n")
            return False
        
        try:
            api_key = getpass.getpass("Enter your Zotero API key: ").strip()
            if not api_key:
                print("No API key provided. Skipping Zotero setup.\n")
                return False
            
            user_id = input("Enter your Zotero User ID: ").strip()
            if not user_id:
                print("No User ID provided. Skipping Zotero setup.\n")
                return False
            
            print("Testing connection...")
            client = ZoteroClient(api_key, user_id)
            
            if client.test_connection():
                print("✅ Connection successful!")
                
                self.config_manager.save_zotero_config(api_key, user_id)
                print("✅ Zotero configuration saved!\n")
                return True
            else:
                print("❌ Connection failed. Please check your credentials.\n")
                return False
                
        except KeyboardInterrupt:
            print("\nSetup cancelled.\n")
            return False
        except Exception as e:
            print(f"❌ Error during setup: {e}\n")
            return False
    
    def setup_journals(self) -> bool:
        """Setup journal configurations."""
        print("📖 Journal Configuration")
        print("-" * 30)
        print("Configure journals to fetch literature from.")
        print()
        
        config_journals = input("Configure journals? (y/n) [y]: ").strip().lower()
        
        if config_journals in ['n', 'no']:
            print("Using default journal configuration.\n")
            return True
        
        try:
            journal_manager = JournalConfigManager()
            
            print(f"Current journals ({len(journal_manager.journals)}):")
            for i, journal in enumerate(journal_manager.journals, 1):
                status = "✓" if journal.active else "✗"
                print(f"  {i}. {status} {journal.name} ({journal.issn})")
            print()
            
            while True:
                print("Options:")
                print("1. Add new journal")
                print("2. Configure Zotero collections")
                print("3. Toggle journal active/inactive")
                print("4. Continue")
                
                choice = input("Choose option (1-4) [4]: ").strip() or "4"
                
                if choice == "1":
                    self._add_journal(journal_manager)
                elif choice == "2":
                    self._configure_zotero_collections(journal_manager)
                elif choice == "3":
                    self._toggle_journal(journal_manager)
                elif choice == "4":
                    break
                else:
                    print("Invalid choice. Try again.")
            
            print("✅ Journal configuration completed!\n")
            return True
            
        except KeyboardInterrupt:
            print("\nSetup cancelled.\n")
            return False
        except Exception as e:
            print(f"❌ Error during journal setup: {e}\n")
            return False
    
    def _add_journal(self, journal_manager: JournalConfigManager):
        """Add a new journal."""
        print("\nAdd New Journal:")
        
        name = input("Journal name: ").strip()
        if not name:
            print("Name required. Cancelled.")
            return
        
        issn = input("ISSN: ").strip()
        if not issn:
            print("ISSN required. Cancelled.")
            return
        
        publisher = input("Publisher [Unknown]: ").strip() or "Unknown"
        subject = input("Subject area [General]: ").strip() or "General"
        collection = input("Zotero collection name [auto]: ").strip()
        
        if not collection:
            collection = f"{name} Articles"
        
        journal = JournalConfig(
            name=name,
            issn=issn,
            publisher=publisher,
            subject_area=subject,
            zotero_collection=collection
        )
        
        journal_manager.add_journal(journal)
        print(f"✅ Added journal: {name}\n")
    
    def _configure_zotero_collections(self, journal_manager: JournalConfigManager):
        """Configure Zotero collection names."""
        print("\nConfigure Zotero Collections:")
        
        for i, journal in enumerate(journal_manager.journals, 1):
            current = journal.zotero_collection or f"{journal.name} Articles"
            print(f"{i}. {journal.name}")
            print(f"   Current collection: {current}")
            
            new_collection = input(f"   New collection name [keep current]: ").strip()
            
            if new_collection:
                journal_manager.update_journal(journal.issn, zotero_collection=new_collection)
                print(f"   ✅ Updated to: {new_collection}")
            
            print()
    
    def _toggle_journal(self, journal_manager: JournalConfigManager):
        """Toggle journal active status."""
        print("\nToggle Journal Status:")
        
        for i, journal in enumerate(journal_manager.journals, 1):
            status = "✓ Active" if journal.active else "✗ Inactive"
            print(f"{i}. {journal.name} - {status}")
        
        try:
            choice = int(input("Select journal number to toggle: ").strip())
            if 1 <= choice <= len(journal_manager.journals):
                journal = journal_manager.journals[choice - 1]
                new_status = not journal.active
                journal_manager.update_journal(journal.issn, active=new_status)
                
                status_text = "activated" if new_status else "deactivated"
                print(f"✅ {journal.name} {status_text}\n")
            else:
                print("Invalid selection.\n")
        except (ValueError, IndexError):
            print("Invalid input.\n")
    
    def setup_scheduler(self) -> bool:
        """Setup scheduler configuration."""
        print("⏰ Scheduler Configuration")
        print("-" * 30)
        print("Configure automated literature fetching.")
        print()
        
        setup_scheduler = input("Configure scheduler? (y/n) [n]: ").strip().lower()
        
        if setup_scheduler not in ['y', 'yes']:
            print("Using default scheduler settings.\n")
            return True
        
        try:
            config = self.config_manager.load_scheduler_config()
            
            print("Current settings:")
            print(f"  Fetch interval: {config['fetch_interval_hours']} hours")
            print(f"  Export CSV: {config['export_csv']}")
            print(f"  Export RIS: {config['export_ris']}")
            print(f"  Export to Zotero: {config['export_zotero']}")
            print()
            
            # Update settings
            new_interval = input(f"Fetch interval in hours [{config['fetch_interval_hours']}]: ").strip()
            if new_interval:
                try:
                    config['fetch_interval_hours'] = int(new_interval)
                except ValueError:
                    print("Invalid interval. Using default.")
            
            export_csv = input(f"Export CSV files? (y/n) [{'y' if config['export_csv'] else 'n'}]: ").strip().lower()
            if export_csv:
                config['export_csv'] = export_csv in ['y', 'yes']
            
            export_ris = input(f"Export RIS files? (y/n) [{'y' if config['export_ris'] else 'n'}]: ").strip().lower()
            if export_ris:
                config['export_ris'] = export_ris in ['y', 'yes']
            
            # Only ask about Zotero export if Zotero is configured
            if self.config_manager.load_zotero_config():
                export_zotero = input(f"Auto-import to Zotero? (y/n) [{'y' if config['export_zotero'] else 'n'}]: ").strip().lower()
                if export_zotero:
                    config['export_zotero'] = export_zotero in ['y', 'yes']
            
            self.config_manager.save_scheduler_config(config)
            print("✅ Scheduler configuration saved!\n")
            return True
            
        except KeyboardInterrupt:
            print("\nSetup cancelled.\n")
            return False
        except Exception as e:
            print(f"❌ Error during scheduler setup: {e}\n")
            return False
    
    def completion(self):
        """Display completion message."""
        print("🎉 Setup Complete!")
        print("=" * 50)
        print("Your Literature Fetcher is ready to use!")
        print()
        print("Quick start commands:")
        print("  python main.py --email you@email.com fetch-single 0028-0836")
        print("  python main.py --email you@email.com fetch-all")
        print("  python main.py --email you@email.com config --list")
        print()
        print("For help: python main.py --help")
    
    def run(self):
        """Run the complete setup wizard."""
        try:
            self.welcome()
            
            # Run setup sections
            self.setup_zotero()
            self.setup_journals()
            self.setup_scheduler()
            
            self.completion()
            return True
            
        except KeyboardInterrupt:
            print("\n\nSetup cancelled by user.")
            return False
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            return False

if __name__ == "__main__":
    wizard = SetupWizard()
    success = wizard.run()
    sys.exit(0 if success else 1)