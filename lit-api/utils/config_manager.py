#!/usr/bin/env python3
"""
Configuration management utilities for Literature Fetcher.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages application configuration files."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load_zotero_config(self, config_file: str = "zotero_config.json") -> Optional[Dict]:
        """Load Zotero API configuration."""
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            return None
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            required_fields = ['api_key', 'user_id']
            if all(field in config for field in required_fields):
                return config
            else:
                logger.warning(f"Zotero config missing required fields: {required_fields}")
                return None
                
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading Zotero config: {e}")
            return None
    
    def save_zotero_config(self, api_key: str, user_id: str, library_type: str = "user", 
                          config_file: str = "zotero_config.json") -> bool:
        """Save Zotero API configuration."""
        config = {
            "api_key": api_key,
            "user_id": user_id,
            "library_type": library_type,
            "create_collection": True
        }
        
        config_path = self.config_dir / config_file
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved Zotero config to {config_path}")
            return True
            
        except IOError as e:
            logger.error(f"Error saving Zotero config: {e}")
            return False
    
    def load_scheduler_config(self, config_file: str = "scheduler_config.json") -> Dict:
        """Load scheduler configuration with defaults."""
        config_path = self.config_dir / config_file
        
        default_config = {
            "fetch_interval_hours": 24,
            "output_directory": "output/data",
            "export_csv": True,
            "export_ris": True,
            "export_zotero": False,
            "max_workers": 5,
            "rate_limit_delay": 1.0,
            "log_level": "INFO",
            "retention_days": 30,
            "email_notifications": False,
            "notification_email": None
        }
        
        if not config_path.exists():
            self.save_scheduler_config(default_config, config_file)
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Merge with defaults
            return {**default_config, **config}
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading scheduler config: {e}. Using defaults.")
            return default_config
    
    def save_scheduler_config(self, config: Dict, config_file: str = "scheduler_config.json") -> bool:
        """Save scheduler configuration."""
        config_path = self.config_dir / config_file
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved scheduler config to {config_path}")
            return True
            
        except IOError as e:
            logger.error(f"Error saving scheduler config: {e}")
            return False
    
    def get_output_paths(self) -> Dict[str, Path]:
        """Get standardized output paths."""
        base_output = Path("output")
        
        return {
            "data": base_output / "data",
            "exports": base_output / "exports", 
            "logs": base_output / "logs",
            "ris": base_output / "exports" / "ris",
            "csv": base_output / "exports" / "csv"
        }
    
    def ensure_output_directories(self):
        """Create all output directories."""
        paths = self.get_output_paths()
        
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
            
        logger.info("Created output directories")