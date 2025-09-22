#!/usr/bin/env python3
"""
Interactive setup for Zotero API integration
"""

import json
import getpass
from pathlib import Path
from zotero_integration import ZoteroIntegration

def setup_zotero_credentials():
    """Interactive setup for Zotero API credentials."""
    print("🔑 Zotero API Setup")
    print("=" * 50)
    print("You'll need:")
    print("1. API Key from https://www.zotero.org/settings/keys")
    print("2. Your User ID (shown on the same page)")
    print()
    
    # Get credentials
    api_key = getpass.getpass("Enter your Zotero API key: ").strip()
    user_id = input("Enter your Zotero User ID: ").strip()
    
    # Library type
    print("\nLibrary type:")
    print("1. Personal library (user)")
    print("2. Group library (group)")
    choice = input("Choose (1 or 2) [default: 1]: ").strip() or "1"
    library_type = "user" if choice == "1" else "group"
    
    if library_type == "group":
        group_id = input("Enter Group ID: ").strip()
        user_id = group_id  # For groups, we use group_id as the identifier
    
    # Collection preference
    create_collection = input("Create new collection for imports? (y/n) [default: y]: ").strip().lower()
    create_collection = create_collection in ['y', 'yes', ''] 
    
    # Save configuration
    config = {
        "api_key": api_key,
        "user_id": user_id,
        "library_type": library_type,
        "create_collection": create_collection
    }
    
    config_file = "zotero_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Configuration saved to {config_file}")
    return config

def test_zotero_connection(config):
    """Test the Zotero API connection."""
    print("\n🧪 Testing Zotero connection...")
    
    try:
        zotero = ZoteroIntegration(**config)
        
        # Test by trying to create a collection
        if config.get("create_collection"):
            collection_name = "Literature Fetcher Test"
            collection_key = zotero.create_collection_via_api(collection_name)
            
            if collection_key:
                print(f"✅ Successfully created test collection: {collection_name}")
                print(f"   Collection key: {collection_key}")
                return True
            else:
                print("❌ Failed to create collection")
                return False
        else:
            print("✅ Configuration looks valid (collection creation skipped)")
            return True
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def import_articles_via_api(json_file, config):
    """Import articles directly via Zotero API."""
    print(f"\n📚 Importing articles from {json_file}...")
    
    with open(json_file, 'r') as f:
        articles = json.load(f)
    
    # Filter out non-articles
    articles = [a for a in articles if a.get('title') and a.get('title') != "Issue Information"]
    
    zotero = ZoteroIntegration(**config)
    
    # Create collection if requested
    collection_key = None
    if config.get("create_collection"):
        from datetime import datetime
        collection_name = f"Literature Import {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        collection_key = zotero.create_collection_via_api(collection_name)
        
        if collection_key:
            print(f"📁 Created collection: {collection_name}")
        else:
            print("⚠️  Could not create collection, importing to main library")
    
    # Import articles
    success = zotero.import_via_api(articles, collection_key)
    
    if success:
        print(f"✅ Successfully imported {len(articles)} articles to Zotero!")
        if collection_key:
            print(f"   Added to collection: {collection_name}")
    else:
        print("❌ Import failed")
    
    return success

if __name__ == "__main__":
    import sys
    
    print("🔗 Zotero API Direct Integration")
    print("=" * 50)
    
    # Check if config exists
    config_file = Path("zotero_config.json")
    
    if config_file.exists():
        print("Found existing configuration.")
        use_existing = input("Use existing config? (y/n) [default: y]: ").strip().lower()
        
        if use_existing in ['y', 'yes', '']:
            with open(config_file) as f:
                config = json.load(f)
        else:
            config = setup_zotero_credentials()
    else:
        config = setup_zotero_credentials()
    
    # Test connection
    if test_zotero_connection(config):
        print("\n🎉 Zotero API is ready!")
        
        # Import articles if JSON file provided
        if len(sys.argv) > 1:
            json_file = sys.argv[1]
            import_articles_via_api(json_file, config)
        else:
            print(f"\nTo import articles, run:")
            print(f"python setup_zotero_api.py <json_file>")
            print(f"Example: python setup_zotero_api.py jbl_articles_auburn.json")
    else:
        print("\n❌ Please check your credentials and try again")