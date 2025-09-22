#!/usr/bin/env python3
"""
Direct Zotero API import with manual configuration
"""

import json
import sys
from zotero_integration import ZoteroIntegration

def import_to_zotero_api(json_file, api_key, user_id, library_type="user", collection_name=None):
    """
    Import articles directly to Zotero via API.
    
    Args:
        json_file: Path to JSON file with articles
        api_key: Zotero API key
        user_id: Zotero user ID
        library_type: "user" or "group"
        collection_name: Name for new collection (optional)
    """
    print(f"📚 Importing articles from {json_file} to Zotero...")
    
    # Load articles
    with open(json_file, 'r') as f:
        articles = json.load(f)
    
    # Filter valid articles
    articles = [a for a in articles if a.get('title') and a.get('title') != "Issue Information"]
    print(f"Found {len(articles)} articles to import")
    
    # Setup Zotero integration
    config = {
        "api_key": api_key,
        "user_id": user_id,
        "library_type": library_type
    }
    
    zotero = ZoteroIntegration(**config)
    
    # Create collection if requested
    collection_key = None
    if collection_name:
        print(f"📁 Creating collection: {collection_name}")
        collection_key = zotero.create_collection_via_api(collection_name)
        
        if collection_key:
            print(f"✅ Created collection with key: {collection_key}")
        else:
            print("⚠️  Could not create collection, importing to main library")
    
    # Import articles
    print("🚀 Importing articles...")
    success = zotero.import_via_api(articles, collection_key)
    
    if success:
        print(f"✅ Successfully imported {len(articles)} articles!")
        if collection_name and collection_key:
            print(f"   Added to collection: {collection_name}")
        print("\n📖 Articles imported:")
        for i, article in enumerate(articles[:5], 1):
            print(f"   {i}. {article.get('title', 'No title')[:60]}...")
        if len(articles) > 5:
            print(f"   ... and {len(articles) - 5} more")
    else:
        print("❌ Import failed")
    
    return success

def show_usage():
    """Show usage instructions."""
    print("🔗 Direct Zotero API Import")
    print("=" * 50)
    print("Usage:")
    print("  python direct_zotero_import.py <json_file> <api_key> <user_id> [collection_name]")
    print()
    print("Get credentials from: https://www.zotero.org/settings/keys")
    print()
    print("Example:")
    print("  python direct_zotero_import.py jbl_articles_auburn.json YOUR_API_KEY YOUR_USER_ID 'JBL Articles'")
    print()
    print("Quick test (will create 'Literature Import YYYY-MM-DD' collection):")
    print("  python direct_zotero_import.py jbl_articles_auburn.json YOUR_API_KEY YOUR_USER_ID auto")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        show_usage()
        sys.exit(1)
    
    json_file = sys.argv[1]
    api_key = sys.argv[2]
    user_id = sys.argv[3]
    
    # Collection name handling
    collection_name = None
    if len(sys.argv) > 4:
        if sys.argv[4] == "auto":
            from datetime import datetime
            collection_name = f"Literature Import {datetime.now().strftime('%Y-%m-%d')}"
        else:
            collection_name = sys.argv[4]
    
    try:
        success = import_to_zotero_api(json_file, api_key, user_id, collection_name=collection_name)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)