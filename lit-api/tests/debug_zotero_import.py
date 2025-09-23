#!/usr/bin/env python3
"""
Debug Zotero import by testing single article
"""

import json
import requests
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ris_exporter import RISExporter

def test_single_article_import(api_key, user_id, article):
    """Test importing a single article to debug issues."""
    print(f"🧪 Testing single article import...")
    print(f"Article: {article.get('title', 'No title')[:60]}...")
    
    headers = {
        'Zotero-API-Key': api_key,
        'Content-Type': 'application/json',
        'User-Agent': 'Literature Fetcher/1.0'
    }
    
    # Convert to Zotero format
    creators = []
    for author in article.get('authors', []):
        # Split name into first and last
        parts = author.strip().split()
        if len(parts) >= 2:
            creators.append({
                "creatorType": "author",
                "firstName": " ".join(parts[:-1]),
                "lastName": parts[-1]
            })
        else:
            creators.append({
                "creatorType": "author",
                "name": author
            })
    
    # Clean abstract
    ris_exporter = RISExporter()
    abstract = ris_exporter.clean_abstract_text(article.get('abstract', ''))
    
    # Create minimal item first
    item = {
        "itemType": "journalArticle",
        "title": article.get('title', ''),
        "creators": creators[:5],  # Limit to 5 authors to avoid issues
        "publicationTitle": article.get('journal_name', ''),
        "date": article.get('published_date', ''),
        "DOI": article.get('doi', ''),
        "url": article.get('open_access_url') or article.get('url', ''),
        "abstractNote": abstract[:10000] if abstract else ""  # Limit abstract length
    }
    
    # Remove empty fields
    item = {k: v for k, v in item.items() if v}
    
    print(f"📋 Item data:")
    print(json.dumps(item, indent=2)[:500] + "...")
    
    try:
        url = f"https://api.zotero.org/users/{user_id}/items"
        response = requests.post(url, headers=headers, json=[item], timeout=30)
        
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 201:
            print("✅ Success! Article imported.")
            result = response.json()
            print(f"Result: {result}")
            return True
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def import_articles_one_by_one(api_key, user_id, articles):
    """Import articles one by one to identify problematic ones."""
    print(f"🔄 Importing {len(articles)} articles one by one...")
    
    successful = 0
    failed = 0
    
    for i, article in enumerate(articles, 1):
        if not article.get('title') or article.get('title') == "Issue Information":
            continue
            
        print(f"\n--- Article {i}/{len(articles)} ---")
        
        if test_single_article_import(api_key, user_id, article):
            successful += 1
            print(f"✅ Success ({successful} total)")
        else:
            failed += 1
            print(f"❌ Failed ({failed} total)")
        
        # Small delay between requests
        import time
        time.sleep(1)
    
    print(f"\n📊 Summary:")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    
    return successful > 0

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python debug_zotero_import.py <json_file> <api_key> <user_id>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    api_key = sys.argv[2]
    user_id = sys.argv[3]
    
    with open(json_file, 'r') as f:
        articles = json.load(f)
    
    # Filter valid articles
    articles = [a for a in articles if a.get('title') and a.get('title') != "Issue Information"]
    
    import_articles_one_by_one(api_key, user_id, articles)