#!/usr/bin/env python3
"""
Fix collection creation and verify articles are properly assigned
"""

import json
import requests
import time

def create_collection_and_import(api_key, user_id, json_file, collection_name):
    """Create collection and import articles with proper assignment."""
    
    headers = {
        'Zotero-API-Key': api_key,
        'Content-Type': 'application/json',
        'User-Agent': 'Literature Fetcher/1.0'
    }
    
    base_url = f"https://api.zotero.org/users/{user_id}"
    
    print(f"📁 Creating collection: {collection_name}")
    
    # Step 1: Create collection
    collection_data = [{
        "name": collection_name,
        "parentCollection": False
    }]
    
    try:
        response = requests.post(f"{base_url}/collections", headers=headers, json=collection_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if 'successful' in result and '0' in result['successful']:
                collection_key = result['successful']['0']['key']
                print(f"✅ Collection created with key: {collection_key}")
            else:
                print(f"❌ Unexpected collection response: {result}")
                return False
        else:
            print(f"❌ Collection creation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        return False
    
    # Step 2: Load and prepare articles
    with open(json_file, 'r') as f:
        articles = json.load(f)
    
    articles = [a for a in articles if a.get('title') and a.get('title') != "Issue Information"]
    print(f"📚 Preparing {len(articles)} articles for import")
    
    # Step 3: Import articles with collection assignment
    items = []
    for article in articles:
        # Clean abstract
        abstract = article.get('abstract', '')
        if abstract:
            abstract = abstract.replace('<jats:title>ABSTRACT</jats:title>', '')
            abstract = abstract.replace('<jats:p>', '').replace('</jats:p>', '')
            abstract = abstract.replace('<scp>', '').replace('</scp>', '')
            abstract = abstract.replace('<i>', '').replace('</i>', '')
            abstract = ' '.join(abstract.split())[:10000]  # Limit length
        
        # Create creators
        creators = []
        for author in article.get('authors', []):
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
        
        item = {
            "itemType": "journalArticle",
            "title": article.get('title', ''),
            "creators": creators,
            "publicationTitle": article.get('journal_name', ''),
            "volume": article.get('volume', ''),
            "issue": article.get('issue', ''),
            "date": article.get('published_date', ''),
            "DOI": article.get('doi', ''),
            "ISSN": article.get('issn', ''),
            "url": article.get('open_access_url') or article.get('url', ''),
            "abstractNote": abstract,
            "collections": [collection_key]  # Assign to collection
        }
        
        # Remove empty fields
        item = {k: v for k, v in item.items() if v}
        items.append(item)
    
    # Step 4: Import all articles at once
    print(f"🚀 Importing {len(items)} articles to collection...")
    
    try:
        response = requests.post(f"{base_url}/items", headers=headers, json=items, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            successful_count = len(result.get('successful', {}))
            failed_count = len(result.get('failed', {}))
            
            print(f"✅ Import completed!")
            print(f"   Successful: {successful_count}")
            print(f"   Failed: {failed_count}")
            
            if failed_count > 0:
                print(f"❌ Failed items: {result.get('failed', {})}")
            
            return successful_count > 0
            
        else:
            print(f"❌ Import failed: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"❌ Error during import: {e}")
        return False

def verify_collection_contents(api_key, user_id, collection_name):
    """Verify that the collection contains the expected articles."""
    
    headers = {
        'Zotero-API-Key': api_key,
        'User-Agent': 'Literature Fetcher/1.0'
    }
    
    base_url = f"https://api.zotero.org/users/{user_id}"
    
    print(f"\n🔍 Verifying collection contents...")
    
    # Get all collections
    try:
        response = requests.get(f"{base_url}/collections", headers=headers, timeout=30)
        
        if response.status_code == 200:
            collections = response.json()
            target_collection = None
            
            for collection in collections:
                if collection.get('data', {}).get('name') == collection_name:
                    target_collection = collection
                    break
            
            if not target_collection:
                print(f"❌ Collection '{collection_name}' not found")
                return False
            
            collection_key = target_collection['key']
            print(f"📁 Found collection: {collection_name} (key: {collection_key})")
            
            # Get items in collection
            response = requests.get(f"{base_url}/collections/{collection_key}/items", headers=headers, timeout=30)
            
            if response.status_code == 200:
                items = response.json()
                print(f"📚 Collection contains {len(items)} items:")
                
                for i, item in enumerate(items[:5], 1):
                    title = item.get('data', {}).get('title', 'No title')
                    print(f"   {i}. {title[:60]}...")
                
                if len(items) > 5:
                    print(f"   ... and {len(items) - 5} more")
                
                return len(items) > 0
            else:
                print(f"❌ Could not get collection items: {response.status_code}")
                return False
                
        else:
            print(f"❌ Could not get collections: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error verifying collection: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 5:
        print("Usage: python fix_collection_import.py <json_file> <api_key> <user_id> <collection_name>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    api_key = sys.argv[2]
    user_id = sys.argv[3]
    collection_name = sys.argv[4]
    
    print(f"🔧 Fixing collection import for: {collection_name}")
    print("=" * 60)
    
    # Create collection and import articles
    if create_collection_and_import(api_key, user_id, json_file, collection_name):
        print("\n✅ Import completed successfully!")
        
        # Verify the results
        time.sleep(2)  # Give Zotero a moment to update
        verify_collection_contents(api_key, user_id, collection_name)
        
    else:
        print("\n❌ Import failed")