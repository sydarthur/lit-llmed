#!/usr/bin/env python3
"""Search Zotero library more broadly."""

import json
import requests
from pathlib import Path

# Load config
config_path = Path("config/zotero.json")
with open(config_path) as f:
    config = json.load(f)

# Setup session
session = requests.Session()
session.headers.update({
    "Zotero-API-Key": config["api_key"],
    "User-Agent": "lit-llmed/0.1",
})

base_url = f"https://api.zotero.org/{config['library_type']}s/{config['user_id']}"

# Search for telemedicine
search_term = "telemedicine"

print(f"🔍 Searching for: '{search_term}'")
print()

# Search
params = {"q": search_term, "qmode": "everything", "limit": 20}
response = session.get(f"{base_url}/items", params=params, timeout=30)
response.raise_for_status()

items = response.json()

print(f"📚 Found {len(items)} item(s)")
print()

journal_articles = [item for item in items if item.get("data", {}).get("itemType") == "journalArticle"]

if journal_articles:
    print(f"Journal Articles ({len(journal_articles)}):")
    print()
    for i, item in enumerate(journal_articles, 1):
        title = item.get("data", {}).get("title", "")
        doi = item.get("data", {}).get("DOI", "")
        key = item.get("key", "")

        print(f"{i}. {title[:70]}")
        print(f"   DOI: {doi if doi else '(no DOI)'}")
        print(f"   Key: {key}")
        print()
else:
    print("No journal articles found.")
    print()
    print("Try:")
    print("  1. Import the article into Zotero first")
    print("  2. Make sure it's a 'Journal Article' item type")
    print("  3. Ensure the DOI field is populated")
