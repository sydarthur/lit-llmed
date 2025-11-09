#!/usr/bin/env python3
"""Check if a specific DOI exists in Zotero library."""

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

# DOI to search for
doi = "10.1002/joom.70021"

print(f"🔍 Searching for DOI: {doi}")
print(f"   In library: {config['library_type']} #{config['user_id']}")
print()

# Search by DOI
params = {"q": doi, "qmode": "everything", "limit": 10}
response = session.get(f"{base_url}/items", params=params, timeout=30)
response.raise_for_status()

items = response.json()

print(f"📚 Found {len(items)} item(s)")
print()

if items:
    for i, item in enumerate(items, 1):
        item_doi = item.get("data", {}).get("DOI", "")
        title = item.get("data", {}).get("title", "")
        key = item.get("key", "")
        item_type = item.get("data", {}).get("itemType", "")

        print(f"Item #{i}:")
        print(f"  Title: {title[:80]}")
        print(f"  DOI: {item_doi}")
        print(f"  Key: {key}")
        print(f"  Type: {item_type}")

        if item_doi.lower() == doi.lower():
            print(f"  ✅ EXACT MATCH!")
        print()
else:
    print("❌ No items found with this DOI")
    print()
    print("Possible reasons:")
    print("  1. Item not in your Zotero library")
    print("  2. DOI mismatch")
    print("  3. Item doesn't have DOI field populated")
    print()
    print("Try searching in Zotero desktop app for:")
    print(f"  '{doi}'")
