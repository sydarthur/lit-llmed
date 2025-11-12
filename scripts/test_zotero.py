#!/usr/bin/env python3
"""Quick script to test Zotero connection."""

import json
from pathlib import Path
from src.integrate.zotero import ZoteroClient
from src.core.log import get_logger

LOGGER = get_logger(__name__)


def test_zotero_connection():
    """Test Zotero API connection and display library info."""

    # Load config
    config_path = Path("config/zotero.json")
    if not config_path.exists():
        print("❌ config/zotero.json not found")
        return False

    with open(config_path) as f:
        config = json.load(f)

    # Check for placeholder values
    if config.get("api_key") == "YOUR_ZOTERO_API_KEY":
        print("❌ Please update config/zotero.json with your actual Zotero credentials")
        print("\n📝 To get your credentials:")
        print("   1. API Key: https://www.zotero.org/settings/keys")
        print("   2. User ID: Shown on the same page")
        print("\n   Then update config/zotero.json with:")
        print('   {"api_key": "your-actual-key", "user_id": "12345678", "library_type": "user"}')
        return False

    # Initialize client
    try:
        client = ZoteroClient(**config)
        print(f"🔍 Testing connection to Zotero...")
        print(f"   Library: {config['library_type']} #{config['user_id']}")

        # Test connection
        if client.test_connection():
            print("✅ Connection successful!")

            # Try to get collections
            try:
                response = client.session.get(f"{client.base_url}/collections", timeout=10)
                if response.status_code == 200:
                    collections = response.json()
                    print(f"\n📚 Found {len(collections)} collection(s) in your library:")
                    for coll in collections[:5]:  # Show first 5
                        name = coll.get("data", {}).get("name", "Unknown")
                        key = coll.get("key", "")
                        print(f"   - {name} ({key})")
                    if len(collections) > 5:
                        print(f"   ... and {len(collections) - 5} more")
                else:
                    print(f"\n⚠️  Could not fetch collections (status: {response.status_code})")
            except Exception as e:
                print(f"\n⚠️  Could not fetch collections: {e}")

            return True
        else:
            print("❌ Connection failed!")
            print("   Check your API key and user ID in config/zotero.json")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    test_zotero_connection()
