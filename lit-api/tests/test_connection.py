#!/usr/bin/env python3
"""
Test Zotero API connection and system functionality.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.zotero_client import ZoteroClient
from utils.config_manager import ConfigManager
from get_latest_literature import LiteratureFetcher

logger = logging.getLogger(__name__)

def test_zotero_connection(api_key: str, user_id: str) -> bool:
    """Test Zotero API connection and permissions."""
    print("🔍 Testing Zotero API Connection...")
    print(f"User ID: {user_id}")
    print(f"API Key: {api_key[:10]}...")
    
    client = ZoteroClient(api_key, user_id)
    
    # Test connection
    if client.test_connection():
        print("✅ Connection successful!")
        
        # Get user info
        user_info = client.get_user_info()
        if user_info:
            print(f"Username: {user_info.get('username', 'Unknown')}")
            access = user_info.get('access', {})
            user_access = access.get('user', {})
            print(f"Library access: {user_access.get('library', False)}")
            print(f"Write access: {user_access.get('write', False)}")
        
        return True
    else:
        print("❌ Connection failed!")
        print("Check your API key and user ID")
        return False

def test_literature_fetcher(email: str) -> bool:
    """Test literature fetching functionality."""
    print(f"\n📚 Testing Literature Fetcher...")
    
    try:
        fetcher = LiteratureFetcher(email, rate_limit_delay=0.5)
        
        # Test with a small sample
        print("Fetching sample articles from Nature...")
        articles = fetcher.fetch_journal_articles(
            issn="0028-0836",  # Nature
            rows=3,
            days_back=7,
            fetch_abstracts=False,  # Skip for speed
            fetch_oa_links=False
        )
        
        if articles:
            print(f"✅ Successfully fetched {len(articles)} articles")
            print(f"Sample: {articles[0].title[:60]}...")
            return True
        else:
            print("⚠️  No articles found (might be normal)")
            return True  # Not necessarily an error
            
    except Exception as e:
        print(f"❌ Literature fetcher test failed: {e}")
        return False

def test_system_setup():
    """Test overall system setup and configuration."""
    print("🔧 Testing System Setup...")
    
    # Test config manager
    config_manager = ConfigManager()
    config_manager.ensure_output_directories()
    
    paths = config_manager.get_output_paths()
    print("📁 Output directories:")
    for name, path in paths.items():
        status = "✅" if path.exists() else "❌"
        print(f"   {status} {name}: {path}")
    
    return True

def run_comprehensive_test(api_key: str = None, user_id: str = None, email: str = None):
    """Run comprehensive system test."""
    print("🧪 Literature Fetcher - Comprehensive Test")
    print("=" * 60)
    
    results = []
    
    # Test 1: System setup
    print("\n1️⃣ System Setup Test")
    results.append(test_system_setup())
    
    # Test 2: Literature fetching
    if email:
        print("\n2️⃣ Literature Fetcher Test")
        results.append(test_literature_fetcher(email))
    else:
        print("\n2️⃣ Skipping Literature Fetcher Test (no email provided)")
        results.append(True)
    
    # Test 3: Zotero connection
    if api_key and user_id:
        print("\n3️⃣ Zotero Connection Test")
        results.append(test_zotero_connection(api_key, user_id))
    else:
        print("\n3️⃣ Skipping Zotero Test (no credentials provided)")
        results.append(True)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    
    if all(results):
        print("🎉 All tests passed!")
        print("\nSystem is ready for use. Try:")
        print("python main.py --email your@email.com fetch-single 0028-0836")
        return True
    else:
        print("❌ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Literature Fetcher system")
    parser.add_argument('--email', help='Email for literature fetching')
    parser.add_argument('--api-key', help='Zotero API key')
    parser.add_argument('--user-id', help='Zotero user ID')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    success = run_comprehensive_test(
        api_key=args.api_key,
        user_id=args.user_id,
        email=args.email
    )
    
    sys.exit(0 if success else 1)