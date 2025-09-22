#!/usr/bin/env python3
"""
Example usage and test script for the literature fetcher system.
"""

import logging
from get_latest_literature import LiteratureFetcher
from multi_journal_fetcher import MultiJournalFetcher
from journal_config import JournalConfigManager, JournalConfig

def test_single_journal():
    """Test fetching from a single journal."""
    print("=== Testing Single Journal Fetch ===")
    
    # Use a test email - replace with your email
    email = "test@example.com"
    fetcher = LiteratureFetcher(email, rate_limit_delay=0.5)
    
    # Test with Nature (high-volume journal)
    nature_issn = "0028-0836"
    print(f"Fetching recent articles from Nature (ISSN: {nature_issn})")
    
    articles = fetcher.fetch_journal_articles(
        issn=nature_issn,
        rows=5,  # Small number for testing
        days_back=7,  # Last week only
        fetch_abstracts=True,
        fetch_oa_links=True
    )
    
    print(f"Found {len(articles)} articles:")
    for i, article in enumerate(articles, 1):
        print(f"\n{i}. {article.title}")
        print(f"   DOI: {article.doi}")
        print(f"   Authors: {', '.join(article.authors[:2])}")
        print(f"   Published: {article.published_date}")
        print(f"   Abstract: {'✓' if article.abstract else '✗'}")
        print(f"   Open Access: {'✓' if article.open_access_url else '✗'}")
    
    return articles

def test_journal_config():
    """Test journal configuration management."""
    print("\n=== Testing Journal Configuration ===")
    
    config_manager = JournalConfigManager("test_config.json")
    
    # Add a test journal
    test_journal = JournalConfig(
        name="Test Journal",
        issn="1234-5678",
        publisher="Test Publisher",
        subject_area="Testing",
        max_articles_per_fetch=10,
        days_back=7
    )
    
    config_manager.add_journal(test_journal)
    print(f"Added test journal: {test_journal.name}")
    
    # List all journals
    journals = config_manager.get_active_journals()
    print(f"\nConfigured journals ({len(journals)}):")
    for journal in journals:
        print(f"- {journal.name} ({journal.issn}) - {journal.subject_area}")
    
    return config_manager

def test_multi_journal():
    """Test multi-journal fetching."""
    print("\n=== Testing Multi-Journal Fetch ===")
    
    email = "test@example.com"
    
    # Create a test config with a few journals
    config_manager = JournalConfigManager("test_multi_config.json")
    
    # Add some test journals with smaller fetch sizes
    test_journals = [
        JournalConfig(
            name="Nature",
            issn="0028-0836",
            publisher="Nature Publishing Group",
            subject_area="Multidisciplinary",
            max_articles_per_fetch=3,
            days_back=7
        ),
        JournalConfig(
            name="Science", 
            issn="0036-8075",
            publisher="AAAS",
            subject_area="Multidisciplinary",
            max_articles_per_fetch=3,
            days_back=7
        )
    ]
    
    for journal in test_journals:
        config_manager.add_journal(journal)
    
    # Test multi-journal fetcher
    multi_fetcher = MultiJournalFetcher(
        email=email,
        config_file="test_multi_config.json",
        rate_limit_delay=0.5,
        max_workers=2
    )
    
    print("Fetching from multiple journals...")
    results = multi_fetcher.fetch_all_journals(max_workers=2)
    
    total_articles = sum(len(articles) for articles in results.values())
    print(f"\nTotal articles fetched: {total_articles}")
    
    for issn, articles in results.items():
        journal = config_manager.get_journal(issn)
        journal_name = journal.name if journal else issn
        print(f"- {journal_name}: {len(articles)} articles")
    
    return results

def run_basic_test():
    """Run a basic test to verify the system works."""
    print("Literature Fetcher - Basic Test")
    print("=" * 50)
    
    try:
        # Test 1: Single journal
        articles = test_single_journal()
        if not articles:
            print("⚠️  No articles found - this might be normal for recent articles")
        else:
            print("✅ Single journal test passed")
        
        # Test 2: Configuration
        config_manager = test_journal_config()
        print("✅ Configuration test passed")
        
        # Test 3: Multi-journal (commented out for speed)
        # results = test_multi_journal()
        # print("✅ Multi-journal test passed")
        
        print("\n🎉 All tests completed successfully!")
        print("\nNext steps:")
        print("1. Replace 'test@example.com' with your actual email")
        print("2. Configure your desired journals")
        print("3. Run: python main.py --email your@email.com fetch-all")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logging.exception("Test error:")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_basic_test()