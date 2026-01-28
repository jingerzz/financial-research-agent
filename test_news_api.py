#!/usr/bin/env python3
"""
Test script for News API functionality in Financial Research Agent.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.news_search import NewsSearchManager, NewsAPIBackend, AnthropicSearchBackend, CachedNewsBackend

def test_news_backends():
    """Test each news backend."""
    print("=" * 60)
    print("Testing News API Backends")
    print("=" * 60)
    
    # Test NewsAPI backend
    print("\n1. Testing NewsAPI Backend:")
    print("-" * 60)
    newsapi_key = input("Enter NewsAPI key (or press Enter to skip): ").strip()
    
    if newsapi_key:
        newsapi = NewsAPIBackend(api_key=newsapi_key)
        print(f"   Configured: {newsapi.is_configured()}")
        
        if newsapi.is_configured():
            test_date = datetime.now() - timedelta(days=7)
            test_query = "Apple stock"
            print(f"   Testing search for '{test_query}' around {test_date.strftime('%Y-%m-%d')}...")
            
            try:
                results = newsapi.search(test_query, test_date, "AAPL")
                print(f"   Results: {len(results)} articles found")
                for i, result in enumerate(results[:3], 1):
                    print(f"   {i}. {result.headline}")
                    print(f"      Source: {result.source}")
                    print(f"      Summary: {result.summary[:100]}...")
            except Exception as e:
                print(f"   Error: {e}")
    else:
        print("   Skipped (no API key provided)")
    
    # Test Anthropic backend
    print("\n2. Testing Anthropic Claude Backend:")
    print("-" * 60)
    anthropic_key = input("Enter Anthropic API key (or press Enter to skip): ").strip()
    
    if anthropic_key:
        anthropic = AnthropicSearchBackend(api_key=anthropic_key)
        print(f"   Configured: {anthropic.is_configured()}")
        
        if anthropic.is_configured():
            test_date = datetime.now() - timedelta(days=7)
            test_query = "Apple stock news"
            print(f"   Testing search for '{test_query}' around {test_date.strftime('%Y-%m-%d')}...")
            
            try:
                results = anthropic.search(test_query, test_date, "AAPL")
                print(f"   Results: {len(results)} articles found")
                for i, result in enumerate(results, 1):
                    print(f"   {i}. {result.headline}")
                    print(f"      Source: {result.source}")
                    print(f"      Summary: {result.summary[:200]}...")
            except Exception as e:
                print(f"   Error: {e}")
    else:
        print("   Skipped (no API key provided)")
    
    # Test Cached backend
    print("\n3. Testing Cached Backend:")
    print("-" * 60)
    cached = CachedNewsBackend()
    print(f"   Configured: {cached.is_configured()}")
    
    test_date = datetime(2020, 3, 16)  # COVID crash date
    test_query = "market crash"
    print(f"   Testing search for '{test_query}' on {test_date.strftime('%Y-%m-%d')}...")
    
    results = cached.search(test_query, test_date, "SPY")
    print(f"   Results: {len(results)} articles found")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result.headline}")
        print(f"      Source: {result.source}")
        print(f"      Summary: {result.summary[:200]}...")
    
    # Test NewsSearchManager
    print("\n4. Testing NewsSearchManager (with all backends):")
    print("-" * 60)
    
    manager = NewsSearchManager(
        anthropic_key=anthropic_key if anthropic_key else None,
        newsapi_key=newsapi_key if newsapi_key else None
    )
    
    available = manager.get_available_backends()
    print(f"   Available backends: {', '.join(available)}")
    
    # Test event-based search
    test_date = datetime(2020, 3, 16)
    print(f"\n   Testing event search for AAPL on {test_date.strftime('%Y-%m-%d')}...")
    
    result = manager.search_news_for_event(
        ticker="AAPL",
        date=test_date,
        event_type="Market Crash",
        pct_change=-12.0
    )
    
    print(f"   Success: {result['success']}")
    print(f"   Backend used: {result['backend']}")
    print(f"   Results: {len(result['results'])} articles")
    
    for i, article in enumerate(result['results'][:2], 1):
        print(f"   {i}. {article.get('headline', 'N/A')}")
        print(f"      Source: {article.get('source', 'N/A')}")
    
    # Test general search
    print(f"\n   Testing general search for 'Apple earnings'...")
    general_result = manager.search_general("Apple earnings", ticker="AAPL")
    
    print(f"   Success: {general_result['success']}")
    if general_result['success']:
        print(f"   Backend used: {general_result['backend']}")
        print(f"   Results: {len(general_result['results'])} articles")
    else:
        print(f"   Error: {general_result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_news_backends()
