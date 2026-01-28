#!/usr/bin/env python3
"""
Simple test for News API functionality - checks configuration and basic functionality.
"""

import sys
import os
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
    
    # Check environment variables
    newsapi_key = os.environ.get("NEWSAPI_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    # Test NewsAPI backend
    print("\n1. Testing NewsAPI Backend:")
    print("-" * 60)
    if newsapi_key:
        newsapi = NewsAPIBackend(api_key=newsapi_key)
        print(f"   ✓ API Key found in environment")
        print(f"   ✓ Configured: {newsapi.is_configured()}")
        
        if newsapi.is_configured():
            test_date = datetime.now() - timedelta(days=7)
            test_query = "Apple stock"
            print(f"   Testing search for '{test_query}' around {test_date.strftime('%Y-%m-%d')}...")
            
            try:
                results = newsapi.search(test_query, test_date, "AAPL")
                print(f"   ✓ Results: {len(results)} articles found")
                if results:
                    for i, result in enumerate(results[:2], 1):
                        print(f"      {i}. {result.headline[:60]}...")
                        print(f"         Source: {result.source}")
                else:
                    print("      ⚠ No results returned (may be rate limited or no articles found)")
            except Exception as e:
                print(f"   ✗ Error: {e}")
        else:
            print("   ✗ Backend not properly configured")
    else:
        print("   ⚠ No NEWSAPI_KEY found in environment")
        print("   ℹ NewsAPI backend will not be available")
    
    # Test Anthropic backend
    print("\n2. Testing Anthropic Claude Backend:")
    print("-" * 60)
    if anthropic_key:
        anthropic = AnthropicSearchBackend(api_key=anthropic_key)
        print(f"   ✓ API Key found in environment")
        print(f"   ✓ Configured: {anthropic.is_configured()}")
        
        if anthropic.is_configured():
            test_date = datetime.now() - timedelta(days=7)
            test_query = "Apple stock news"
            print(f"   Testing search for '{test_query}' around {test_date.strftime('%Y-%m-%d')}...")
            
            try:
                results = anthropic.search(test_query, test_date, "AAPL")
                print(f"   ✓ Results: {len(results)} articles found")
                if results:
                    for i, result in enumerate(results, 1):
                        print(f"      {i}. {result.headline[:60]}...")
                        print(f"         Source: {result.source}")
            except Exception as e:
                print(f"   ✗ Error: {e}")
        else:
            print("   ✗ Backend not properly configured")
    else:
        print("   ⚠ No ANTHROPIC_API_KEY found in environment")
        print("   ℹ Anthropic backend will not be available")
    
    # Test Cached backend (always available)
    print("\n3. Testing Cached Backend:")
    print("-" * 60)
    cached = CachedNewsBackend()
    print(f"   ✓ Configured: {cached.is_configured()}")
    
    test_date = datetime(2020, 3, 16)  # COVID crash date
    test_query = "market crash"
    print(f"   Testing search for '{test_query}' on {test_date.strftime('%Y-%m-%d')}...")
    
    results = cached.search(test_query, test_date, "SPY")
    print(f"   ✓ Results: {len(results)} articles found")
    for i, result in enumerate(results, 1):
        print(f"      {i}. {result.headline}")
        print(f"         Source: {result.source}")
    
    # Test NewsSearchManager
    print("\n4. Testing NewsSearchManager:")
    print("-" * 60)
    
    manager = NewsSearchManager(
        anthropic_key=anthropic_key,
        newsapi_key=newsapi_key
    )
    
    available = manager.get_available_backends()
    print(f"   Available backends: {', '.join(available)}")
    
    # Test event-based search
    test_date = datetime(2020, 3, 16)
    print(f"\n   Testing event search for AAPL on {test_date.strftime('%Y-%m-%d')}...")
    
    try:
        result = manager.search_news_for_event(
            ticker="AAPL",
            date=test_date,
            event_type="Market Crash",
            pct_change=-12.0
        )
        
        print(f"   ✓ Success: {result['success']}")
        print(f"   ✓ Backend used: {result['backend']}")
        print(f"   ✓ Results: {len(result['results'])} articles")
        
        for i, article in enumerate(result['results'][:2], 1):
            headline = article.get('headline', 'N/A')[:60]
            print(f"      {i}. {headline}...")
            print(f"         Source: {article.get('source', 'N/A')}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test general search
    print(f"\n   Testing general search for 'Apple earnings'...")
    try:
        general_result = manager.search_general("Apple earnings", ticker="AAPL")
        
        print(f"   ✓ Success: {general_result['success']}")
        if general_result['success']:
            print(f"   ✓ Backend used: {general_result['backend']}")
            print(f"   ✓ Results: {len(general_result['results'])} articles")
            for i, article in enumerate(general_result['results'][:2], 1):
                headline = article.get('headline', 'N/A')[:60]
                print(f"      {i}. {headline}...")
        else:
            print(f"   ⚠ No results: {general_result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"NewsAPI: {'✓ Available' if newsapi_key and NewsAPIBackend(api_key=newsapi_key).is_configured() else '✗ Not configured'}")
    print(f"Anthropic: {'✓ Available' if anthropic_key and AnthropicSearchBackend(api_key=anthropic_key).is_configured() else '✗ Not configured'}")
    print(f"Cached: ✓ Always available")
    print("=" * 60)

if __name__ == "__main__":
    test_news_backends()
