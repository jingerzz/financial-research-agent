"""
News Search Module for Financial Research Agent.

Provides news context for significant stock price events.
Supports multiple backends: Anthropic Claude, NewsAPI, and cached responses.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class NewsResult:
    """Represents a news search result."""
    headline: str
    summary: str
    source: str
    url: Optional[str] = None
    date: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "headline": self.headline,
            "summary": self.summary,
            "source": self.source,
            "url": self.url,
            "date": self.date.isoformat() if self.date else None
        }


class NewsSearchBackend(ABC):
    """Abstract base class for news search backends."""

    @abstractmethod
    def search(self, query: str, date: datetime, ticker: str) -> List[NewsResult]:
        """Search for news around a specific date."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the backend is properly configured."""
        pass


class AnthropicSearchBackend(NewsSearchBackend):
    """Uses Anthropic's Claude API with web search capability."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = None

        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.warning("anthropic package not installed")

    def is_configured(self) -> bool:
        return self.client is not None

    def search(self, query: str, date: datetime, ticker: str) -> List[NewsResult]:
        if not self.is_configured():
            return []

        try:
            date_str = date.strftime("%B %d, %Y")

            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search"
                }],
                messages=[{
                    "role": "user",
                    "content": f"""Search for news about {ticker} stock around {date_str}.
                    Query: {query}

                    Provide a brief summary of what happened and why the stock moved significantly.
                    Include the main news headline, a 2-3 sentence summary, and the source."""
                }]
            )

            response_text = ""
            for block in message.content:
                if hasattr(block, 'text'):
                    response_text += block.text

            return [NewsResult(
                headline=f"{ticker} News - {date_str}",
                summary=response_text,
                source="Claude Web Search",
                date=date
            )]

        except Exception as e:
            logger.error(f"Anthropic search error: {e}")
            return []


class OpenAISearchBackend(NewsSearchBackend):
    """Uses OpenAI's chat API to provide context about historical events."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5.2"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.client = None

        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("openai package not installed")

    def is_configured(self) -> bool:
        return self.client is not None

    def search(self, query: str, date: datetime, ticker: str) -> List[NewsResult]:
        if not self.is_configured():
            return []

        try:
            date_str = date.strftime("%B %d, %Y")

            # Use OpenAI chat to provide context (not live web search)
            response = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": f"""Based on your knowledge, what significant news or events affected {ticker} stock around {date_str}?
                    Context: {query}

                    Provide a brief summary including:
                    - Main headline or event
                    - 2-3 sentence explanation
                    - Note if this is from your training data (not live search)"""
                }]
            )

            response_text = response.choices[0].message.content if response.choices else ""

            return [NewsResult(
                headline=f"{ticker} News Context - {date_str}",
                summary=response_text,
                source=f"OpenAI {self.model} (training data, not live search)",
                date=date
            )]

        except Exception as e:
            logger.error(f"OpenAI search error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []


class DuckDuckGoSearchBackend(NewsSearchBackend):
    """Uses DuckDuckGo for free web search (no API key required)."""

    def __init__(self):
        self._ddgs = None

    def _get_ddgs(self):
        """Lazy load DuckDuckGo search."""
        if self._ddgs is None:
            try:
                # Try new package name first (ddgs)
                from ddgs import DDGS
                self._ddgs = DDGS()
            except ImportError:
                try:
                    # Fall back to old package name
                    from duckduckgo_search import DDGS
                    self._ddgs = DDGS()
                except ImportError:
                    logger.warning("ddgs or duckduckgo-search package not installed. Run: pip install ddgs")
                    return None
        return self._ddgs

    def is_configured(self) -> bool:
        return self._get_ddgs() is not None

    def search(self, query: str, date: datetime, ticker: str) -> List[NewsResult]:
        ddgs = self._get_ddgs()
        if not ddgs:
            return []

        try:
            # Search for news articles
            results = []

            # Try news search first
            try:
                news_results = list(ddgs.news(
                    query,
                    max_results=5,
                    timelimit="y"  # Past year
                ))

                for article in news_results[:5]:
                    pub_date = None
                    if article.get("date"):
                        try:
                            pub_date = datetime.fromisoformat(article["date"].replace("Z", "+00:00"))
                        except (ValueError, AttributeError):
                            pass

                    results.append(NewsResult(
                        headline=article.get("title", ""),
                        summary=article.get("body", ""),
                        source=article.get("source", "DuckDuckGo News"),
                        url=article.get("url"),
                        date=pub_date
                    ))
            except Exception as e:
                logger.debug(f"DuckDuckGo news search failed, trying text search: {e}")

            # If no news results, try general text search
            if not results:
                text_results = list(ddgs.text(
                    query,
                    max_results=5
                ))

                for item in text_results[:5]:
                    results.append(NewsResult(
                        headline=item.get("title", ""),
                        summary=item.get("body", ""),
                        source=item.get("href", "").split("/")[2] if item.get("href") else "Web",
                        url=item.get("href"),
                        date=date
                    ))

            return results

        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    def search_with_date_range(
        self,
        query: str,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        max_results: int = 5
    ) -> List[NewsResult]:
        """Search for news within a specific date range."""
        ddgs = self._get_ddgs()
        if not ddgs:
            return []

        try:
            # Format date range for search query
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Add date range to query
            date_query = f"{query} {start_str}..{end_str}"

            results = []
            text_results = list(ddgs.text(
                date_query,
                max_results=max_results
            ))

            for item in text_results:
                results.append(NewsResult(
                    headline=item.get("title", ""),
                    summary=item.get("body", ""),
                    source=item.get("href", "").split("/")[2] if item.get("href") else "Web",
                    url=item.get("href"),
                    date=start_date
                ))

            return results

        except Exception as e:
            logger.error(f"DuckDuckGo date range search error: {e}")
            return []


class NewsAPIBackend(NewsSearchBackend):
    """Uses NewsAPI.org for historical news search."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("NEWSAPI_KEY")
        self.base_url = "https://newsapi.org/v2/everything"

    def is_configured(self) -> bool:
        return self.api_key is not None

    def search(self, query: str, date: datetime, ticker: str) -> List[NewsResult]:
        if not self.is_configured():
            return []

        try:
            import requests

            from_date = (date - timedelta(days=3)).strftime("%Y-%m-%d")
            to_date = (date + timedelta(days=3)).strftime("%Y-%m-%d")

            params = {
                "q": query,
                "from": from_date,
                "to": to_date,
                "sortBy": "relevancy",
                "pageSize": 5,
                "apiKey": self.api_key
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()

            results = []
            for article in data.get("articles", [])[:3]:
                pub_date = None
                if article.get("publishedAt"):
                    try:
                        pub_date = datetime.fromisoformat(
                            article["publishedAt"].replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass

                results.append(NewsResult(
                    headline=article.get("title", ""),
                    summary=article.get("description", ""),
                    source=article.get("source", {}).get("name", "Unknown"),
                    url=article.get("url"),
                    date=pub_date
                ))

            return results

        except Exception as e:
            logger.error(f"NewsAPI search error: {e}")
            return []


class CachedNewsBackend(NewsSearchBackend):
    """Uses pre-cached responses for common market events."""

    def __init__(self):
        # Pre-loaded knowledge about major market events
        self.market_events = {
            (1998, 8): {
                "event": "Russian Financial Crisis",
                "summary": "Russia defaulted on domestic debt and devalued the ruble on August 17, 1998. This triggered a global flight to quality and the near-collapse of Long-Term Capital Management (LTCM) hedge fund.",
                "source": "Historical Market Data"
            },
            (2008, 9): {
                "event": "Lehman Brothers Collapse",
                "summary": "Lehman Brothers filed for bankruptcy on September 15, 2008, triggering the worst phase of the global financial crisis. Credit markets froze, and the government began massive intervention programs.",
                "source": "Historical Market Data"
            },
            (2008, 10): {
                "event": "2008 Financial Crisis Peak",
                "summary": "October 2008 saw some of the most volatile trading in market history. The Dow experienced multiple 500+ point swings. Congress passed TARP, and the Fed implemented emergency lending facilities.",
                "source": "Historical Market Data"
            },
            (2020, 3): {
                "event": "COVID-19 Crash",
                "summary": "March 2020 saw the fastest bear market in history. The S&P 500 fell 34% from peak to trough. March 16 was the worst day since 1987, with the Dow falling 2,997 points (-12.9%).",
                "source": "Historical Market Data"
            },
            (2022, 6): {
                "event": "2022 Bear Market",
                "summary": "Markets officially entered bear market territory in June 2022 as the Fed aggressively raised rates to combat inflation. The S&P 500 fell more than 20% from its January peak.",
                "source": "Historical Market Data"
            },
            (2023, 3): {
                "event": "Banking Crisis",
                "summary": "Silicon Valley Bank and Signature Bank collapsed in March 2023, sparking fears of a broader banking crisis. The Fed and FDIC intervened to protect depositors and stabilize the banking system.",
                "source": "Historical Market Data"
            },
        }

    def is_configured(self) -> bool:
        return True

    def search(self, query: str, date: datetime, ticker: str) -> List[NewsResult]:
        results = []
        year = date.year
        month = date.month

        # Check for general market events
        if (year, month) in self.market_events:
            event = self.market_events[(year, month)]
            results.append(NewsResult(
                headline=event["event"],
                summary=event["summary"],
                source=event["source"],
                date=date
            ))

        if not results:
            results.append(NewsResult(
                headline=f"Market Event - {date.strftime('%B %Y')}",
                summary=f"No specific cached news available for {ticker} on {date.strftime('%Y-%m-%d')}. Enable Anthropic API or NewsAPI for live search.",
                source="System",
                date=date
            ))

        return results


class NewsSearchManager:
    """Manages multiple news search backends with fallback logic."""

    def __init__(
        self,
        anthropic_key: Optional[str] = None,
        newsapi_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        openai_model: str = "gpt-5.2"
    ):
        self.backends: List[tuple] = []

        # Add DuckDuckGo first (free, no API key required)
        ddg_backend = DuckDuckGoSearchBackend()
        if ddg_backend.is_configured():
            self.backends.append(("DuckDuckGo", ddg_backend))

        # Add other backends in priority order
        anthropic_backend = AnthropicSearchBackend(anthropic_key)
        if anthropic_backend.is_configured():
            self.backends.append(("Anthropic Claude", anthropic_backend))

        openai_backend = OpenAISearchBackend(openai_key, model=openai_model)
        if openai_backend.is_configured():
            self.backends.append(("OpenAI", openai_backend))

        newsapi_backend = NewsAPIBackend(newsapi_key)
        if newsapi_backend.is_configured():
            self.backends.append(("NewsAPI", newsapi_backend))

        # Always add cached backend as fallback
        self.backends.append(("Cached", CachedNewsBackend()))

        # Store DuckDuckGo backend for direct access
        self.ddg_backend = ddg_backend

    def get_available_backends(self) -> List[str]:
        """Return list of configured backend names."""
        return [name for name, _ in self.backends]

    def search_news_for_event(
        self,
        ticker: str,
        date: datetime,
        event_type: str,
        pct_change: Optional[float] = None,
        preferred_backend: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for news context for a specific stock event.

        Args:
            ticker: Stock ticker symbol
            date: Date of the event
            event_type: Type of event (e.g., "All-Time High", "Top Rally")
            pct_change: Percentage change if applicable
            preferred_backend: Specific backend to use (or None for auto)

        Returns:
            Dictionary with news results and metadata
        """
        date_str = date.strftime("%B %d, %Y")

        # Build search query based on event type
        if event_type == "All-Time High":
            query = f"{ticker} stock all-time high record {date_str}"
        elif event_type == "All-Time Low":
            query = f"{ticker} stock low {date_str}"
        elif "Rally" in event_type and pct_change:
            query = f"{ticker} stock surge rally {pct_change:.0f}% {date_str}"
        elif "Drawdown" in event_type and pct_change:
            query = f"{ticker} stock drop decline crash {abs(pct_change):.0f}% {date_str}"
        else:
            query = f"{ticker} stock news {date_str}"

        # Select backends to try
        backends_to_try = self.backends
        if preferred_backend:
            backends_to_try = [
                (n, b) for n, b in self.backends if n == preferred_backend
            ]
            if not backends_to_try:
                backends_to_try = self.backends

        # Try backends in order
        for backend_name, backend in backends_to_try:
            results = backend.search(query, date, ticker)
            if results and results[0].summary and "No specific cached news" not in results[0].summary:
                return {
                    "success": True,
                    "backend": backend_name,
                    "query": query,
                    "results": [r.to_dict() for r in results],
                    "date": date.isoformat(),
                    "event_type": event_type
                }

        # Return cached fallback
        cached_backend = CachedNewsBackend()
        results = cached_backend.search(query, date, ticker)
        return {
            "success": len(results) > 0,
            "backend": "Cached",
            "query": query,
            "results": [r.to_dict() for r in results],
            "date": date.isoformat(),
            "event_type": event_type
        }

    def search_general(
        self,
        query: str,
        ticker: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform a general news search (not tied to a specific date).

        Args:
            query: Search query
            ticker: Optional ticker for context

        Returns:
            Dictionary with search results
        """
        now = datetime.now()

        for backend_name, backend in self.backends:
            # Skip cached backend for general searches
            if backend_name == "Cached":
                continue

            results = backend.search(query, now, ticker or "")
            if results:
                return {
                    "success": True,
                    "backend": backend_name,
                    "query": query,
                    "results": [r.to_dict() for r in results]
                }

        return {
            "success": False,
            "backend": None,
            "query": query,
            "results": [],
            "error": "No configured backends available for general search"
        }

    def search_for_price_events(
        self,
        ticker: str,
        rallies: List[Any],
        drawdowns: List[Any],
        max_events: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for news context around top rallies and drawdowns.

        Args:
            ticker: Stock ticker symbol
            rallies: List of rally events (need .date and .pct_change attributes)
            drawdowns: List of drawdown events
            max_events: Maximum number of each event type to search

        Returns:
            Dictionary with rally_news and drawdown_news lists
        """
        results = {
            "rally_news": [],
            "drawdown_news": []
        }

        # Search for rally news
        for rally in rallies[:max_events]:
            date = rally.date
            pct = rally.pct_change

            date_str = date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date)
            query = f"{ticker} stock surge rally news {date_str}"

            search_result = self.search_news_for_event(
                ticker=ticker,
                date=date if isinstance(date, datetime) else datetime.strptime(str(date)[:10], "%Y-%m-%d"),
                event_type="Rally",
                pct_change=pct
            )

            if search_result.get("results"):
                results["rally_news"].append({
                    "date": date_str,
                    "pct_change": pct,
                    "news": search_result["results"]
                })

        # Search for drawdown news
        for dd in drawdowns[:max_events]:
            date = dd.date
            pct = dd.pct_change

            date_str = date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date)
            query = f"{ticker} stock drop decline news {date_str}"

            search_result = self.search_news_for_event(
                ticker=ticker,
                date=date if isinstance(date, datetime) else datetime.strptime(str(date)[:10], "%Y-%m-%d"),
                event_type="Drawdown",
                pct_change=pct
            )

            if search_result.get("results"):
                results["drawdown_news"].append({
                    "date": date_str,
                    "pct_change": pct,
                    "news": search_result["results"]
                })

        return results

    def web_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a simple web search using DuckDuckGo.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search results with title, snippet, url
        """
        if not self.ddg_backend or not self.ddg_backend.is_configured():
            return []

        try:
            ddgs = self.ddg_backend._get_ddgs()
            if not ddgs:
                return []

            results = []
            text_results = list(ddgs.text(query, max_results=max_results))

            for item in text_results:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("body", ""),
                    "url": item.get("href", ""),
                    "source": item.get("href", "").split("/")[2] if item.get("href") and "/" in item.get("href", "") else "Web"
                })

            return results

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []

    def format_search_results_for_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results into a string suitable for LLM context.

        Args:
            results: List of search results

        Returns:
            Formatted string with search results
        """
        if not results:
            return ""

        lines = ["**Web Search Results:**\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title") or r.get("headline", "")
            snippet = r.get("snippet") or r.get("summary", "")
            url = r.get("url", "")
            source = r.get("source", "")

            lines.append(f"{i}. **{title}**")
            if snippet:
                lines.append(f"   {snippet[:300]}{'...' if len(snippet) > 300 else ''}")
            if url:
                lines.append(f"   Source: {source} | URL: {url}")
            lines.append("")

        return "\n".join(lines)
