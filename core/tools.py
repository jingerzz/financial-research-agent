"""
Tool Definitions and Executor for Financial Research Agent.

Defines all available tools and handles their execution.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

TOOL_DEFINITIONS = [
    {
        "name": "get_company_info",
        "description": "Get basic company information from SEC EDGAR including CIK, name, industry (SIC code), state of incorporation, and fiscal year end.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., 'AAPL', 'MSFT')"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_sec_filing",
        "description": "Fetch SEC filings for a company. Can retrieve full filing text or specific sections like risk factors (Item 1A), business description (Item 1), or MD&A (Item 7).",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "form_type": {
                    "type": "string",
                    "description": "SEC form type (e.g., '10-K', '10-Q', '8-K', 'DEF 14A')",
                    "enum": ["10-K", "10-Q", "8-K", "DEF 14A", "S-1", "13F-HR"]
                },
                "section": {
                    "type": "string",
                    "description": "Optional specific section to extract (e.g., 'Item 1A' for risk factors, 'Item 7' for MD&A)"
                },
                "count": {
                    "type": "integer",
                    "description": "Number of recent filings to list (default: 1)",
                    "default": 1
                }
            },
            "required": ["ticker", "form_type"]
        }
    },
    {
        "name": "get_financial_statements",
        "description": "Get structured financial statements (balance sheet, income statement, cash flow) from SEC filings. Returns parsed financial data with key metrics.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "form_type": {
                    "type": "string",
                    "description": "Form type to extract from ('10-K' for annual, '10-Q' for quarterly)",
                    "enum": ["10-K", "10-Q"],
                    "default": "10-K"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "analyze_price_history",
        "description": "Analyze historical stock price data to identify key events: all-time highs/lows, largest single-day rallies and drawdowns, and statistical summary.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "period": {
                    "type": "string",
                    "description": "Historical period to analyze",
                    "enum": ["1y", "2y", "5y", "10y", "max"],
                    "default": "5y"
                },
                "num_events": {
                    "type": "integer",
                    "description": "Number of top rallies/drawdowns to identify",
                    "default": 5
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_price_around_event",
        "description": "Get stock price data around a specific date or event. Useful for examining price movements before and after significant events.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "date": {
                    "type": "string",
                    "description": "Target date in YYYY-MM-DD format"
                },
                "days_before": {
                    "type": "integer",
                    "description": "Trading days before the event to include",
                    "default": 5
                },
                "days_after": {
                    "type": "integer",
                    "description": "Trading days after the event to include",
                    "default": 5
                }
            },
            "required": ["ticker", "date"]
        }
    },
    {
        "name": "search_news",
        "description": "Search for news articles about a company around a specific date or for a general topic. Provides context for price movements and corporate events.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "date": {
                    "type": "string",
                    "description": "Target date in YYYY-MM-DD format (optional for general search)"
                },
                "event_type": {
                    "type": "string",
                    "description": "Type of event to search for (e.g., 'earnings', 'acquisition', 'guidance')"
                },
                "query": {
                    "type": "string",
                    "description": "Custom search query (alternative to event_type)"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "calculate_financial_ratios",
        "description": "Calculate key financial ratios from SEC filings: liquidity ratios (current, quick), profitability ratios (ROE, ROA, margins), and leverage ratios (debt-to-equity).",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "ratios": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific ratios to calculate (default: all). Options: 'current_ratio', 'quick_ratio', 'debt_to_equity', 'roe', 'roa', 'gross_margin', 'operating_margin', 'net_margin'"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "compare_companies",
        "description": "Compare key financial metrics across multiple companies. Useful for peer analysis and competitive comparisons.",
        "parameters": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols to compare (2-5 companies)"
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Metrics to compare. Options: 'market_cap', 'revenue', 'net_income', 'pe_ratio', 'debt_to_equity', 'roe', 'price_change_1y'"
                }
            },
            "required": ["tickers"]
        }
    },
    {
        "name": "web_search",
        "description": "Search the web for current news, articles, and information. Use this to find recent news about companies, market events, earnings reports, analyst opinions, or any other current information not available in SEC filings.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'Apple Q4 2024 earnings results', 'Tesla stock news today', 'Fed interest rate decision')"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5, max: 10)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
]


# =============================================================================
# TOOL EXECUTOR
# =============================================================================

@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None

    def to_string(self) -> str:
        """Convert result to a string for LLM consumption."""
        if self.success:
            if isinstance(self.data, dict):
                return json.dumps(self.data, indent=2, default=str)
            return str(self.data)
        return f"Error: {self.error}"


class ToolExecutor:
    """
    Executes tools and manages tool dependencies.
    """

    def __init__(
        self,
        sec_connector=None,
        news_manager=None,
        anthropic_key: Optional[str] = None,
        newsapi_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        openai_model: str = "gpt-5.2"
    ):
        self.sec_connector = sec_connector
        self.news_manager = news_manager
        self._anthropic_key = anthropic_key
        self._newsapi_key = newsapi_key
        self._openai_key = openai_key
        self._openai_model = openai_model

        # Lazy initialization
        self._price_analyzer_cache = {}

    def _get_sec_connector(self):
        """Lazy-load SEC connector."""
        if self.sec_connector is None:
            from data.sec_edgar import SECEdgarConnector
            self.sec_connector = SECEdgarConnector()
        return self.sec_connector

    def _get_news_manager(self):
        """Lazy-load news manager."""
        if self.news_manager is None:
            from data.news_search import NewsSearchManager
            self.news_manager = NewsSearchManager(
                anthropic_key=self._anthropic_key,
                newsapi_key=self._newsapi_key,
                openai_key=self._openai_key,
                openai_model=self._openai_model
            )
        return self.news_manager

    def _get_price_analyzer(self, ticker: str, period: str = "5y"):
        """Get or create price analyzer for a ticker."""
        cache_key = f"{ticker}_{period}"
        if cache_key not in self._price_analyzer_cache:
            from data.price_analyzer import StockPriceAnalyzer
            try:
                self._price_analyzer_cache[cache_key] = StockPriceAnalyzer.from_yfinance(
                    ticker, period=period
                )
            except Exception as e:
                logger.error(f"Failed to create price analyzer: {e}")
                return None
        return self._price_analyzer_cache[cache_key]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            ToolResult with execution outcome
        """
        try:
            method = getattr(self, f"_execute_{tool_name}", None)
            if method is None:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown tool: {tool_name}"
                )
            return method(arguments)
        except Exception as e:
            logger.exception(f"Tool execution error: {tool_name}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

    def _execute_get_company_info(self, args: Dict) -> ToolResult:
        """Execute get_company_info tool."""
        ticker = args.get("ticker", "").upper()
        if not ticker:
            return ToolResult(False, None, "Ticker is required")

        sec = self._get_sec_connector()
        if not sec.is_available():
            return ToolResult(False, None, "SEC EDGAR not available. Install edgartools.")

        info = sec.get_company(ticker)
        if info:
            return ToolResult(True, info.to_dict())
        return ToolResult(False, None, f"Company not found: {ticker}")

    def _execute_get_sec_filing(self, args: Dict) -> ToolResult:
        """Execute get_sec_filing tool."""
        ticker = args.get("ticker", "").upper()
        form_type = args.get("form_type", "10-K")
        section = args.get("section")
        count = args.get("count", 1)

        if not ticker:
            return ToolResult(False, None, "Ticker is required")

        sec = self._get_sec_connector()
        if not sec.is_available():
            return ToolResult(False, None, "SEC EDGAR not available")

        # Get filing list
        filings = sec.get_filings(ticker, form_type, count)
        if not filings:
            return ToolResult(False, None, f"No {form_type} filings found for {ticker}")

        # If section is requested, get the text
        if section:
            text = sec.get_filing_text(ticker, form_type, section)
            if text:
                # Truncate if too long
                if len(text) > 50000:
                    text = text[:50000] + "\n\n[Truncated - text exceeds 50,000 characters]"
                return ToolResult(True, {
                    "filings": [f.to_dict() for f in filings],
                    "section": section,
                    "text": text
                })
            return ToolResult(False, None, f"Could not extract section: {section}")

        return ToolResult(True, {
            "filings": [f.to_dict() for f in filings]
        })

    def _execute_get_financial_statements(self, args: Dict) -> ToolResult:
        """Execute get_financial_statements tool."""
        ticker = args.get("ticker", "").upper()
        form_type = args.get("form_type", "10-K")

        if not ticker:
            return ToolResult(False, None, "Ticker is required")

        sec = self._get_sec_connector()
        if not sec.is_available():
            return ToolResult(False, None, "SEC EDGAR not available")

        statements = sec.get_financial_statements(ticker, form_type)
        if statements:
            return ToolResult(True, statements.to_dict())
        return ToolResult(False, None, f"Could not retrieve financial statements for {ticker}")

    def _execute_analyze_price_history(self, args: Dict) -> ToolResult:
        """Execute analyze_price_history tool."""
        ticker = args.get("ticker", "").upper()
        period = args.get("period", "5y")
        num_events = args.get("num_events", 5)

        if not ticker:
            return ToolResult(False, None, "Ticker is required")

        analyzer = self._get_price_analyzer(ticker, period)
        if analyzer is None:
            return ToolResult(False, None, f"Could not fetch price data for {ticker}")

        try:
            results = analyzer.run_full_analysis(num_events)
            return ToolResult(True, results.to_dict())
        except Exception as e:
            return ToolResult(False, None, str(e))

    def _execute_get_price_around_event(self, args: Dict) -> ToolResult:
        """Execute get_price_around_event tool."""
        ticker = args.get("ticker", "").upper()
        date_str = args.get("date")
        days_before = args.get("days_before", 5)
        days_after = args.get("days_after", 5)

        if not ticker or not date_str:
            return ToolResult(False, None, "Ticker and date are required")

        analyzer = self._get_price_analyzer(ticker, "5y")
        if analyzer is None:
            return ToolResult(False, None, f"Could not fetch price data for {ticker}")

        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            df = analyzer.get_price_around_date(target_date, days_before, days_after)

            if df is not None and not df.empty:
                # Convert to dict format
                records = df.to_dict(orient='records')
                for r in records:
                    if 'Date' in r and hasattr(r['Date'], 'isoformat'):
                        r['Date'] = r['Date'].isoformat()
                return ToolResult(True, {
                    "ticker": ticker,
                    "target_date": date_str,
                    "data": records
                })
            return ToolResult(False, None, "No data found for the specified date range")
        except ValueError:
            return ToolResult(False, None, "Invalid date format. Use YYYY-MM-DD")

    def _execute_search_news(self, args: Dict) -> ToolResult:
        """Execute search_news tool."""
        ticker = args.get("ticker", "").upper()
        date_str = args.get("date")
        event_type = args.get("event_type", "")
        query = args.get("query", "")

        if not ticker:
            return ToolResult(False, None, "Ticker is required")

        try:
            news = self._get_news_manager()

            # Get available backends for informational purposes
            available_backends = news.get_available_backends()
            has_live_search = any(b in available_backends for b in ["Anthropic Claude", "OpenAI", "NewsAPI"])

            if date_str:
                try:
                    target_date = datetime.strptime(date_str, "%Y-%m-%d")
                    result = news.search_news_for_event(
                        ticker=ticker,
                        date=target_date,
                        event_type=event_type or "News",
                        pct_change=None
                    )

                    # Add info about search capabilities
                    result["available_backends"] = available_backends
                    result["has_live_search"] = has_live_search

                    if not has_live_search:
                        result["note"] = (
                            "Live news search not available. Only cached historical data is being used. "
                            "For live news search, configure an Anthropic API key, OpenAI API key, or NewsAPI key."
                        )

                    return ToolResult(True, result)
                except ValueError:
                    return ToolResult(False, None, "Invalid date format. Use YYYY-MM-DD")
            else:
                # General search
                search_query = query or f"{ticker} stock news"
                result = news.search_general(search_query, ticker)

                # Add info about search capabilities
                result["available_backends"] = available_backends
                result["has_live_search"] = has_live_search

                if not has_live_search:
                    result["note"] = (
                        "Live news search not available. Only cached historical data is being used. "
                        "For live news search, configure an Anthropic API key with web search or NewsAPI key."
                    )

                return ToolResult(True, result)

        except Exception as e:
            logger.error(f"News search error: {e}")
            return ToolResult(False, None, f"News search failed: {str(e)}")

    def _execute_calculate_financial_ratios(self, args: Dict) -> ToolResult:
        """Execute calculate_financial_ratios tool."""
        ticker = args.get("ticker", "").upper()
        requested_ratios = args.get("ratios", [])

        if not ticker:
            return ToolResult(False, None, "Ticker is required")

        # Try to get financial data from yfinance for quick ratios
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            info = stock.info

            ratios = {}

            # Valuation ratios
            if not requested_ratios or 'pe_ratio' in requested_ratios:
                ratios['pe_ratio'] = info.get('trailingPE')
            if not requested_ratios or 'forward_pe' in requested_ratios:
                ratios['forward_pe'] = info.get('forwardPE')
            if not requested_ratios or 'peg_ratio' in requested_ratios:
                ratios['peg_ratio'] = info.get('pegRatio')
            if not requested_ratios or 'price_to_book' in requested_ratios:
                ratios['price_to_book'] = info.get('priceToBook')

            # Profitability ratios
            if not requested_ratios or 'roe' in requested_ratios:
                ratios['roe'] = info.get('returnOnEquity')
            if not requested_ratios or 'roa' in requested_ratios:
                ratios['roa'] = info.get('returnOnAssets')
            if not requested_ratios or 'gross_margin' in requested_ratios:
                ratios['gross_margin'] = info.get('grossMargins')
            if not requested_ratios or 'operating_margin' in requested_ratios:
                ratios['operating_margin'] = info.get('operatingMargins')
            if not requested_ratios or 'net_margin' in requested_ratios:
                ratios['net_margin'] = info.get('profitMargins')

            # Liquidity ratios
            if not requested_ratios or 'current_ratio' in requested_ratios:
                ratios['current_ratio'] = info.get('currentRatio')
            if not requested_ratios or 'quick_ratio' in requested_ratios:
                ratios['quick_ratio'] = info.get('quickRatio')

            # Leverage ratios
            if not requested_ratios or 'debt_to_equity' in requested_ratios:
                ratios['debt_to_equity'] = info.get('debtToEquity')

            # Additional info
            ratios['market_cap'] = info.get('marketCap')
            ratios['enterprise_value'] = info.get('enterpriseValue')
            ratios['revenue'] = info.get('totalRevenue')
            ratios['net_income'] = info.get('netIncomeToCommon')

            # Format percentages
            for key in ['roe', 'roa', 'gross_margin', 'operating_margin', 'net_margin']:
                if ratios.get(key) is not None:
                    ratios[f'{key}_pct'] = f"{ratios[key] * 100:.2f}%"

            return ToolResult(True, {
                "ticker": ticker,
                "company_name": info.get('longName') or info.get('shortName'),
                "ratios": ratios
            })

        except Exception as e:
            return ToolResult(False, None, f"Could not calculate ratios: {e}")

    def _execute_compare_companies(self, args: Dict) -> ToolResult:
        """Execute compare_companies tool."""
        tickers = args.get("tickers", [])
        metrics = args.get("metrics", [])

        if not tickers or len(tickers) < 2:
            return ToolResult(False, None, "At least 2 tickers are required")

        if len(tickers) > 5:
            tickers = tickers[:5]

        try:
            import yfinance as yf

            comparison = {}
            default_metrics = [
                'market_cap', 'revenue', 'net_income', 'pe_ratio',
                'debt_to_equity', 'roe', 'gross_margin'
            ]
            metrics_to_use = metrics if metrics else default_metrics

            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker.upper())
                    info = stock.info

                    company_data = {
                        "name": info.get('longName') or info.get('shortName') or ticker,
                        "sector": info.get('sector'),
                        "industry": info.get('industry')
                    }

                    metric_map = {
                        'market_cap': 'marketCap',
                        'revenue': 'totalRevenue',
                        'net_income': 'netIncomeToCommon',
                        'pe_ratio': 'trailingPE',
                        'forward_pe': 'forwardPE',
                        'debt_to_equity': 'debtToEquity',
                        'roe': 'returnOnEquity',
                        'roa': 'returnOnAssets',
                        'gross_margin': 'grossMargins',
                        'operating_margin': 'operatingMargins',
                        'net_margin': 'profitMargins',
                        'current_ratio': 'currentRatio',
                        'price_to_book': 'priceToBook',
                        'dividend_yield': 'dividendYield'
                    }

                    for metric in metrics_to_use:
                        yf_key = metric_map.get(metric, metric)
                        value = info.get(yf_key)
                        company_data[metric] = value

                    # Get 1-year price change
                    if 'price_change_1y' in metrics_to_use or not metrics:
                        try:
                            hist = stock.history(period="1y")
                            if len(hist) > 0:
                                start_price = hist['Close'].iloc[0]
                                end_price = hist['Close'].iloc[-1]
                                company_data['price_change_1y'] = (
                                    (end_price - start_price) / start_price * 100
                                )
                        except Exception:
                            pass

                    comparison[ticker.upper()] = company_data

                except Exception as e:
                    logger.warning(f"Error fetching data for {ticker}: {e}")
                    comparison[ticker.upper()] = {"error": str(e)}

            return ToolResult(True, {
                "comparison": comparison,
                "metrics": metrics_to_use
            })

        except ImportError:
            return ToolResult(False, None, "yfinance not installed")

    def _execute_web_search(self, args: Dict) -> ToolResult:
        """Execute web_search tool using DuckDuckGo."""
        query = args.get("query", "")
        max_results = min(args.get("max_results", 5), 10)

        if not query:
            return ToolResult(False, None, "Search query is required")

        try:
            news = self._get_news_manager()
            results = news.web_search(query, max_results=max_results)

            if results:
                # Format results for display
                formatted_results = []
                for r in results:
                    formatted_results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("snippet", ""),
                        "url": r.get("url", ""),
                        "source": r.get("source", "")
                    })

                return ToolResult(True, {
                    "query": query,
                    "num_results": len(formatted_results),
                    "results": formatted_results
                })
            else:
                return ToolResult(True, {
                    "query": query,
                    "num_results": 0,
                    "results": [],
                    "note": "No results found. Try a different query."
                })

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return ToolResult(False, None, f"Web search failed: {str(e)}")

    def get_tool_definitions(self) -> List[Dict]:
        """Get all tool definitions."""
        return TOOL_DEFINITIONS
