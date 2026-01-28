"""
SEC EDGAR Connector using EdgarTools.

Provides access to SEC filings with TTL-based caching.
"""

import json
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CompanyInfo:
    """Basic company information from SEC EDGAR."""
    cik: str
    ticker: str
    name: str
    sic: Optional[str] = None
    sic_description: Optional[str] = None
    state: Optional[str] = None
    fiscal_year_end: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Filing:
    """Represents a single SEC filing."""
    form_type: str
    filing_date: datetime
    accession_number: str
    description: Optional[str] = None
    primary_document: Optional[str] = None
    size: Optional[int] = None
    _filing_obj: Any = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "form_type": self.form_type,
            "filing_date": self.filing_date.isoformat(),
            "accession_number": self.accession_number,
            "description": self.description,
            "primary_document": self.primary_document,
            "size": self.size
        }


@dataclass
class FinancialStatements:
    """Structured financial statements from SEC filings."""
    ticker: str
    fiscal_year: Optional[int] = None
    fiscal_period: Optional[str] = None
    balance_sheet: Optional[Dict[str, Any]] = None
    income_statement: Optional[Dict[str, Any]] = None
    cash_flow: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CacheManager:
    """Manages TTL-based caching for SEC data."""

    def __init__(self, cache_dir: Path, default_ttl: int = 86400):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl

        # TTL settings by form type (in seconds)
        self.ttl_by_form = {
            "10-K": 365 * 24 * 3600,  # 1 year
            "10-Q": 90 * 24 * 3600,   # 90 days
            "8-K": 7 * 24 * 3600,     # 7 days
            "DEF 14A": 180 * 24 * 3600,  # 6 months
            "S-1": 30 * 24 * 3600,    # 30 days
            "13F-HR": 90 * 24 * 3600, # 90 days
        }

    def _get_cache_key(self, *args) -> str:
        """Generate a cache key from arguments."""
        key_str = ":".join(str(a) for a in args)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{key}.json"

    def get(self, *args, form_type: Optional[str] = None) -> Optional[Dict]:
        """
        Get cached data if valid.

        Args:
            *args: Arguments used to generate the cache key
            form_type: Optional form type for TTL lookup

        Returns:
            Cached data or None if expired/missing
        """
        key = self._get_cache_key(*args)
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)

            cached_time = datetime.fromisoformat(data.get('_cached_at', '1970-01-01'))
            ttl = self.ttl_by_form.get(form_type, self.default_ttl) if form_type else self.default_ttl

            if (datetime.now() - cached_time).total_seconds() > ttl:
                logger.debug(f"Cache expired for key: {key}")
                return None

            return data.get('data')

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Cache read error: {e}")
            return None

    def set(self, data: Any, *args) -> None:
        """
        Cache data.

        Args:
            data: Data to cache (must be JSON-serializable)
            *args: Arguments used to generate the cache key
        """
        key = self._get_cache_key(*args)
        cache_path = self._get_cache_path(key)

        cache_data = {
            '_cached_at': datetime.now().isoformat(),
            'data': data
        }

        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
        except (TypeError, IOError) as e:
            logger.warning(f"Cache write error: {e}")

    def clear(self) -> int:
        """Clear all cached data. Returns number of files removed."""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except IOError:
                pass
        return count


class SECEdgarConnector:
    """
    Connector for SEC EDGAR data using EdgarTools.

    Features:
    - Company information lookup
    - Filing retrieval with section extraction
    - Financial statement parsing
    - TTL-based caching
    """

    def __init__(
        self,
        user_agent: str = "Financial Research Agent research@example.com",
        cache_dir: Optional[Path] = None
    ):
        self.user_agent = user_agent
        self._edgar_available = False

        # Initialize cache
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "cache" / "sec"
        self.cache = CacheManager(cache_dir)

        # Try to import and configure edgartools
        try:
            from edgar import set_identity
            set_identity(user_agent)
            self._edgar_available = True
            logger.info("EdgarTools initialized successfully")
        except ImportError:
            logger.warning("EdgarTools not installed. Install with: pip install edgartools")
        except Exception as e:
            logger.warning(f"EdgarTools initialization error: {e}")

    def is_available(self) -> bool:
        """Check if SEC EDGAR access is available."""
        return self._edgar_available

    def get_company(self, ticker: str) -> Optional[CompanyInfo]:
        """
        Get basic company information.

        Args:
            ticker: Stock ticker symbol

        Returns:
            CompanyInfo or None if not found
        """
        if not self._edgar_available:
            return None

        # Check cache
        cached = self.cache.get("company", ticker.upper())
        if cached:
            return CompanyInfo(**cached)

        try:
            from edgar import Company

            company = Company(ticker.upper())

            info = CompanyInfo(
                cik=str(company.cik),
                ticker=ticker.upper(),
                name=company.name,
                sic=getattr(company, 'sic', None),
                sic_description=getattr(company, 'sic_description', None),
                state=getattr(company, 'state_of_incorporation', None),
                fiscal_year_end=getattr(company, 'fiscal_year_end', None)
            )

            # Cache the result
            self.cache.set(info.to_dict(), "company", ticker.upper())

            return info

        except Exception as e:
            logger.error(f"Error fetching company {ticker}: {e}")
            return None

    def get_filings(
        self,
        ticker: str,
        form_type: str,
        count: int = 5
    ) -> List[Filing]:
        """
        Get recent filings for a company.

        Args:
            ticker: Stock ticker symbol
            form_type: SEC form type (e.g., "10-K", "10-Q", "8-K")
            count: Number of filings to retrieve

        Returns:
            List of Filing objects
        """
        if not self._edgar_available:
            return []

        # Check cache
        cache_key = ("filings", ticker.upper(), form_type, count)
        cached = self.cache.get(*cache_key, form_type=form_type)
        if cached:
            filings = []
            for f_data in cached:
                f_data['filing_date'] = datetime.fromisoformat(f_data['filing_date'])
                filings.append(Filing(**f_data))
            return filings

        try:
            from edgar import Company

            company = Company(ticker.upper())
            raw_filings = company.get_filings(form=form_type).latest(count)

            filings = []
            for f in raw_filings:
                filing = Filing(
                    form_type=f.form,
                    filing_date=f.filing_date,
                    accession_number=f.accession_number,
                    description=getattr(f, 'description', None),
                    primary_document=getattr(f, 'primary_document', None),
                    size=getattr(f, 'size', None),
                    _filing_obj=f
                )
                filings.append(filing)

            # Cache the result
            self.cache.set([f.to_dict() for f in filings], *cache_key)

            return filings

        except Exception as e:
            logger.error(f"Error fetching filings for {ticker}: {e}")
            return []

    def get_filing_text(
        self,
        ticker: str,
        form_type: str,
        section: Optional[str] = None,
        accession_number: Optional[str] = None
    ) -> Optional[str]:
        """
        Get the text content of a filing, optionally extracting a specific section.

        Args:
            ticker: Stock ticker symbol
            form_type: SEC form type
            section: Optional section to extract (e.g., "Item 1A" for risk factors)
            accession_number: Optional specific filing accession number (if None, gets latest)

        Returns:
            Filing text or None
        """
        if not self._edgar_available:
            return None

        # Check cache - include accession number in cache key
        cache_key = ("filing_text", ticker.upper(), form_type, accession_number or "latest", section or "full")
        cached = self.cache.get(*cache_key, form_type=form_type)
        if cached:
            return cached

        try:
            from edgar import Company

            company = Company(ticker.upper())

            # Get the specific filing or latest
            if accession_number:
                # Find the specific filing by accession number
                filings = company.get_filings(form=form_type).latest(20)  # Get more to find the right one
                filing = None
                for f in filings:
                    if f.accession_number == accession_number:
                        filing = f
                        break
                if filing is None:
                    logger.warning(f"Filing with accession {accession_number} not found, using latest")
                    filing = company.get_filings(form=form_type).latest(1)[0]
            else:
                filing = company.get_filings(form=form_type).latest(1)[0]

            if section:
                # Try to extract specific section
                try:
                    # For 10-K/10-Q, try to get specific items
                    ten_k_q = filing.obj()
                    if hasattr(ten_k_q, section.lower().replace(' ', '_').replace('.', '')):
                        text = getattr(ten_k_q, section.lower().replace(' ', '_').replace('.', ''))
                    elif hasattr(ten_k_q, 'get_section'):
                        text = ten_k_q.get_section(section)
                    else:
                        # Fall back to full text and manual extraction
                        full_text = filing.text()
                        text = self._extract_section(full_text, section)
                except Exception:
                    full_text = filing.text()
                    text = self._extract_section(full_text, section)
            else:
                text = filing.text()

            # Cache the result
            if text:
                self.cache.set(text, *cache_key)

            return text

        except Exception as e:
            logger.error(f"Error fetching filing text for {ticker}: {e}")
            return None

    def _extract_section(self, text: str, section: str) -> str:
        """Extract a section from filing text using pattern matching."""
        import re

        # Common section patterns
        section_pattern = rf"(?:^|\n)({re.escape(section)}[.\s]*(?:Risk Factors|Business|Management|Financial)?[^\n]*)\n"
        next_section_pattern = r"\n(?:Item\s*\d+[A-Z]?\.)"

        match = re.search(section_pattern, text, re.IGNORECASE)
        if not match:
            return text[:50000]  # Return truncated text if section not found

        start = match.start()
        next_match = re.search(next_section_pattern, text[start + len(match.group(1)):])

        if next_match:
            end = start + len(match.group(1)) + next_match.start()
        else:
            end = min(start + 100000, len(text))

        return text[start:end]

    def get_financial_statements(
        self,
        ticker: str,
        form_type: str = "10-K"
    ) -> Optional[FinancialStatements]:
        """
        Get structured financial statements from SEC filings.

        Args:
            ticker: Stock ticker symbol
            form_type: Form type to extract from (10-K or 10-Q)

        Returns:
            FinancialStatements object or None
        """
        if not self._edgar_available:
            return None

        # Check cache
        cache_key = ("financials", ticker.upper(), form_type)
        cached = self.cache.get(*cache_key, form_type=form_type)
        if cached:
            return FinancialStatements(**cached)

        try:
            from edgar import Company

            company = Company(ticker.upper())
            filing = company.get_filings(form=form_type).latest(1)[0]

            # Try to get structured financials
            try:
                ten_k = filing.obj()

                balance_sheet = None
                income_statement = None
                cash_flow = None

                if hasattr(ten_k, 'financials'):
                    financials = ten_k.financials

                    if hasattr(financials, 'balance_sheet'):
                        bs = financials.balance_sheet
                        balance_sheet = bs.to_dict() if hasattr(bs, 'to_dict') else str(bs)

                    if hasattr(financials, 'income_statement'):
                        inc = financials.income_statement
                        income_statement = inc.to_dict() if hasattr(inc, 'to_dict') else str(inc)

                    if hasattr(financials, 'cash_flow_statement'):
                        cf = financials.cash_flow_statement
                        cash_flow = cf.to_dict() if hasattr(cf, 'to_dict') else str(cf)

                statements = FinancialStatements(
                    ticker=ticker.upper(),
                    fiscal_year=getattr(ten_k, 'fiscal_year', None),
                    fiscal_period=getattr(ten_k, 'fiscal_period', None),
                    balance_sheet=balance_sheet,
                    income_statement=income_statement,
                    cash_flow=cash_flow,
                    metadata={
                        "filing_date": filing.filing_date.isoformat(),
                        "form_type": form_type,
                        "accession_number": filing.accession_number
                    }
                )

                # Cache the result
                self.cache.set(statements.to_dict(), *cache_key)

                return statements

            except Exception as e:
                logger.warning(f"Could not parse structured financials: {e}")
                return FinancialStatements(
                    ticker=ticker.upper(),
                    metadata={
                        "error": "Could not parse structured financials",
                        "filing_date": filing.filing_date.isoformat()
                    }
                )

        except Exception as e:
            logger.error(f"Error fetching financial statements for {ticker}: {e}")
            return None

    def search_filings(
        self,
        ticker: str,
        keywords: List[str],
        form_types: Optional[List[str]] = None,
        count: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search filings for specific keywords.

        Args:
            ticker: Stock ticker symbol
            keywords: List of keywords to search for
            form_types: Optional list of form types to search
            count: Maximum number of filings to search

        Returns:
            List of matches with filing info and context
        """
        if not self._edgar_available:
            return []

        form_types = form_types or ["10-K", "10-Q", "8-K"]
        results = []

        try:
            from edgar import Company

            company = Company(ticker.upper())

            for form_type in form_types:
                filings = company.get_filings(form=form_type).latest(count)

                for filing in filings:
                    try:
                        text = filing.text()[:100000]  # Limit search scope

                        for keyword in keywords:
                            if keyword.lower() in text.lower():
                                # Find context around keyword
                                idx = text.lower().find(keyword.lower())
                                start = max(0, idx - 200)
                                end = min(len(text), idx + 200)
                                context = text[start:end]

                                results.append({
                                    "form_type": filing.form,
                                    "filing_date": filing.filing_date.isoformat(),
                                    "accession_number": filing.accession_number,
                                    "keyword": keyword,
                                    "context": f"...{context}..."
                                })
                                break  # One match per filing

                    except Exception as e:
                        logger.warning(f"Error searching filing: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error searching filings: {e}")

        return results

    def clear_cache(self) -> int:
        """Clear the cache. Returns number of items cleared."""
        return self.cache.clear()
