"""
Sidebar Configuration UI for Financial Research Agent.

Provides configuration options for LLM providers, SEC filings, and data sources.
"""

import os
import streamlit as st
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from config import get_config

logger = logging.getLogger(__name__)


def get_rag_manager():
    """
    Get or create the RAG manager singleton from session state.

    Returns:
        RAGManager instance or None if RAG is disabled
    """
    config = get_config()

    if not config.rag.enabled:
        return None

    if "rag_manager" not in st.session_state:
        try:
            from data.rag_manager import RAGManager

            st.session_state.rag_manager = RAGManager(
                persist_directory=config.rag.persist_directory,
                embedding_provider=config.rag.embedding_provider,
                embedding_model=config.rag.embedding_model,
            )
            logger.info("Initialized RAG manager")
        except ImportError as e:
            logger.warning(f"RAG dependencies not installed: {e}")
            st.session_state.rag_manager = None
        except Exception as e:
            logger.error(f"Error initializing RAG manager: {e}")
            st.session_state.rag_manager = None

    return st.session_state.rag_manager


def get_rag_context(query: str, accession_numbers: Optional[List[str]] = None) -> tuple[str, int]:
    """
    Retrieve relevant context using RAG for a query.

    Args:
        query: User query to search for
        accession_numbers: Optional list of accession numbers to filter by

    Returns:
        Tuple of (context string, number of chunks retrieved)
    """
    config = get_config()
    rag_manager = get_rag_manager()

    if not rag_manager or not config.rag.enabled:
        # Fall back to full context
        return get_loaded_filings_context(), 0

    # Get accession numbers from loaded filings if not provided
    if accession_numbers is None:
        accession_numbers = []
        if "loaded_filings" in st.session_state:
            for lf in st.session_state.loaded_filings:
                accession_numbers.append(lf.filing_info.accession_number)

    if not accession_numbers:
        return "", 0

    # Retrieve relevant chunks
    chunks = rag_manager.retrieve(
        query=query,
        top_k=config.rag.top_k,
        accession_numbers=accession_numbers,
        similarity_threshold=config.rag.similarity_threshold
    )

    if not chunks:
        logger.info(f"No relevant chunks found for query: {query[:50]}...")
        return "", 0

    # Format context
    context = rag_manager.format_context(
        chunks=chunks,
        max_tokens=config.rag.max_context_tokens
    )

    logger.info(f"RAG retrieved {len(chunks)} chunks ({len(context)} chars) for query")
    return context, len(chunks)


def get_credential_mgr():
    """Get or create the credential manager singleton."""
    if "credential_manager" not in st.session_state:
        try:
            from core.credential_manager import get_credential_manager
            st.session_state.credential_manager = get_credential_manager()
        except ImportError as e:
            logger.warning(f"Credential manager not available: {e}")
            st.session_state.credential_manager = None
        except Exception as e:
            logger.error(f"Error initializing credential manager: {e}")
            st.session_state.credential_manager = None
    return st.session_state.credential_manager


def get_api_key(provider: str) -> tuple[Optional[str], str]:
    """
    Get API key from various sources in priority order.

    Priority:
    1. Environment variable
    2. Streamlit secrets
    3. OS Keyring (via credential manager)
    4. Encrypted local file (via credential manager)
    5. Session state (temporary)

    Returns:
        Tuple of (api_key, source) where source indicates where the key came from.
    """
    # Map UI provider names to credential manager provider names
    provider_map = {
        "openai": "openai",
        "claude": "anthropic",
        "gemini": "google"
    }
    
    cred_provider = provider_map.get(provider, provider)
    
    # Try credential manager first (handles env vars, secrets, keyring, encrypted file)
    cred_mgr = get_credential_mgr()
    if cred_mgr:
        key, source = cred_mgr.get_key(cred_provider)
        if key:
            return key, source
    else:
        # Fallback if credential manager not available
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY"
        }
        secret_key_map = {
            "openai": "openai_api_key",
            "claude": "anthropic_api_key",
            "gemini": "google_api_key"
        }

        env_var = env_var_map.get(provider, "ANTHROPIC_API_KEY")
        secret_key = secret_key_map.get(provider, "anthropic_api_key")

        # Check environment variable
        env_key = os.environ.get(env_var)
        if env_key:
            return env_key, "environment"

        # Check Streamlit secrets
        try:
            if secret_key in st.secrets:
                return st.secrets[secret_key], "secrets"
        except Exception:
            pass

    # Last resort: Check session state (temporary, lost on refresh)
    session_key = f"{provider}_api_key"
    if session_key in st.session_state and st.session_state[session_key]:
        return st.session_state[session_key], "session"

    return None, "none"


def store_api_key(provider: str, key: str) -> tuple[bool, str]:
    """
    Store an API key securely.
    
    Args:
        provider: UI provider name (openai, claude, gemini)
        key: The API key to store
        
    Returns:
        Tuple of (success, message)
    """
    provider_map = {
        "openai": "openai",
        "claude": "anthropic",
        "gemini": "google"
    }
    
    cred_provider = provider_map.get(provider, provider)
    cred_mgr = get_credential_mgr()
    
    if cred_mgr:
        success, result = cred_mgr.store_key(cred_provider, key)
        if success:
            return True, f"Saved to {result}"
        return False, result
    
    # Fallback to session state
    session_key = f"{provider}_api_key"
    st.session_state[session_key] = key
    return True, "session (temporary)"


def delete_api_key(provider: str) -> tuple[bool, str]:
    """
    Delete a stored API key.
    
    Args:
        provider: UI provider name (openai, claude, gemini)
        
    Returns:
        Tuple of (success, message)
    """
    provider_map = {
        "openai": "openai",
        "claude": "anthropic",
        "gemini": "google"
    }
    
    cred_provider = provider_map.get(provider, provider)
    cred_mgr = get_credential_mgr()
    
    deleted_from = []
    
    if cred_mgr:
        success, msg = cred_mgr.delete_key(cred_provider)
        if success:
            deleted_from.append(msg)
    
    # Also clear session state
    session_key = f"{provider}_api_key"
    if session_key in st.session_state:
        del st.session_state[session_key]
        deleted_from.append("session")
    
    if deleted_from:
        return True, f"Deleted from: {', '.join(deleted_from)}"
    return False, "No key found to delete"


@dataclass
class FilingInfo:
    """Information about a single SEC filing."""
    form_type: str
    filing_date: str
    accession_number: str
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[int] = None
    description: Optional[str] = None

    @property
    def display_name(self) -> str:
        """Generate display name for the filing."""
        if self.form_type in ["10-K", "10-K/A"]:
            year = self.fiscal_year or self.filing_date[:4]
            return f"{self.form_type} {year}"
        elif self.form_type in ["10-Q", "10-Q/A"]:
            year = self.fiscal_year or self.filing_date[:4]
            quarter = f"Q{self.fiscal_quarter}" if self.fiscal_quarter else ""
            return f"{self.form_type} {year} {quarter}".strip()
        else:
            return f"{self.form_type} ({self.filing_date})"


@dataclass
class LoadedFiling:
    """A filing that has been loaded into context."""
    ticker: str
    filing_info: FilingInfo
    content: Optional[str] = None
    sections: Optional[Dict[str, str]] = None
    loaded_at: Optional[datetime] = None


@dataclass
class SidebarConfig:
    """Configuration collected from sidebar inputs."""
    # LLM settings
    llm_provider: str = "openai"
    api_key: Optional[str] = None
    model: Optional[str] = None

    # Company settings
    ticker: str = ""
    company_name: Optional[str] = None

    # SEC filing settings
    form_types: List[str] = field(default_factory=lambda: ["10-K", "10-Q", "8-K"])
    selected_form_type: str = "10-K"

    # Feature toggles
    enable_sec: bool = True
    enable_price: bool = True
    enable_news: bool = True

    # News API settings
    newsapi_key: Optional[str] = None

    # Loaded filings
    loaded_filings: List[LoadedFiling] = field(default_factory=list)

    def is_configured(self) -> bool:
        """Check if minimum configuration is provided."""
        return bool(self.api_key)


def fetch_available_filings(ticker: str, form_type: str = None) -> List[FilingInfo]:
    """
    Fetch available filings for a ticker from SEC EDGAR.

    Args:
        ticker: Stock ticker symbol
        form_type: Optional form type filter

    Returns:
        List of FilingInfo objects
    """
    try:
        from data.sec_edgar import SECEdgarConnector

        connector = SECEdgarConnector()
        if not connector.is_available():
            logger.warning("SEC EDGAR connector not available")
            return []

        filings = []
        # Include all supported form types from the plan: 10-K, 10-Q, 8-K, DEF 14A, S-1, 13F-HR
        form_types_to_fetch = [form_type] if form_type else ["10-K", "10-Q", "8-K", "DEF 14A", "S-1", "13F-HR"]

        for ft in form_types_to_fetch:
            try:
                raw_filings = connector.get_filings(ticker, ft, count=10)
                for f in raw_filings:
                    # Try to extract fiscal year/quarter from filing date
                    filing_date_str = f.filing_date.strftime("%Y-%m-%d") if hasattr(f.filing_date, 'strftime') else str(f.filing_date)

                    fiscal_year = None
                    fiscal_quarter = None

                    if hasattr(f.filing_date, 'year'):
                        fiscal_year = f.filing_date.year
                        if ft in ["10-Q", "10-Q/A"]:
                            month = f.filing_date.month
                            if month <= 3:
                                fiscal_quarter = 4
                                fiscal_year -= 1
                            elif month <= 6:
                                fiscal_quarter = 1
                            elif month <= 9:
                                fiscal_quarter = 2
                            else:
                                fiscal_quarter = 3

                    filings.append(FilingInfo(
                        form_type=f.form_type,
                        filing_date=filing_date_str,
                        accession_number=f.accession_number,
                        fiscal_year=fiscal_year,
                        fiscal_quarter=fiscal_quarter,
                        description=f.description
                    ))
            except Exception as e:
                logger.warning(f"Error fetching {ft} filings: {e}")
                continue

        return filings

    except Exception as e:
        logger.error(f"Error fetching filings for {ticker}: {e}")
        return []


def load_filing_content(ticker: str, filing_info: FilingInfo) -> Optional[LoadedFiling]:
    """
    Load the content of a specific filing.

    Args:
        ticker: Stock ticker symbol
        filing_info: Filing to load

    Returns:
        LoadedFiling with content, or None on failure
    """
    try:
        from data.sec_edgar import SECEdgarConnector

        connector = SECEdgarConnector()
        if not connector.is_available():
            logger.warning("SEC EDGAR connector not available")
            return None

        accession = filing_info.accession_number
        logger.info(f"Loading filing {filing_info.form_type} with accession {accession} for {ticker}")

        # Get filing text using the specific accession number
        text = connector.get_filing_text(
            ticker,
            filing_info.form_type,
            accession_number=accession
        )

        if text:
            logger.info(f"Got filing text: {len(text)} chars")
            # Extract key sections for 10-K and 10-Q - no truncation for full document analysis
            sections = {}
            if filing_info.form_type in ["10-K", "10-K/A"]:
                for section in ["Item 1", "Item 1A", "Item 7", "Item 8"]:
                    section_text = connector.get_filing_text(
                        ticker,
                        filing_info.form_type,
                        section=section,
                        accession_number=accession
                    )
                    if section_text and len(section_text) > 100:
                        sections[section] = section_text  # Full section, no truncation
            elif filing_info.form_type in ["10-Q", "10-Q/A"]:
                for section in ["Item 1", "Item 2", "Item 3"]:
                    section_text = connector.get_filing_text(
                        ticker,
                        filing_info.form_type,
                        section=section,
                        accession_number=accession
                    )
                    if section_text and len(section_text) > 100:
                        sections[section] = section_text  # Full section, no truncation

            loaded_filing = LoadedFiling(
                ticker=ticker,
                filing_info=filing_info,
                content=text,  # Full content, no truncation
                sections=sections if sections else None,
                loaded_at=datetime.now()
            )

            # Index into RAG if enabled
            _index_filing_to_rag(loaded_filing)

            return loaded_filing

        logger.warning(f"No text returned for filing {accession}")
        return None

    except Exception as e:
        logger.error(f"Error loading filing content: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def _index_filing_to_rag(loaded_filing: LoadedFiling) -> None:
    """
    Index a loaded filing into the RAG system.

    Args:
        loaded_filing: The filing to index
    """
    rag_manager = get_rag_manager()
    if not rag_manager:
        return

    try:
        chunks_indexed = rag_manager.index_filing(
            content=loaded_filing.content or "",
            ticker=loaded_filing.ticker,
            form_type=loaded_filing.filing_info.form_type,
            filing_date=loaded_filing.filing_info.filing_date,
            accession_number=loaded_filing.filing_info.accession_number,
            sections=loaded_filing.sections,
            skip_if_exists=True
        )
        if chunks_indexed > 0:
            logger.info(f"Indexed {chunks_indexed} chunks for {loaded_filing.ticker} {loaded_filing.filing_info.display_name}")
    except Exception as e:
        logger.error(f"Error indexing filing to RAG: {e}")


def render_sidebar() -> SidebarConfig:
    """
    Render the sidebar configuration interface.

    Returns:
        SidebarConfig with user selections
    """
    config = SidebarConfig()

    # Initialize session state for filings
    if "available_filings" not in st.session_state:
        st.session_state.available_filings = []
    if "loaded_filings" not in st.session_state:
        st.session_state.loaded_filings = []
    if "ticker_loaded" not in st.session_state:
        st.session_state.ticker_loaded = ""
    if "selected_filing_keys" not in st.session_state:
        st.session_state.selected_filing_keys = set()

    with st.sidebar:
        st.header("Configuration")

        # LLM Provider Selection
        st.subheader("LLM Provider")

        config.llm_provider = st.selectbox(
            "Provider",
            options=["openai", "claude", "gemini"],
            format_func=lambda x: {"openai": "OpenAI", "claude": "Anthropic", "gemini": "Google Gemini"}[x],
            help="Select the LLM provider for analysis"
        )

        # API Key input with multiple source support
        existing_key, key_source = get_api_key(config.llm_provider)
        session_key = f"{config.llm_provider}_api_key"

        if config.llm_provider == "claude":
            provider_name = "Anthropic"
            env_var_name = "ANTHROPIC_API_KEY"
            help_url = "console.anthropic.com"
            model_options = ["claude-sonnet-4-20250514", "claude-opus-4-20250514"]
            model_format = lambda x: x.split("-")[1].title()
        elif config.llm_provider == "gemini":
            provider_name = "Google"
            env_var_name = "GOOGLE_API_KEY"
            help_url = "aistudio.google.com/apikey"
            # Gemini 3 models
            model_options = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"]
            model_format = lambda x: x.replace("gemini-", "Gemini ").replace("-", " ").title()
        else:
            provider_name = "OpenAI"
            env_var_name = "OPENAI_API_KEY"
            help_url = "platform.openai.com"
            # User's available OpenAI models
            model_options = ["gpt-5.2", "gpt-4.1", "gpt-5-mini"]
            model_format = lambda x: x

        # Show key status based on source
        if key_source == "environment":
            st.success(f"✓ Using {env_var_name} from environment")
            config.api_key = existing_key
        elif key_source == "secrets":
            st.success(f"✓ Using API key from secrets.toml")
            config.api_key = existing_key
        elif key_source in ("keyring", "encrypted_file"):
            # Securely stored key
            storage_name = "system keychain" if key_source == "keyring" else "secure storage"
            st.success(f"✓ Key saved in {storage_name}")
            config.api_key = existing_key
            
            # Show preview and delete option
            cred_mgr = get_credential_mgr()
            if cred_mgr:
                key_preview = f"{existing_key[:4]}...{existing_key[-4:]}" if len(existing_key) > 8 else "****"
                st.caption(f"Key: {key_preview}")
            
            if st.button("🗑️ Delete saved key", key=f"delete_{config.llm_provider}"):
                success, msg = delete_api_key(config.llm_provider)
                if success:
                    st.success(f"Key deleted")
                    st.rerun()
                else:
                    st.error(f"Failed to delete: {msg}")
        elif key_source == "session":
            st.info("✓ Using key (session only - will be lost on refresh)")
            config.api_key = existing_key
            st.caption("💡 Enter key below and click Save to persist")
            
            if st.button("Clear session key", key=f"clear_{config.llm_provider}"):
                del st.session_state[session_key]
                st.rerun()
        else:
            # No key found - show input
            st.warning("⚠️ No API key configured")
        
        # Always show input for adding/updating key (except when from env/secrets)
        if key_source not in ("environment", "secrets"):
            with st.expander("Configure API Key" if existing_key else "Enter API Key", expanded=not existing_key):
                api_key_input = st.text_input(
                    f"{provider_name} API Key",
                    type="password",
                    help=f"Get from {help_url}",
                    key=f"key_input_{config.llm_provider}"
                )
                
                col_save, col_test = st.columns(2)
                
                with col_save:
                    if st.button("💾 Save Key", key=f"save_{config.llm_provider}", 
                                disabled=not api_key_input, use_container_width=True):
                        if api_key_input:
                            success, msg = store_api_key(config.llm_provider, api_key_input)
                            if success:
                                st.success(f"✓ Saved to {msg}")
                                st.rerun()
                            else:
                                st.error(f"Failed: {msg}")
                
                with col_test:
                    if st.button("🧪 Test Key", key=f"test_{config.llm_provider}",
                                disabled=not api_key_input, use_container_width=True):
                        if api_key_input:
                            cred_mgr = get_credential_mgr()
                            if cred_mgr:
                                # Map provider name
                                provider_map = {"openai": "openai", "claude": "anthropic", "gemini": "google"}
                                with st.spinner("Testing..."):
                                    success, msg = cred_mgr.test_key(provider_map.get(config.llm_provider), api_key_input)
                                if success:
                                    st.success(f"✓ {msg}")
                                else:
                                    st.error(f"✗ {msg}")
                
                # Use entered key for this session even if not saved
                if api_key_input:
                    config.api_key = api_key_input

        # Model selection
        config.model = st.selectbox(
            "Model",
            options=model_options,
            format_func=model_format
        )

        st.divider()

        # Company Settings with Load button
        st.subheader("Company")

        col1, col2 = st.columns([3, 1])

        with col1:
            ticker_input = st.text_input(
                "Ticker Symbol",
                placeholder="e.g., AAPL, MSFT",
                help="Enter stock ticker for research"
            ).upper().strip()

        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            load_ticker = st.button("Load", type="primary", use_container_width=True)

        config.ticker = ticker_input

        # Load ticker and fetch filings
        if load_ticker and ticker_input:
            with st.spinner(f"Loading {ticker_input}..."):
                filings = fetch_available_filings(ticker_input)
                st.session_state.available_filings = filings
                st.session_state.ticker_loaded = ticker_input
                st.session_state.selected_filing_keys = set()

                if filings:
                    st.success(f"Found {len(filings)} filings for {ticker_input}")
                else:
                    st.warning(f"No filings found for {ticker_input}. Check ticker or install edgartools.")

        # Show ticker status
        if st.session_state.ticker_loaded:
            if st.session_state.ticker_loaded == ticker_input:
                st.caption(f"✓ {ticker_input} loaded")
            else:
                st.caption(f"⚠ Click Load to fetch {ticker_input}")

        st.divider()

        # SEC Filing Selection
        st.subheader("SEC Filings")

        config.enable_sec = st.checkbox(
            "Enable SEC Filing Analysis",
            value=True,
            help="Fetch and analyze SEC EDGAR filings"
        )

        if config.enable_sec and st.session_state.available_filings:
            # Form type filter
            available_form_types = list(set(f.form_type for f in st.session_state.available_filings))
            available_form_types.sort()

            config.selected_form_type = st.selectbox(
                "Form Type",
                options=available_form_types,
                help="Filter filings by form type"
            )

            # Filter filings by selected form type
            filtered_filings = [
                f for f in st.session_state.available_filings
                if f.form_type == config.selected_form_type
            ]

            if filtered_filings:
                st.write("**Select Filings to Load:**")

                # Display checkboxes for each filing
                for filing in filtered_filings[:8]:  # Limit to 8 most recent
                    filing_key = f"{filing.form_type}_{filing.accession_number}"

                    # Create checkbox
                    is_selected = st.checkbox(
                        f"{filing.display_name} (Filed: {filing.filing_date})",
                        key=f"filing_{filing_key}",
                        value=filing_key in st.session_state.selected_filing_keys
                    )

                    if is_selected:
                        st.session_state.selected_filing_keys.add(filing_key)
                    elif filing_key in st.session_state.selected_filing_keys:
                        st.session_state.selected_filing_keys.discard(filing_key)

                # Load Selected Filings button
                selected_count = len(st.session_state.selected_filing_keys)

                if selected_count > 0:
                    if st.button(
                        f"Load {selected_count} Selected Filing{'s' if selected_count > 1 else ''}",
                        type="primary",
                        use_container_width=True
                    ):
                        with st.spinner("Loading filings..."):
                            loaded_count = 0
                            failed_count = 0
                            skipped_count = 0

                            for filing in st.session_state.available_filings:
                                filing_key = f"{filing.form_type}_{filing.accession_number}"
                                if filing_key in st.session_state.selected_filing_keys:
                                    # Check if already loaded
                                    already_loaded = any(
                                        lf.filing_info.accession_number == filing.accession_number
                                        for lf in st.session_state.loaded_filings
                                    )
                                    if already_loaded:
                                        skipped_count += 1
                                        continue

                                    try:
                                        loaded = load_filing_content(st.session_state.ticker_loaded, filing)
                                        if loaded:
                                            st.session_state.loaded_filings.append(loaded)
                                            loaded_count += 1
                                        else:
                                            failed_count += 1
                                            logger.warning(f"Failed to load filing: {filing.display_name}")
                                    except Exception as e:
                                        failed_count += 1
                                        logger.error(f"Exception loading filing {filing.display_name}: {e}")

                            # Show results
                            if loaded_count > 0:
                                st.success(f"Loaded {loaded_count} filing(s)")
                            if skipped_count > 0:
                                st.info(f"{skipped_count} filing(s) already loaded")
                            if failed_count > 0:
                                st.warning(f"{failed_count} filing(s) failed to load. Check if edgartools is installed.")

                        st.rerun()
            else:
                st.info(f"No {config.selected_form_type} filings available")

        elif config.enable_sec and not st.session_state.available_filings:
            st.info("Enter a ticker and click Load to see available filings")

            with st.expander("Form Type Reference"):
                st.markdown("""
                - **10-K**: Annual report with full financials
                - **10-Q**: Quarterly report
                - **8-K**: Material events
                - **DEF 14A**: Proxy statement
                - **S-1**: IPO registration
                - **13F-HR**: Institutional holdings
                """)

        st.divider()

        # Loaded Context Section
        st.subheader("Loaded Context")

        if st.session_state.loaded_filings:
            for i, lf in enumerate(st.session_state.loaded_filings):
                col1, col2 = st.columns([4, 1])
                with col1:
                    sections_info = f" ({len(lf.sections)} sections)" if lf.sections else ""
                    st.caption(f"• {lf.ticker} {lf.filing_info.display_name}{sections_info}")
                with col2:
                    if st.button("✕", key=f"remove_{i}", help="Remove from context"):
                        st.session_state.loaded_filings.pop(i)
                        st.rerun()

            if st.button("Clear All", use_container_width=True):
                st.session_state.loaded_filings = []
                st.session_state.selected_filing_keys = set()
                st.rerun()
        else:
            st.caption("No filings loaded yet")

        # Store loaded filings in config
        config.loaded_filings = st.session_state.loaded_filings

        st.divider()

        # Data Source Settings
        st.subheader("Data Sources")

        config.enable_price = st.checkbox(
            "Enable Price Analysis",
            value=True,
            help="Analyze historical stock prices"
        )

        config.enable_news = st.checkbox(
            "Enable News Search",
            value=True,
            help="Search for relevant news articles"
        )

        if config.enable_news:
            with st.expander("News API Settings"):
                config.newsapi_key = st.text_input(
                    "NewsAPI Key (optional)",
                    type="password",
                    help="Get from newsapi.org for enhanced news search"
                )

        st.divider()

        # Status
        st.subheader("Status")

        if config.is_configured():
            st.success("API key configured")
            if config.ticker:
                st.info(f"Ready to research: {config.ticker}")
        else:
            st.warning("Enter API key to start")

        # Help
        with st.expander("Help"):
            st.markdown("""
            ### Getting Started

            1. **Configure your API key** (see below)
            2. **Enter a ticker symbol** and click **Load**
            3. **Select filings** to load into context
            4. Use the **Research Chat** tab to ask questions
            5. Use **Workflows** for automated analysis

            ### API Key Configuration

            Your API keys can be stored securely:

            **1. Environment Variable** (best for servers)
            ```bash
            export OPENAI_API_KEY="sk-..."
            export ANTHROPIC_API_KEY="sk-ant-..."
            export GOOGLE_API_KEY="..."
            ```

            **2. Streamlit Secrets** (good for development)
            Create `.streamlit/secrets.toml`:
            ```toml
            openai_api_key = "sk-..."
            anthropic_api_key = "sk-ant-..."
            google_api_key = "..."
            ```

            **3. Secure Storage** (recommended for desktop)
            Click **Save Key** above to store in:
            - macOS: Keychain
            - Windows: Credential Manager
            - Linux: Secret Service

            Keys persist across browser refreshes!

            ### Example Questions

            - "What are the main risk factors in the 2024 10-K?"
            - "Compare risk factors between 2024 and 2023"
            - "Summarize the business description"
            - "What changed in revenue year-over-year?"
            """)

    return config


def get_loaded_filings_context() -> str:
    """
    Get a formatted string of loaded filings for use as context in chat.

    Returns:
        Formatted string with filing information (full content, no truncation)
    """
    if "loaded_filings" not in st.session_state or not st.session_state.loaded_filings:
        return ""

    context_parts = []

    for lf in st.session_state.loaded_filings:
        context_parts.append(f"\n{'='*60}")
        context_parts.append(f"FILING: {lf.ticker} {lf.filing_info.display_name}")
        context_parts.append(f"Filed: {lf.filing_info.filing_date}")
        context_parts.append(f"{'='*60}\n")

        if lf.sections:
            for section_name, section_content in lf.sections.items():
                context_parts.append(f"\n--- {section_name} ---\n")
                context_parts.append(section_content)  # Full section content
        elif lf.content:
            context_parts.append(lf.content)  # Full content

    result = "\n".join(context_parts)
    logger.info(f"Generated filing context: {len(result)} chars from {len(st.session_state.loaded_filings)} filing(s)")
    return result


def render_quick_actions(config: SidebarConfig) -> Optional[str]:
    """
    Render quick action buttons in the sidebar.

    Returns the selected action prompt, or None.
    """
    if not config.ticker:
        return None

    with st.sidebar:
        st.subheader("Quick Actions")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Latest Financials", use_container_width=True):
                return f"Get the latest financial statements for {config.ticker} and summarize key metrics."

            if st.button("Risk Factors", use_container_width=True):
                return f"Extract and summarize the main risk factors from {config.ticker}'s latest 10-K filing."

        with col2:
            if st.button("Price Events", use_container_width=True):
                return f"Analyze {config.ticker}'s price history and identify the most significant events."

            if st.button("Company Overview", use_container_width=True):
                return f"Provide a comprehensive overview of {config.ticker} including business description, financials, and recent developments."

    return None
