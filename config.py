"""
Configuration module for Financial Research Agent.

Handles API keys, caching settings, and application configuration.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)


@dataclass
class CacheConfig:
    """Configuration for caching settings."""
    enabled: bool = True

    # TTL in seconds for different filing types
    ttl_10k: int = 365 * 24 * 3600   # 1 year
    ttl_10q: int = 90 * 24 * 3600    # 90 days
    ttl_8k: int = 7 * 24 * 3600      # 7 days
    ttl_default: int = 30 * 24 * 3600  # 30 days

    # Cache directory
    cache_dir: Path = CACHE_DIR

    def get_ttl_for_form(self, form_type: str) -> int:
        """Get TTL based on form type."""
        form_upper = form_type.upper()
        if '10-K' in form_upper:
            return self.ttl_10k
        elif '10-Q' in form_upper:
            return self.ttl_10q
        elif '8-K' in form_upper:
            return self.ttl_8k
        return self.ttl_default


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str = "claude"  # "claude" or "openai"

    # Claude settings
    claude_api_key: Optional[str] = None
    claude_model: str = "claude-sonnet-4-20250514"

    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4.1"

    # Common settings
    max_tokens: int = 4096
    temperature: float = 0.7

    def get_api_key(self) -> Optional[str]:
        """Get the API key for the current provider."""
        if self.provider == "claude":
            return self.claude_api_key or os.environ.get("ANTHROPIC_API_KEY")
        else:
            return self.openai_api_key or os.environ.get("OPENAI_API_KEY")

    def get_model(self) -> str:
        """Get the model name for the current provider."""
        if self.provider == "claude":
            return self.claude_model
        return self.openai_model


@dataclass
class SECConfig:
    """Configuration for SEC EDGAR access."""
    # Required: Your identity for SEC EDGAR (email)
    user_agent: str = "Financial Research Agent research@example.com"

    # Rate limiting (SEC allows 10 requests/second)
    requests_per_second: int = 8

    # Default form types to fetch
    default_form_types: List[str] = field(default_factory=lambda: [
        "10-K", "10-Q", "8-K", "DEF 14A"
    ])

    # Additional form types available
    available_form_types: List[str] = field(default_factory=lambda: [
        "10-K", "10-Q", "8-K", "DEF 14A", "S-1", "13F-HR",
        "4", "SC 13G", "SC 13D", "424B"
    ])


@dataclass
class RAGConfig:
    """Configuration for RAG (Retrieval-Augmented Generation)."""
    # Enable/disable RAG
    enabled: bool = True

    # Embedding settings
    embedding_provider: str = "sentence-transformers"  # "sentence-transformers" or "openai"
    embedding_model: str = "all-MiniLM-L6-v2"  # or "text-embedding-ada-002" for OpenAI

    # Chunking settings
    chunk_size: int = 1000  # Target chunk size in characters
    chunk_overlap: int = 200  # Overlap between chunks

    # Retrieval settings
    top_k: int = 10  # Number of chunks to retrieve
    similarity_threshold: float = 0.7  # Minimum similarity score (0-1)
    max_context_tokens: int = 32000  # Maximum tokens for context

    # Storage
    persist_directory: Path = field(default_factory=lambda: CACHE_DIR / "chromadb")


@dataclass
class AppConfig:
    """Main application configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    sec: SECConfig = field(default_factory=SECConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)

    # UI settings
    page_title: str = "Financial Research Agent"
    page_icon: str = "chart_with_upwards_trend"
    layout: str = "wide"

    # Feature flags
    enable_price_analysis: bool = True
    enable_sec_analysis: bool = True
    enable_news_search: bool = True
    enable_workflows: bool = True


def load_config() -> AppConfig:
    """Load configuration from environment and defaults."""
    config = AppConfig()

    # Load from environment
    if os.environ.get("ANTHROPIC_API_KEY"):
        config.llm.claude_api_key = os.environ["ANTHROPIC_API_KEY"]

    if os.environ.get("OPENAI_API_KEY"):
        config.llm.openai_api_key = os.environ["OPENAI_API_KEY"]

    if os.environ.get("SEC_USER_AGENT"):
        config.sec.user_agent = os.environ["SEC_USER_AGENT"]

    return config


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def update_config(**kwargs) -> AppConfig:
    """Update config with new values."""
    global _config
    config = get_config()

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.llm, key):
            setattr(config.llm, key, value)
        elif hasattr(config.cache, key):
            setattr(config.cache, key, value)
        elif hasattr(config.sec, key):
            setattr(config.sec, key, value)
        elif hasattr(config.rag, key):
            setattr(config.rag, key, value)

    return config
