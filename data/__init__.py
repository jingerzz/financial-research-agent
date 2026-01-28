"""
Data connectors and analyzers for the Financial Research Agent.
"""

from .sec_edgar import SECEdgarConnector, CompanyInfo, Filing, FinancialStatements
from .price_analyzer import StockPriceAnalyzer
from .news_search import NewsSearchManager, NewsResult
from .chunking import FilingChunk, SECFilingChunker
from .rag_manager import RAGManager, RetrievedChunk

__all__ = [
    'SECEdgarConnector',
    'CompanyInfo',
    'Filing',
    'FinancialStatements',
    'StockPriceAnalyzer',
    'NewsSearchManager',
    'NewsResult',
    'FilingChunk',
    'SECFilingChunker',
    'RAGManager',
    'RetrievedChunk',
]
