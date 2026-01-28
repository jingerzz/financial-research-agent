"""
RAG Manager for Financial Research Agent.

Provides vector storage and retrieval using ChromaDB for semantic search over SEC filings.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

from .chunking import FilingChunk, SECFilingChunker

logger = logging.getLogger(__name__)

# Lazy imports to avoid loading heavy dependencies until needed
_chromadb = None
_SentenceTransformer = None


def _get_chromadb():
    """Lazily import chromadb."""
    global _chromadb
    if _chromadb is None:
        import chromadb
        _chromadb = chromadb
    return _chromadb


def _get_sentence_transformer():
    """Lazily import sentence_transformers."""
    global _SentenceTransformer
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer
    return _SentenceTransformer


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store."""
    chunk_id: str
    content: str
    metadata: Dict[str, str]
    similarity_score: float

    @property
    def ticker(self) -> Optional[str]:
        return self.metadata.get("ticker")

    @property
    def form_type(self) -> Optional[str]:
        return self.metadata.get("form_type")

    @property
    def filing_date(self) -> Optional[str]:
        return self.metadata.get("filing_date")

    @property
    def accession_number(self) -> Optional[str]:
        return self.metadata.get("accession_number")

    @property
    def section_name(self) -> Optional[str]:
        return self.metadata.get("section_name")


class RAGManager:
    """
    Manages RAG operations for SEC filings.

    Features:
    - Index filings into ChromaDB with metadata
    - Semantic search with metadata filtering
    - Support for sentence-transformers (local) or OpenAI embeddings
    """

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        embedding_provider: str = "sentence-transformers",
        embedding_model: str = "all-MiniLM-L6-v2",
        openai_api_key: Optional[str] = None,
        collection_name: str = "sec_filings"
    ):
        """
        Initialize the RAG manager.

        Args:
            persist_directory: Directory for ChromaDB persistence. If None, uses in-memory.
            embedding_provider: "sentence-transformers" or "openai"
            embedding_model: Model name for embeddings
            openai_api_key: Required if using OpenAI embeddings
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = persist_directory
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.openai_api_key = openai_api_key
        self.collection_name = collection_name

        self._client = None
        self._collection = None
        self._embedding_fn = None
        self._chunker = SECFilingChunker()

        # Track indexed filings
        self._indexed_accessions: Set[str] = set()

    def _ensure_initialized(self) -> None:
        """Ensure ChromaDB client and collection are initialized."""
        if self._client is not None:
            return

        chromadb = _get_chromadb()

        # Create client
        if self.persist_directory:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_directory))
            logger.info(f"Initialized persistent ChromaDB at {self.persist_directory}")
        else:
            self._client = chromadb.Client()
            logger.info("Initialized in-memory ChromaDB")

        # Create embedding function
        self._embedding_fn = self._create_embedding_function()

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        # Load existing accession numbers
        self._load_indexed_accessions()

    def _create_embedding_function(self):
        """Create the embedding function based on provider."""
        chromadb = _get_chromadb()

        if self.embedding_provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key required for OpenAI embeddings")

            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
            return OpenAIEmbeddingFunction(
                api_key=self.openai_api_key,
                model_name=self.embedding_model
            )
        else:
            # Default to sentence-transformers
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            return SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )

    def _load_indexed_accessions(self) -> None:
        """Load set of already-indexed accession numbers."""
        try:
            results = self._collection.get(
                include=["metadatas"]
            )
            if results and results.get("metadatas"):
                for metadata in results["metadatas"]:
                    if metadata and "accession_number" in metadata:
                        self._indexed_accessions.add(metadata["accession_number"])
            logger.info(f"Loaded {len(self._indexed_accessions)} indexed filings")
        except Exception as e:
            logger.warning(f"Error loading indexed accessions: {e}")

    def is_filing_indexed(self, accession_number: str) -> bool:
        """Check if a filing is already indexed."""
        self._ensure_initialized()
        return accession_number in self._indexed_accessions

    def index_filing(
        self,
        content: str,
        ticker: str,
        form_type: str,
        filing_date: str,
        accession_number: str,
        sections: Optional[Dict[str, str]] = None,
        skip_if_exists: bool = True
    ) -> int:
        """
        Index a filing into ChromaDB.

        Args:
            content: Full filing text
            ticker: Stock ticker symbol
            form_type: Form type (10-K, 10-Q, etc.)
            filing_date: Filing date string
            accession_number: SEC accession number
            sections: Pre-extracted sections (if available)
            skip_if_exists: Skip if already indexed

        Returns:
            Number of chunks indexed
        """
        self._ensure_initialized()

        if skip_if_exists and self.is_filing_indexed(accession_number):
            logger.info(f"Filing {accession_number} already indexed, skipping")
            return 0

        # Chunk the filing
        chunks = self._chunker.chunk_filing(
            content=content,
            ticker=ticker,
            form_type=form_type,
            filing_date=filing_date,
            accession_number=accession_number,
            sections=sections
        )

        if not chunks:
            logger.warning(f"No chunks generated for {accession_number}")
            return 0

        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Add to collection
        try:
            self._collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            self._indexed_accessions.add(accession_number)
            logger.info(f"Indexed {len(chunks)} chunks for {ticker} {form_type} ({accession_number})")
            return len(chunks)
        except Exception as e:
            logger.error(f"Error indexing filing {accession_number}: {e}")
            return 0

    def index_chunks(self, chunks: List[FilingChunk]) -> int:
        """
        Index pre-chunked filing chunks.

        Args:
            chunks: List of FilingChunk objects

        Returns:
            Number of chunks indexed
        """
        self._ensure_initialized()

        if not chunks:
            return 0

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        try:
            self._collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

            # Track accession numbers
            for chunk in chunks:
                if chunk.accession_number:
                    self._indexed_accessions.add(chunk.accession_number)

            return len(chunks)
        except Exception as e:
            logger.error(f"Error indexing chunks: {e}")
            return 0

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        accession_numbers: Optional[List[str]] = None,
        tickers: Optional[List[str]] = None,
        form_types: Optional[List[str]] = None,
        similarity_threshold: float = 0.0
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query
            top_k: Maximum number of chunks to retrieve
            accession_numbers: Filter to specific filings
            tickers: Filter to specific tickers
            form_types: Filter to specific form types
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of RetrievedChunk objects, sorted by relevance
        """
        self._ensure_initialized()

        if not query:
            return []

        # Build where clause for filtering
        where = None
        where_clauses = []

        if accession_numbers:
            where_clauses.append({
                "accession_number": {"$in": accession_numbers}
            })

        if tickers:
            where_clauses.append({
                "ticker": {"$in": tickers}
            })

        if form_types:
            where_clauses.append({
                "form_type": {"$in": form_types}
            })

        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        # Query collection
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return []

        # Convert results to RetrievedChunk objects
        chunks = []
        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results.get("documents") else [None] * len(ids)
            metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
            distances = results["distances"][0] if results.get("distances") else [1.0] * len(ids)

            for i, chunk_id in enumerate(ids):
                # Convert distance to similarity (ChromaDB uses cosine distance)
                similarity = 1 - distances[i]

                if similarity >= similarity_threshold:
                    chunks.append(RetrievedChunk(
                        chunk_id=chunk_id,
                        content=documents[i] or "",
                        metadata=metadatas[i] or {},
                        similarity_score=similarity
                    ))

        # Sort by similarity (highest first)
        chunks.sort(key=lambda x: x.similarity_score, reverse=True)

        logger.info(f"Retrieved {len(chunks)} chunks for query: {query[:50]}...")
        return chunks

    def format_context(
        self,
        chunks: List[RetrievedChunk],
        max_tokens: int = 32000,
        include_metadata: bool = True
    ) -> str:
        """
        Format retrieved chunks for LLM context.

        Args:
            chunks: List of retrieved chunks
            max_tokens: Maximum context size (approximate, based on chars)
            include_metadata: Include chunk metadata in context

        Returns:
            Formatted context string
        """
        if not chunks:
            return ""

        # Rough estimate: 4 chars per token
        max_chars = max_tokens * 4

        context_parts = []
        current_chars = 0

        # Group chunks by filing
        chunks_by_filing = {}
        for chunk in chunks:
            key = (chunk.ticker, chunk.form_type, chunk.filing_date)
            if key not in chunks_by_filing:
                chunks_by_filing[key] = []
            chunks_by_filing[key].append(chunk)

        for (ticker, form_type, filing_date), filing_chunks in chunks_by_filing.items():
            if current_chars >= max_chars:
                break

            # Filing header - cleaner format
            header = f"\n---\n**{ticker} {form_type}** (Filed: {filing_date})\n---\n"

            if current_chars + len(header) > max_chars:
                break

            context_parts.append(header)
            current_chars += len(header)

            # Add chunks
            for chunk in filing_chunks:
                if current_chars >= max_chars:
                    break

                chunk_text = ""
                if include_metadata and chunk.section_name:
                    chunk_text += f"\n**{chunk.section_name}:**\n"

                # Clean the content - normalize whitespace
                content = chunk.content
                # Replace multiple newlines with double newline
                import re
                content = re.sub(r'\n{3,}', '\n\n', content)
                # Replace multiple spaces with single space
                content = re.sub(r' {2,}', ' ', content)

                chunk_text += content
                chunk_text += "\n"

                if current_chars + len(chunk_text) > max_chars:
                    # Truncate to fit
                    remaining = max_chars - current_chars - 50  # Leave room for truncation note
                    if remaining > 100:
                        chunk_text = chunk_text[:remaining] + "\n[...truncated...]"
                    else:
                        break

                context_parts.append(chunk_text)
                current_chars += len(chunk_text)

        return "".join(context_parts)

    def delete_filing(self, accession_number: str) -> bool:
        """
        Delete a filing from the index.

        Args:
            accession_number: SEC accession number to delete

        Returns:
            True if deleted successfully
        """
        self._ensure_initialized()

        try:
            # Get all chunk IDs for this filing
            results = self._collection.get(
                where={"accession_number": accession_number},
                include=[]
            )

            if results and results.get("ids"):
                self._collection.delete(ids=results["ids"])
                self._indexed_accessions.discard(accession_number)
                logger.info(f"Deleted {len(results['ids'])} chunks for {accession_number}")
                return True

            return False
        except Exception as e:
            logger.error(f"Error deleting filing {accession_number}: {e}")
            return False

    def clear_all(self) -> None:
        """Clear all indexed data."""
        self._ensure_initialized()

        try:
            # Delete and recreate collection
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            self._indexed_accessions.clear()
            logger.info("Cleared all indexed data")
        except Exception as e:
            logger.error(f"Error clearing data: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        self._ensure_initialized()

        try:
            count = self._collection.count()
            return {
                "total_chunks": count,
                "indexed_filings": len(self._indexed_accessions),
                "collection_name": self.collection_name,
                "embedding_provider": self.embedding_provider,
                "embedding_model": self.embedding_model,
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
