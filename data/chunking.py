"""
Chunking module for SEC filings.

Provides hybrid section/semantic chunking for SEC filings to prepare them for RAG indexing.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FilingChunk:
    """A chunk of a SEC filing."""
    chunk_id: str
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)

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


# SEC filing section patterns
SEC_SECTION_PATTERNS = {
    # 10-K sections
    "Item 1": r"(?:ITEM\s*1[.\s]+|PART\s+I\s*[.\s]+ITEM\s*1[.\s]+)(?:BUSINESS|Description\s+of\s+Business)",
    "Item 1A": r"ITEM\s*1A[.\s]+RISK\s*FACTORS",
    "Item 1B": r"ITEM\s*1B[.\s]+UNRESOLVED\s*STAFF\s*COMMENTS",
    "Item 2": r"ITEM\s*2[.\s]+PROPERTIES",
    "Item 3": r"ITEM\s*3[.\s]+LEGAL\s*PROCEEDINGS",
    "Item 4": r"ITEM\s*4[.\s]+MINE\s*SAFETY",
    "Item 5": r"ITEM\s*5[.\s]+MARKET\s*FOR",
    "Item 6": r"ITEM\s*6[.\s]+(?:RESERVED|\[RESERVED\]|SELECTED)",
    "Item 7": r"ITEM\s*7[.\s]+MANAGEMENT['\u2019]?S?\s*DISCUSSION",
    "Item 7A": r"ITEM\s*7A[.\s]+QUANTITATIVE\s*AND\s*QUALITATIVE",
    "Item 8": r"ITEM\s*8[.\s]+FINANCIAL\s*STATEMENTS",
    "Item 9": r"ITEM\s*9[.\s]+CHANGES\s*IN\s*AND\s*DISAGREEMENTS",
    "Item 9A": r"ITEM\s*9A[.\s]+CONTROLS\s*AND\s*PROCEDURES",
    "Item 9B": r"ITEM\s*9B[.\s]+OTHER\s*INFORMATION",
    # 10-Q sections
    "Part I Item 1": r"PART\s+I[.\s]+ITEM\s*1[.\s]+FINANCIAL\s*STATEMENTS",
    "Part I Item 2": r"PART\s+I[.\s]+ITEM\s*2[.\s]+MANAGEMENT",
    "Part I Item 3": r"PART\s+I[.\s]+ITEM\s*3[.\s]+QUANTITATIVE",
    "Part I Item 4": r"PART\s+I[.\s]+ITEM\s*4[.\s]+CONTROLS",
    "Part II Item 1": r"PART\s+II[.\s]+ITEM\s*1[.\s]+LEGAL",
    "Part II Item 1A": r"PART\s+II[.\s]+ITEM\s*1A[.\s]+RISK",
    "Part II Item 2": r"PART\s+II[.\s]+ITEM\s*2[.\s]+UNREGISTERED",
    "Part II Item 6": r"PART\s+II[.\s]+ITEM\s*6[.\s]+EXHIBITS",
}

# Key sections to prioritize in chunking
KEY_SECTIONS = {
    "10-K": ["Item 1", "Item 1A", "Item 7", "Item 7A", "Item 8"],
    "10-Q": ["Part I Item 1", "Part I Item 2", "Part I Item 3", "Part II Item 1A"],
    "8-K": [],  # 8-Ks are typically short, chunk as whole
    "DEF 14A": [],
}


class SECFilingChunker:
    """
    Chunker for SEC filings using hybrid section/semantic chunking.

    Strategy:
    1. First attempts to extract SEC-standard sections (Item 1, 1A, 7, etc.)
    2. Chunks within sections using semantic boundaries (paragraphs, sentences)
    3. Falls back to fixed-size chunking if sections aren't found
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        max_chunk_size: int = 3000
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target size for chunks in characters
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size (chunks smaller than this are merged)
            max_chunk_size: Maximum chunk size (chunks larger than this are split)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk_filing(
        self,
        content: str,
        ticker: str,
        form_type: str,
        filing_date: str,
        accession_number: str,
        sections: Optional[Dict[str, str]] = None
    ) -> List[FilingChunk]:
        """
        Chunk a SEC filing into smaller pieces for RAG indexing.

        Args:
            content: Full text content of the filing
            ticker: Stock ticker symbol
            form_type: Form type (10-K, 10-Q, etc.)
            filing_date: Filing date string
            accession_number: SEC accession number
            sections: Pre-extracted sections (if available)

        Returns:
            List of FilingChunk objects
        """
        chunks = []
        base_metadata = {
            "ticker": ticker,
            "form_type": form_type,
            "filing_date": filing_date,
            "accession_number": accession_number,
        }

        # If sections are provided, chunk each section
        if sections:
            for section_name, section_content in sections.items():
                section_chunks = self._chunk_text(
                    section_content,
                    section_name=section_name
                )
                for i, chunk_text in enumerate(section_chunks):
                    chunk_id = f"{accession_number}_{section_name.replace(' ', '_')}_{i}"
                    chunks.append(FilingChunk(
                        chunk_id=chunk_id,
                        content=chunk_text,
                        metadata={
                            **base_metadata,
                            "section_name": section_name,
                            "chunk_index": str(i),
                        }
                    ))
        else:
            # Try to extract sections from content
            extracted_sections = self._extract_sections(content, form_type)

            if extracted_sections:
                for section_name, section_content in extracted_sections.items():
                    section_chunks = self._chunk_text(
                        section_content,
                        section_name=section_name
                    )
                    for i, chunk_text in enumerate(section_chunks):
                        chunk_id = f"{accession_number}_{section_name.replace(' ', '_')}_{i}"
                        chunks.append(FilingChunk(
                            chunk_id=chunk_id,
                            content=chunk_text,
                            metadata={
                                **base_metadata,
                                "section_name": section_name,
                                "chunk_index": str(i),
                            }
                        ))
            else:
                # Fall back to simple chunking
                simple_chunks = self._chunk_text(content, section_name="Full Document")
                for i, chunk_text in enumerate(simple_chunks):
                    chunk_id = f"{accession_number}_chunk_{i}"
                    chunks.append(FilingChunk(
                        chunk_id=chunk_id,
                        content=chunk_text,
                        metadata={
                            **base_metadata,
                            "section_name": "Full Document",
                            "chunk_index": str(i),
                        }
                    ))

        logger.info(f"Chunked {ticker} {form_type} ({filing_date}) into {len(chunks)} chunks")
        return chunks

    def _extract_sections(self, content: str, form_type: str) -> Dict[str, str]:
        """
        Extract SEC-standard sections from filing content.

        Args:
            content: Full filing text
            form_type: Form type to determine which sections to look for

        Returns:
            Dictionary mapping section names to content
        """
        sections = {}

        # Normalize form type
        form_upper = form_type.upper()
        if "10-K" in form_upper:
            section_list = KEY_SECTIONS.get("10-K", [])
        elif "10-Q" in form_upper:
            section_list = KEY_SECTIONS.get("10-Q", [])
        else:
            return sections  # Return empty for other form types

        # Find all section positions
        section_positions = []
        for section_name in section_list:
            pattern = SEC_SECTION_PATTERNS.get(section_name)
            if pattern:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    section_positions.append((section_name, match.start()))

        # Sort by position
        section_positions.sort(key=lambda x: x[1])

        # Extract content between sections
        for i, (section_name, start_pos) in enumerate(section_positions):
            if i + 1 < len(section_positions):
                end_pos = section_positions[i + 1][1]
            else:
                # Last section - take reasonable amount or to end
                end_pos = min(start_pos + 100000, len(content))

            section_content = content[start_pos:end_pos].strip()
            if len(section_content) > self.min_chunk_size:
                sections[section_name] = section_content

        return sections

    def _chunk_text(
        self,
        text: str,
        section_name: Optional[str] = None
    ) -> List[str]:
        """
        Chunk text using semantic boundaries.

        Strategy:
        1. Split by paragraphs (double newlines)
        2. If paragraphs are too large, split by sentences
        3. Combine small chunks to meet minimum size
        4. Ensure overlap between chunks

        Args:
            text: Text to chunk
            section_name: Section name for logging

        Returns:
            List of chunk strings
        """
        if not text or len(text) < self.min_chunk_size:
            return [text] if text else []

        # Clean text
        text = self._clean_text(text)

        # Split by paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # If paragraph is too large, split by sentences
            if len(para) > self.max_chunk_size:
                sentences = self._split_sentences(para)
                for sent in sentences:
                    if len(current_chunk) + len(sent) + 1 <= self.chunk_size:
                        current_chunk = current_chunk + " " + sent if current_chunk else sent
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sent
            elif len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        # Apply overlap
        chunks = self._apply_overlap(chunks)

        # Merge chunks that are too small
        chunks = self._merge_small_chunks(chunks)

        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove HTML artifacts
        text = re.sub(r'<[^>]+>', '', text)
        # Remove page numbers and headers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        return text.strip()

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Apply overlap between consecutive chunks."""
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks

        overlapped = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]

            # Get overlap from end of previous chunk
            overlap_text = ""
            if len(prev_chunk) > self.chunk_overlap:
                # Find a sentence boundary for overlap
                overlap_candidate = prev_chunk[-self.chunk_overlap:]
                sentence_match = re.search(r'\.\s+', overlap_candidate)
                if sentence_match:
                    overlap_text = overlap_candidate[sentence_match.end():]

            if overlap_text:
                overlapped.append(f"[...] {overlap_text}\n\n{current_chunk}")
            else:
                overlapped.append(current_chunk)

        return overlapped

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge chunks that are smaller than min_chunk_size."""
        if not chunks:
            return chunks

        merged = []
        current = chunks[0]

        for i in range(1, len(chunks)):
            if len(current) < self.min_chunk_size:
                current = current + "\n\n" + chunks[i]
            else:
                merged.append(current)
                current = chunks[i]

        if current:
            merged.append(current)

        return merged
