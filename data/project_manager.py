"""
Project Manager for Financial Research Agent.

Manages projects that contain SEC filings and uploaded documents.
Supports multi-user with automatic default project creation.
"""

import json
import shutil
import logging
import getpass
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import uuid4

logger = logging.getLogger(__name__)

# Size limits
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB per file
MAX_PROJECT_SIZE_BYTES = 500 * 1024 * 1024  # 500MB per project


@dataclass
class ProjectDocument:
    """Represents an uploaded document in a project."""
    id: str
    filename: str  # Stored filename (sanitized)
    original_name: str  # Original upload name
    file_type: str  # pdf, docx, txt, csv, md, html
    size_bytes: int
    uploaded_at: str  # ISO format
    indexed: bool = False  # Whether indexed in RAG
    chunk_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectDocument":
        return cls(**data)


@dataclass
class ProjectSECFiling:
    """Reference to an SEC filing in a project."""
    ticker: str
    form_type: str
    filing_date: str
    accession_number: str
    display_name: str
    added_at: str  # ISO format
    indexed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectSECFiling":
        return cls(**data)


@dataclass
class Project:
    """Represents a research project."""
    id: str
    name: str
    description: str
    created_at: str  # ISO format
    updated_at: str  # ISO format
    owner: str  # Username
    tickers: List[str] = field(default_factory=list)
    documents: List[ProjectDocument] = field(default_factory=list)
    sec_filings: List[ProjectSECFiling] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["documents"] = [d.to_dict() if isinstance(d, ProjectDocument) else d for d in self.documents]
        data["sec_filings"] = [f.to_dict() if isinstance(f, ProjectSECFiling) else f for f in self.sec_filings]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
        documents = [ProjectDocument.from_dict(d) if isinstance(d, dict) else d for d in data.get("documents", [])]
        sec_filings = [ProjectSECFiling.from_dict(f) if isinstance(f, dict) else f for f in data.get("sec_filings", [])]
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            owner=data.get("owner", "default"),
            tickers=data.get("tickers", []),
            documents=documents,
            sec_filings=sec_filings
        )
    
    def get_total_size(self) -> int:
        """Get total size of all documents in bytes."""
        return sum(d.size_bytes for d in self.documents)
    
    def get_document_count(self) -> int:
        """Get count of documents plus SEC filings."""
        return len(self.documents) + len(self.sec_filings)


class ProjectManager:
    """
    Manages research projects with documents and SEC filings.
    
    Features:
    - Create, read, update, delete projects
    - Add/remove documents and SEC filings
    - Multi-user support (projects are per-user)
    - Auto-create default project when needed
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the project manager.
        
        Args:
            base_dir: Base directory for project storage.
                     Defaults to workspace/projects/
        """
        if base_dir is None:
            # Default to workspace/projects/
            base_dir = Path(__file__).parent.parent / "projects"
        
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self._projects_cache: Dict[str, Project] = {}
        self._load_projects_index()
        
        logger.info(f"ProjectManager initialized at {self.base_dir}")
    
    def _get_current_user(self) -> str:
        """Get current username."""
        return getpass.getuser()
    
    def _get_projects_index_path(self) -> Path:
        """Get path to projects index file."""
        return self.base_dir / "projects_index.json"
    
    def _get_project_dir(self, project_id: str) -> Path:
        """Get directory for a specific project."""
        return self.base_dir / project_id
    
    def _get_project_metadata_path(self, project_id: str) -> Path:
        """Get path to project metadata file."""
        return self._get_project_dir(project_id) / "metadata.json"
    
    def _get_project_documents_dir(self, project_id: str) -> Path:
        """Get directory for project documents."""
        doc_dir = self._get_project_dir(project_id) / "documents"
        doc_dir.mkdir(parents=True, exist_ok=True)
        return doc_dir
    
    def _load_projects_index(self) -> None:
        """Load the projects index from disk."""
        index_path = self._get_projects_index_path()
        
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    index_data = json.load(f)
                
                for project_id, project_data in index_data.get("projects", {}).items():
                    try:
                        self._projects_cache[project_id] = Project.from_dict(project_data)
                    except Exception as e:
                        logger.warning(f"Failed to load project {project_id}: {e}")
                
                logger.info(f"Loaded {len(self._projects_cache)} projects from index")
            except Exception as e:
                logger.error(f"Failed to load projects index: {e}")
                self._projects_cache = {}
        else:
            self._projects_cache = {}
    
    def _save_projects_index(self) -> None:
        """Save the projects index to disk."""
        index_path = self._get_projects_index_path()
        
        try:
            index_data = {
                "version": 1,
                "updated_at": datetime.now().isoformat(),
                "projects": {pid: p.to_dict() for pid, p in self._projects_cache.items()}
            }
            
            with open(index_path, "w") as f:
                json.dump(index_data, f, indent=2)
            
            logger.debug("Saved projects index")
        except Exception as e:
            logger.error(f"Failed to save projects index: {e}")
    
    def _save_project_metadata(self, project: Project) -> None:
        """Save individual project metadata."""
        project_dir = self._get_project_dir(project.id)
        project_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_path = self._get_project_metadata_path(project.id)
        
        try:
            with open(metadata_path, "w") as f:
                json.dump(project.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save project metadata: {e}")
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename for safe storage."""
        # Remove path separators and null bytes
        sanitized = filename.replace("/", "_").replace("\\", "_").replace("\x00", "")
        # Limit length
        if len(sanitized) > 200:
            name, ext = sanitized.rsplit(".", 1) if "." in sanitized else (sanitized, "")
            sanitized = name[:190] + ("." + ext if ext else "")
        return sanitized
    
    def list_projects(self, owner: Optional[str] = None) -> List[Project]:
        """
        List all projects, optionally filtered by owner.
        
        Args:
            owner: Filter by owner username. If None, uses current user.
        
        Returns:
            List of Project objects
        """
        if owner is None:
            owner = self._get_current_user()
        
        projects = [p for p in self._projects_cache.values() if p.owner == owner]
        # Sort by updated_at descending
        projects.sort(key=lambda p: p.updated_at, reverse=True)
        return projects
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """
        Get a project by ID.
        
        Args:
            project_id: Project ID
            
        Returns:
            Project or None if not found
        """
        return self._projects_cache.get(project_id)
    
    def create_project(
        self,
        name: str,
        description: str = "",
        tickers: Optional[List[str]] = None,
        owner: Optional[str] = None
    ) -> Project:
        """
        Create a new project.
        
        Args:
            name: Project name
            description: Project description
            tickers: Optional list of ticker symbols
            owner: Owner username. Defaults to current user.
            
        Returns:
            Created Project
        """
        if owner is None:
            owner = self._get_current_user()
        
        project_id = str(uuid4())
        now = datetime.now().isoformat()
        
        project = Project(
            id=project_id,
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
            owner=owner,
            tickers=tickers or [],
            documents=[],
            sec_filings=[]
        )
        
        # Create project directory
        project_dir = self._get_project_dir(project_id)
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        self._projects_cache[project_id] = project
        self._save_project_metadata(project)
        self._save_projects_index()
        
        logger.info(f"Created project '{name}' (id={project_id}) for {owner}")
        return project
    
    def update_project(
        self,
        project_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tickers: Optional[List[str]] = None
    ) -> Optional[Project]:
        """
        Update a project's metadata.
        
        Args:
            project_id: Project ID
            name: New name (optional)
            description: New description (optional)
            tickers: New tickers list (optional)
            
        Returns:
            Updated Project or None if not found
        """
        project = self._projects_cache.get(project_id)
        if not project:
            return None
        
        if name is not None:
            project.name = name
        if description is not None:
            project.description = description
        if tickers is not None:
            project.tickers = tickers
        
        project.updated_at = datetime.now().isoformat()
        
        self._save_project_metadata(project)
        self._save_projects_index()
        
        logger.info(f"Updated project {project_id}")
        return project
    
    def delete_project(self, project_id: str) -> bool:
        """
        Delete a project and all its contents.
        
        Args:
            project_id: Project ID
            
        Returns:
            True if deleted, False if not found
        """
        project = self._projects_cache.get(project_id)
        if not project:
            return False
        
        # Remove from cache
        del self._projects_cache[project_id]
        
        # Delete project directory
        project_dir = self._get_project_dir(project_id)
        if project_dir.exists():
            try:
                shutil.rmtree(project_dir)
            except Exception as e:
                logger.error(f"Failed to delete project directory: {e}")
        
        # Save updated index
        self._save_projects_index()
        
        logger.info(f"Deleted project {project_id}")
        return True
    
    def add_document(
        self,
        project_id: str,
        file_content: bytes,
        original_filename: str,
        file_type: str
    ) -> Optional[ProjectDocument]:
        """
        Add a document to a project.
        
        Args:
            project_id: Project ID
            file_content: File content as bytes
            original_filename: Original filename
            file_type: File type (pdf, docx, txt, etc.)
            
        Returns:
            ProjectDocument or None if failed
        """
        project = self._projects_cache.get(project_id)
        if not project:
            logger.error(f"Project {project_id} not found")
            return None
        
        # Check file size
        if len(file_content) > MAX_FILE_SIZE_BYTES:
            logger.error(f"File too large: {len(file_content)} bytes (max {MAX_FILE_SIZE_BYTES})")
            return None
        
        # Check project size limit
        current_size = project.get_total_size()
        if current_size + len(file_content) > MAX_PROJECT_SIZE_BYTES:
            logger.error(f"Project size limit exceeded")
            return None
        
        # Generate document ID and sanitized filename
        doc_id = str(uuid4())[:8]
        sanitized_name = self._sanitize_filename(original_filename)
        stored_filename = f"{doc_id}_{sanitized_name}"
        
        # Save file
        doc_dir = self._get_project_documents_dir(project_id)
        file_path = doc_dir / stored_filename
        
        try:
            file_path.write_bytes(file_content)
        except Exception as e:
            logger.error(f"Failed to save document: {e}")
            return None
        
        # Create document record
        document = ProjectDocument(
            id=doc_id,
            filename=stored_filename,
            original_name=original_filename,
            file_type=file_type.lower(),
            size_bytes=len(file_content),
            uploaded_at=datetime.now().isoformat(),
            indexed=False,
            chunk_count=0
        )
        
        # Add to project
        project.documents.append(document)
        project.updated_at = datetime.now().isoformat()
        
        self._save_project_metadata(project)
        self._save_projects_index()
        
        logger.info(f"Added document '{original_filename}' to project {project_id}")
        return document
    
    def remove_document(self, project_id: str, document_id: str) -> bool:
        """
        Remove a document from a project.
        
        Args:
            project_id: Project ID
            document_id: Document ID
            
        Returns:
            True if removed, False if not found
        """
        project = self._projects_cache.get(project_id)
        if not project:
            return False
        
        # Find document
        doc_to_remove = None
        for doc in project.documents:
            if doc.id == document_id:
                doc_to_remove = doc
                break
        
        if not doc_to_remove:
            return False
        
        # Remove file
        doc_dir = self._get_project_documents_dir(project_id)
        file_path = doc_dir / doc_to_remove.filename
        
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete file: {e}")
        
        # Remove from project
        project.documents.remove(doc_to_remove)
        project.updated_at = datetime.now().isoformat()
        
        self._save_project_metadata(project)
        self._save_projects_index()
        
        logger.info(f"Removed document {document_id} from project {project_id}")
        return True
    
    def get_document_path(self, project_id: str, document_id: str) -> Optional[Path]:
        """
        Get the file path for a document.
        
        Args:
            project_id: Project ID
            document_id: Document ID
            
        Returns:
            Path to document file or None
        """
        project = self._projects_cache.get(project_id)
        if not project:
            return None
        
        for doc in project.documents:
            if doc.id == document_id:
                return self._get_project_documents_dir(project_id) / doc.filename
        
        return None
    
    def add_sec_filing(
        self,
        project_id: str,
        ticker: str,
        form_type: str,
        filing_date: str,
        accession_number: str,
        display_name: str
    ) -> Optional[ProjectSECFiling]:
        """
        Add an SEC filing reference to a project.
        
        Args:
            project_id: Project ID
            ticker: Stock ticker
            form_type: Form type (10-K, 10-Q, etc.)
            filing_date: Filing date
            accession_number: SEC accession number
            display_name: Display name for the filing
            
        Returns:
            ProjectSECFiling or None if failed
        """
        project = self._projects_cache.get(project_id)
        if not project:
            return None
        
        # Check if already added
        for filing in project.sec_filings:
            if filing.accession_number == accession_number:
                logger.info(f"Filing {accession_number} already in project")
                return filing
        
        filing = ProjectSECFiling(
            ticker=ticker,
            form_type=form_type,
            filing_date=filing_date,
            accession_number=accession_number,
            display_name=display_name,
            added_at=datetime.now().isoformat(),
            indexed=False
        )
        
        project.sec_filings.append(filing)
        
        # Add ticker if not present
        if ticker.upper() not in [t.upper() for t in project.tickers]:
            project.tickers.append(ticker.upper())
        
        project.updated_at = datetime.now().isoformat()
        
        self._save_project_metadata(project)
        self._save_projects_index()
        
        logger.info(f"Added SEC filing {accession_number} to project {project_id}")
        return filing
    
    def remove_sec_filing(self, project_id: str, accession_number: str) -> bool:
        """
        Remove an SEC filing reference from a project.
        
        Args:
            project_id: Project ID
            accession_number: SEC accession number
            
        Returns:
            True if removed, False if not found
        """
        project = self._projects_cache.get(project_id)
        if not project:
            return False
        
        filing_to_remove = None
        for filing in project.sec_filings:
            if filing.accession_number == accession_number:
                filing_to_remove = filing
                break
        
        if not filing_to_remove:
            return False
        
        project.sec_filings.remove(filing_to_remove)
        project.updated_at = datetime.now().isoformat()
        
        self._save_project_metadata(project)
        self._save_projects_index()
        
        logger.info(f"Removed SEC filing {accession_number} from project {project_id}")
        return True
    
    def get_or_create_default_project(self, ticker: str) -> Project:
        """
        Get or create a default project for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Project (existing or newly created)
        """
        ticker = ticker.upper()
        owner = self._get_current_user()
        
        # Look for existing project for this ticker
        for project in self.list_projects(owner):
            if ticker in [t.upper() for t in project.tickers]:
                return project
        
        # Create new project
        return self.create_project(
            name=f"{ticker} Research",
            description=f"Research project for {ticker}",
            tickers=[ticker],
            owner=owner
        )
    
    def mark_document_indexed(
        self,
        project_id: str,
        document_id: str,
        chunk_count: int
    ) -> bool:
        """
        Mark a document as indexed in RAG.
        
        Args:
            project_id: Project ID
            document_id: Document ID
            chunk_count: Number of chunks indexed
            
        Returns:
            True if updated, False if not found
        """
        project = self._projects_cache.get(project_id)
        if not project:
            return False
        
        for doc in project.documents:
            if doc.id == document_id:
                doc.indexed = True
                doc.chunk_count = chunk_count
                project.updated_at = datetime.now().isoformat()
                self._save_project_metadata(project)
                self._save_projects_index()
                return True
        
        return False
    
    def mark_filing_indexed(self, project_id: str, accession_number: str) -> bool:
        """
        Mark an SEC filing as indexed in RAG.
        
        Args:
            project_id: Project ID
            accession_number: SEC accession number
            
        Returns:
            True if updated, False if not found
        """
        project = self._projects_cache.get(project_id)
        if not project:
            return False
        
        for filing in project.sec_filings:
            if filing.accession_number == accession_number:
                filing.indexed = True
                project.updated_at = datetime.now().isoformat()
                self._save_project_metadata(project)
                self._save_projects_index()
                return True
        
        return False


# Global instance
_project_manager: Optional[ProjectManager] = None


def get_project_manager() -> ProjectManager:
    """Get or create the global project manager instance."""
    global _project_manager
    if _project_manager is None:
        _project_manager = ProjectManager()
    return _project_manager
