# Feature Plan: Projects & Secure API Key Management

## Overview

This document outlines the proposed implementation for two new features:
1. **Projects** - A way to organize research materials (SEC filings + uploaded documents)
2. **Secure API Key Storage** - Persistent, secure storage for API keys across browser sessions

---

## Feature 1: Projects

### Concept

Similar to Claude Desktop's Projects feature, users can:
- Create named projects (e.g., "AAPL Research", "Tech Sector Analysis")
- Add SEC filings from EDGAR (existing functionality)
- Upload custom documents (PDFs, Word docs, text files, CSVs)
- All project documents are indexed for RAG-powered search
- Projects persist across sessions

### User Experience

```
┌─────────────────────────────────────────────────────────────┐
│ Sidebar                                                      │
├─────────────────────────────────────────────────────────────┤
│ 📁 Projects                                                  │
│   [+ New Project]                                            │
│                                                              │
│   ▼ AAPL Deep Dive (active)                                 │
│     • 10-K 2024 (SEC)                                       │
│     • 10-Q Q3 2024 (SEC)                                    │
│     • analyst_report.pdf (uploaded)                         │
│     • earnings_call_transcript.txt (uploaded)               │
│     [+ Add Files] [+ Add SEC Filing]                        │
│                                                              │
│   ▶ Tech Sector Comparison                                  │
│   ▶ Portfolio Risk Analysis                                 │
└─────────────────────────────────────────────────────────────┘
```

### Technical Architecture

#### Storage Structure

```
workspace/
└── projects/
    ├── projects.json              # Project metadata index
    └── {project_id}/
        ├── metadata.json          # Project details
        ├── documents/
        │   ├── {doc_id}_report.pdf
        │   └── {doc_id}_notes.txt
        └── chromadb/              # Project-specific vector store
```

#### Data Models

```python
@dataclass
class Project:
    id: str                        # UUID
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    tickers: List[str]             # Associated tickers
    documents: List[ProjectDocument]
    sec_filings: List[str]         # Accession numbers

@dataclass
class ProjectDocument:
    id: str                        # UUID
    filename: str
    original_name: str
    file_type: str                 # pdf, docx, txt, csv, md
    size_bytes: int
    uploaded_at: datetime
    indexed: bool                  # Whether indexed in RAG
    chunk_count: int
```

#### Supported File Types

| Type | Extensions | Processing |
|------|------------|------------|
| PDF | .pdf | PyPDF2 or pdfplumber text extraction |
| Word | .docx, .doc | python-docx extraction |
| Text | .txt, .md | Direct read |
| CSV/Excel | .csv, .xlsx | pandas + text summary |
| HTML | .html | BeautifulSoup extraction |

#### New Components

1. **`data/project_manager.py`** - Project CRUD operations
2. **`data/document_processor.py`** - File upload, text extraction, chunking
3. **`ui/projects_sidebar.py`** - Project selection and management UI
4. **Update `data/rag_manager.py`** - Support project-scoped collections

### Integration with Existing Features

- **SEC Filings**: When loading an SEC filing, user can optionally add it to the active project
- **RAG**: Project documents are chunked and indexed alongside SEC filings
- **Chat**: Context includes all documents in the active project
- **Price Analysis**: Automatically uses tickers from active project

---

## Feature 2: Secure API Key Storage

### Current Limitations

| Method | Persists? | Secure? | User Friction |
|--------|-----------|---------|---------------|
| Environment variables | Yes | Yes | High (terminal setup) |
| secrets.toml | Yes | Medium | Medium (file creation) |
| Session state | No | Yes | High (re-enter each time) |

### Proposed Solution: OS Keyring + Encrypted Local Config

Use a **hybrid approach**:

1. **Primary: OS Keyring** (via `keyring` library)
   - macOS: Keychain
   - Windows: Credential Manager  
   - Linux: Secret Service (GNOME Keyring, KWallet)
   
2. **Fallback: Encrypted local config** (for systems without keyring)
   - AES-256 encrypted file
   - Master password derived from machine-specific identifier

### User Experience

```
┌─────────────────────────────────────────────────────────────┐
│ API Key Configuration                                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ OpenAI API Key                                               │
│ ┌─────────────────────────────────────┐                     │
│ │ ●●●●●●●●●●●●●●●●●●●●               │  [Show] [Delete]    │
│ └─────────────────────────────────────┘                     │
│ ✓ Stored securely in system keychain                        │
│                                                              │
│ Anthropic API Key                                            │
│ ┌─────────────────────────────────────┐                     │
│ │ sk-ant-...                          │  [Save] [Test]      │
│ └─────────────────────────────────────┘                     │
│ ☐ Save to secure storage                                    │
│                                                              │
│ Google API Key                                               │
│ [Not configured - Enter key above]                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Technical Architecture

#### Priority Order for Key Retrieval

```python
def get_api_key(provider: str) -> tuple[str, str]:
    """
    Returns (key, source) with priority:
    1. Environment variable (ANTHROPIC_API_KEY, etc.)
    2. Streamlit secrets.toml
    3. OS Keyring
    4. Encrypted local config (fallback)
    5. Session state (temporary)
    """
```

#### New Components

1. **`core/credential_manager.py`** - Unified credential management
   ```python
   class CredentialManager:
       def store_key(provider: str, key: str) -> bool
       def get_key(provider: str) -> Optional[str]
       def delete_key(provider: str) -> bool
       def list_stored_providers() -> List[str]
       def test_key(provider: str, key: str) -> bool
   ```

2. **Update `ui/sidebar.py`** - New API key management UI

#### Dependencies

```
keyring>=24.0.0          # OS keyring access
cryptography>=41.0.0     # AES encryption for fallback
```

#### Security Considerations

| Concern | Mitigation |
|---------|------------|
| Keys in memory | Cleared from session state after storage |
| File permissions | Config file created with 0600 (owner read/write only) |
| Encryption key derivation | PBKDF2 with machine-specific salt |
| Keyring access | Requires OS authentication on some systems |

---

## Implementation Plan

### Phase 1: Secure API Key Storage (simpler, immediate value)

1. Add `keyring` and `cryptography` to requirements.txt
2. Create `core/credential_manager.py`
3. Update `ui/sidebar.py` with new key management UI
4. Update `get_api_key()` function to use new priority order
5. Test across platforms

**Estimated scope**: ~300-400 lines of new code

### Phase 2: Projects Foundation

1. Create data models and `data/project_manager.py`
2. Create `data/document_processor.py` for file handling
3. Add project storage directory structure
4. Create `ui/projects_sidebar.py`

**Estimated scope**: ~500-600 lines of new code

### Phase 3: Projects + RAG Integration

1. Update RAG manager for project-scoped collections
2. Implement document chunking for uploaded files
3. Update chat context to use active project
4. Add drag-and-drop upload UI

**Estimated scope**: ~400-500 lines of new code

### Phase 4: Polish & UX

1. Project import/export
2. Document preview
3. Bulk operations
4. Search across projects

---

## Questions for Review

1. **Projects scope**: Should projects be shareable/exportable (zip file with all documents)?

2. **File size limits**: What's a reasonable max file size for uploads? (Suggest: 50MB per file, 500MB per project)

3. **API key storage preference**: 
   - Option A: OS Keyring only (simpler, but may fail on some systems)
   - Option B: Keyring + encrypted fallback (more robust)
   - Option C: Just encrypted local file (works everywhere, slightly less secure)

4. **Multi-user consideration**: Is this single-user only, or might multiple people use the same installation?

5. **Project deletion**: Soft delete (archive) or hard delete with confirmation?

6. **Existing data migration**: Should we auto-migrate currently loaded SEC filings into a "Default" project?

---

## Alternative Approaches Considered

### For API Keys

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Browser localStorage | Simple | Streamlit can't access directly | ❌ |
| SQLite with encryption | Portable | Extra dependency, custom crypto | ❌ |
| OS Keyring | Native security, standard | May not work on all systems | ✅ Primary |
| Encrypted file | Works everywhere | Need to manage encryption | ✅ Fallback |

### For Projects

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| SQLite database | Structured, queryable | Extra complexity | ❌ |
| JSON + files | Simple, debuggable | Less structured | ✅ |
| Cloud storage | Sync across devices | Requires auth, privacy | ❌ Future |

---

## Next Steps

Please review this plan and let me know:
1. Any concerns or changes to the proposed approach
2. Answers to the questions above
3. Which phase to start with (recommend Phase 1: API Keys)

Once approved, I'll begin implementation.
