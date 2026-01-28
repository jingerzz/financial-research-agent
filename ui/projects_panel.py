"""
Projects Panel UI for Financial Research Agent.

Provides UI for managing research projects with documents and SEC filings.
"""

import streamlit as st
import logging
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


def get_project_manager():
    """Get or create the project manager singleton."""
    if "project_manager" not in st.session_state:
        try:
            from data.project_manager import get_project_manager
            st.session_state.project_manager = get_project_manager()
        except ImportError as e:
            logger.warning(f"Project manager not available: {e}")
            st.session_state.project_manager = None
        except Exception as e:
            logger.error(f"Error initializing project manager: {e}")
            st.session_state.project_manager = None
    return st.session_state.project_manager


def get_document_processor():
    """Get or create the document processor singleton."""
    if "document_processor" not in st.session_state:
        try:
            from data.document_processor import get_document_processor
            st.session_state.document_processor = get_document_processor()
        except ImportError as e:
            logger.warning(f"Document processor not available: {e}")
            st.session_state.document_processor = None
        except Exception as e:
            logger.error(f"Error initializing document processor: {e}")
            st.session_state.document_processor = None
    return st.session_state.document_processor


def get_rag_manager():
    """Get the RAG manager from sidebar module."""
    try:
        from ui.sidebar import get_rag_manager as sidebar_get_rag_manager
        return sidebar_get_rag_manager()
    except Exception:
        return None


def index_document_for_rag(
    project_id: str,
    document_id: str,
    file_content: bytes,
    filename: str,
    file_type: str
) -> int:
    """
    Process and index a document for RAG search.
    
    Args:
        project_id: Project ID
        document_id: Document ID
        file_content: Raw file content
        filename: Original filename
        file_type: File type
        
    Returns:
        Number of chunks indexed
    """
    logger.info(f"=== INDEXING DOCUMENT FOR RAG: {filename} ===")
    logger.info(f"  project_id={project_id}, document_id={document_id}, file_type={file_type}")
    logger.info(f"  file_content size: {len(file_content)} bytes")
    
    doc_processor = get_document_processor()
    rag_manager = get_rag_manager()
    
    if not doc_processor:
        logger.error("❌ Document processor not available!")
        return 0
    
    if not rag_manager:
        logger.error("❌ RAG manager not available! Documents will not be searchable.")
        # Try to get config to see if RAG is enabled
        try:
            from config import get_config
            config = get_config()
            logger.error(f"  RAG enabled in config: {config.rag.enabled}")
        except Exception as e:
            logger.error(f"  Could not check config: {e}")
        return 0
    
    logger.info(f"  doc_processor: OK, rag_manager: OK")
    
    try:
        # Extract text from document
        logger.info(f"  Extracting text from {filename}...")
        processed = doc_processor.process(file_content, filename)
        
        if not processed.text:
            logger.warning(f"  ❌ No text extracted from {filename}")
            return 0
        
        logger.info(f"  ✅ Extracted {len(processed.text)} chars from {filename}")
        logger.info(f"  First 200 chars: {processed.text[:200]}...")
        
        # Index into RAG
        logger.info(f"  Indexing into RAG with project_id={project_id}...")
        chunk_count = rag_manager.index_document(
            content=processed.text,
            project_id=project_id,
            document_id=document_id,
            filename=filename,
            file_type=file_type
        )
        
        logger.info(f"  ✅ Indexed {chunk_count} chunks for {filename}")
        
        # Verify indexing by checking stats
        stats = rag_manager.get_stats()
        logger.info(f"  RAG stats after indexing: {stats}")
        
        # Update project metadata
        project_mgr = get_project_manager()
        if project_mgr and chunk_count > 0:
            project_mgr.mark_document_indexed(project_id, document_id, chunk_count)
            logger.info(f"  ✅ Marked document as indexed in project metadata")
        
        return chunk_count
        
    except Exception as e:
        logger.error(f"❌ Error indexing document {filename}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0


def render_projects_panel(current_ticker: str = "") -> Optional[str]:
    """
    Render the projects panel in the sidebar.
    
    Args:
        current_ticker: Currently selected ticker symbol
        
    Returns:
        Active project ID or None
    """
    project_mgr = get_project_manager()
    
    if not project_mgr:
        st.warning("Projects feature unavailable")
        return None
    
    st.subheader("📁 Projects")
    
    # Initialize session state
    if "active_project_id" not in st.session_state:
        st.session_state.active_project_id = None
    
    # Get user's projects
    projects = project_mgr.list_projects()
    
    # Create new project button
    with st.expander("➕ New Project", expanded=len(projects) == 0):
        new_name = st.text_input(
            "Project Name",
            placeholder="e.g., AAPL Research",
            key="new_project_name"
        )
        new_desc = st.text_input(
            "Description (optional)",
            placeholder="Research notes for...",
            key="new_project_desc"
        )
        
        # Pre-fill ticker if available
        new_tickers = st.text_input(
            "Tickers (comma-separated)",
            value=current_ticker,
            placeholder="AAPL, MSFT",
            key="new_project_tickers"
        )
        
        if st.button("Create Project", key="create_project_btn", use_container_width=True):
            if new_name:
                tickers = [t.strip().upper() for t in new_tickers.split(",") if t.strip()]
                project = project_mgr.create_project(
                    name=new_name,
                    description=new_desc,
                    tickers=tickers
                )
                st.session_state.active_project_id = project.id
                st.success(f"Created project: {new_name}")
                st.rerun()
            else:
                st.error("Please enter a project name")
    
    # Project selector
    if projects:
        project_options = {p.id: f"{p.name} ({len(p.documents) + len(p.sec_filings)} items)" for p in projects}
        project_ids = list(project_options.keys())
        
        # Default to first project if none selected
        if st.session_state.active_project_id not in project_ids:
            st.session_state.active_project_id = project_ids[0] if project_ids else None
        
        selected_id = st.selectbox(
            "Active Project",
            options=project_ids,
            format_func=lambda x: project_options.get(x, x),
            index=project_ids.index(st.session_state.active_project_id) if st.session_state.active_project_id in project_ids else 0,
            key="project_selector"
        )
        
        if selected_id != st.session_state.active_project_id:
            st.session_state.active_project_id = selected_id
            st.rerun()
        
        # Show active project details
        active_project = project_mgr.get_project(st.session_state.active_project_id)
        
        if active_project:
            render_project_details(active_project, project_mgr, current_ticker)
    
    else:
        st.info("No projects yet. Create one above!")
        
        # Auto-create default project if ticker is set
        if current_ticker:
            if st.button(f"Quick: Create {current_ticker} Project", use_container_width=True):
                project = project_mgr.get_or_create_default_project(current_ticker)
                st.session_state.active_project_id = project.id
                st.rerun()
    
    return st.session_state.active_project_id


def render_project_details(project, project_mgr, current_ticker: str):
    """Render details and contents of the active project."""
    
    st.markdown(f"**{project.name}**")
    if project.description:
        st.caption(project.description)
    
    if project.tickers:
        st.caption(f"Tickers: {', '.join(project.tickers)}")
    
    # Project actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✏️ Edit", key="edit_project", use_container_width=True):
            st.session_state.editing_project = True
    with col2:
        if st.button("🗑️ Delete", key="delete_project", use_container_width=True):
            st.session_state.confirm_delete_project = True
    
    # Edit dialog
    if st.session_state.get("editing_project"):
        with st.form("edit_project_form"):
            edit_name = st.text_input("Name", value=project.name)
            edit_desc = st.text_input("Description", value=project.description)
            edit_tickers = st.text_input("Tickers", value=", ".join(project.tickers))
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Save"):
                    tickers = [t.strip().upper() for t in edit_tickers.split(",") if t.strip()]
                    project_mgr.update_project(
                        project.id,
                        name=edit_name,
                        description=edit_desc,
                        tickers=tickers
                    )
                    st.session_state.editing_project = False
                    st.rerun()
            with col2:
                if st.form_submit_button("Cancel"):
                    st.session_state.editing_project = False
                    st.rerun()
    
    # Delete confirmation
    if st.session_state.get("confirm_delete_project"):
        st.warning(f"Delete project '{project.name}' and all its documents?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, Delete", key="confirm_delete_yes", type="primary"):
                project_mgr.delete_project(project.id)
                st.session_state.active_project_id = None
                st.session_state.confirm_delete_project = False
                st.success("Project deleted")
                st.rerun()
        with col2:
            if st.button("Cancel", key="confirm_delete_no"):
                st.session_state.confirm_delete_project = False
                st.rerun()
        return  # Don't show rest of UI during delete confirmation
    
    st.divider()
    
    # File upload section
    st.markdown("**Add Documents**")
    
    # Initialize processing tracking
    processing_key = f"processing_files_{project.id}"
    if processing_key not in st.session_state:
        st.session_state[processing_key] = set()
    
    # Check for pending uploads from previous render
    pending_key = f"pending_uploads_{project.id}"
    if pending_key in st.session_state and st.session_state[pending_key]:
        uploaded_files = st.session_state[pending_key]
        del st.session_state[pending_key]  # Clear pending
    else:
        uploaded_files = st.file_uploader(
            "Drag & drop files",
            type=["pdf", "docx", "doc", "txt", "md", "csv", "xlsx", "xls", "html"],
            accept_multiple_files=True,
            key=f"upload_{project.id}",
            help="Supported: PDF, Word, Text, Markdown, CSV, Excel, HTML"
        )
    
    # Process uploaded files
    if uploaded_files:
        logger.info(f"=== FILE UPLOAD: {len(uploaded_files)} file(s) received ===")
        for uf in uploaded_files:
            logger.info(f"  - {uf.name} ({uf.size} bytes)")
        
        doc_processor = get_document_processor()
        if not doc_processor:
            st.error("❌ Document processor not available. Check if dependencies are installed.")
            logger.error("Document processor is None!")
        else:
            logger.info(f"Document processor available, dependencies: {doc_processor._dependencies}")
        
        processed_count = 0
        failed_count = 0
        processed_names = []
        failed_names = []
        
        with st.spinner("Processing files..."):
            for uploaded_file in uploaded_files:
                # Create unique key for this file (name + size + timestamp)
                file_key = f"{uploaded_file.name}_{uploaded_file.size}"
                logger.info(f"Processing file: {uploaded_file.name}, key={file_key}")
                
                # Check if already processed in this session
                if file_key in st.session_state[processing_key]:
                    logger.info(f"  -> Already processed in this session, skipping")
                    continue
                
                # Mark as processing
                st.session_state[processing_key].add(file_key)
                
                try:
                    # Check if file already exists in project (by name and size)
                    already_exists = any(
                        doc.original_name == uploaded_file.name and 
                        doc.size_bytes == uploaded_file.size
                        for doc in project.documents
                    )
                    
                    if already_exists:
                        logger.info(f"  -> Document {uploaded_file.name} already exists in project, skipping")
                        st.info(f"⏭️ Skipped {uploaded_file.name} (already in project)")
                        # Don't mark as processed so user knows it was skipped
                        st.session_state[processing_key].discard(file_key)
                        continue
                    
                    if doc_processor and doc_processor.is_supported(uploaded_file.name):
                        # Read file content (only once)
                        file_content = uploaded_file.read()
                        logger.info(f"  -> Read {len(file_content)} bytes from {uploaded_file.name}")
                        
                        # Size limits
                        MAX_FILE_SIZE_MB = 50
                        MAX_PROJECT_SIZE_MB = 500
                        MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
                        MAX_PROJECT_SIZE_BYTES = MAX_PROJECT_SIZE_MB * 1024 * 1024
                        
                        if len(file_content) == 0:
                            logger.error(f"  -> File content is empty!")
                            st.error(f"❌ **{uploaded_file.name}** is empty or could not be read")
                            st.caption("💡 Try re-saving the file or converting to a different format")
                            st.session_state[processing_key].discard(file_key)
                            failed_count += 1
                            failed_names.append(uploaded_file.name)
                            continue
                        
                        # Check file size limit
                        if len(file_content) > MAX_FILE_SIZE_BYTES:
                            file_size_mb = len(file_content) / (1024 * 1024)
                            logger.error(f"  -> File too large: {file_size_mb:.1f}MB (max {MAX_FILE_SIZE_MB}MB)")
                            st.error(f"❌ **{uploaded_file.name}** is too large ({file_size_mb:.1f}MB)")
                            st.caption(f"💡 Maximum file size is {MAX_FILE_SIZE_MB}MB. Try compressing the file or splitting it.")
                            st.session_state[processing_key].discard(file_key)
                            failed_count += 1
                            failed_names.append(uploaded_file.name)
                            continue
                        
                        # Check project size limit
                        current_project_size = project.get_total_size()
                        if current_project_size + len(file_content) > MAX_PROJECT_SIZE_BYTES:
                            current_mb = current_project_size / (1024 * 1024)
                            file_mb = len(file_content) / (1024 * 1024)
                            logger.error(f"  -> Project size limit exceeded: {current_mb:.1f}MB + {file_mb:.1f}MB > {MAX_PROJECT_SIZE_MB}MB")
                            st.error(f"❌ Project size limit exceeded")
                            st.caption(f"💡 Current: {current_mb:.1f}MB, Adding: {file_mb:.1f}MB, Max: {MAX_PROJECT_SIZE_MB}MB. Remove some documents first.")
                            st.session_state[processing_key].discard(file_key)
                            failed_count += 1
                            failed_names.append(uploaded_file.name)
                            continue
                        
                        file_type = doc_processor.get_file_type(uploaded_file.name)
                        logger.info(f"  -> File type: {file_type}")
                        
                        # Add to project
                        doc = project_mgr.add_document(
                            project.id,
                            file_content,
                            uploaded_file.name,
                            file_type
                        )
                        
                        if doc:
                            processed_count += 1
                            processed_names.append(uploaded_file.name)
                            logger.info(f"  -> ✅ Added document: {uploaded_file.name} to project {project.id}, doc_id={doc.id}")
                            st.toast(f"📄 Added {uploaded_file.name}", icon="✅")
                            
                            # Index document for RAG
                            try:
                                chunks_indexed = index_document_for_rag(project.id, doc.id, file_content, uploaded_file.name, file_type)
                                logger.info(f"  -> ✅ Indexed {chunks_indexed} chunks for RAG")
                                if chunks_indexed > 0:
                                    st.toast(f"🔍 Indexed {chunks_indexed} chunks for {uploaded_file.name}", icon="✅")
                                else:
                                    st.warning(f"⚠️ No text extracted from {uploaded_file.name}")
                            except Exception as e:
                                logger.warning(f"  -> ⚠️ Failed to index document for RAG: {e}")
                                import traceback
                                logger.warning(traceback.format_exc())
                                st.warning(f"⚠️ {uploaded_file.name} added but not indexed for search")
                        else:
                            failed_count += 1
                            failed_names.append(uploaded_file.name)
                            logger.error(f"  -> ❌ Failed to add document: {uploaded_file.name}")
                            st.error(f"❌ Failed to add {uploaded_file.name}")
                            # Remove from processing set so it can be retried
                            st.session_state[processing_key].discard(file_key)
                    else:
                        failed_count += 1
                        failed_names.append(uploaded_file.name)
                        if not doc_processor:
                            logger.warning(f"  -> ❌ No document processor available")
                            st.error(f"❌ Document processor not available")
                            st.caption("💡 Install dependencies: `pip install PyPDF2 python-docx beautifulsoup4`")
                        else:
                            # Get file extension
                            import os
                            ext = os.path.splitext(uploaded_file.name)[1].lower()
                            logger.warning(f"  -> ❌ Unsupported file type: {uploaded_file.name} (ext: {ext})")
                            st.error(f"❌ Unsupported file type: **{uploaded_file.name}**")
                            st.caption(f"💡 Supported formats: PDF, Word (.docx), Text (.txt), Markdown (.md), CSV, Excel (.xlsx), HTML\n\nTry converting '{ext}' to one of these formats.")
                        # Remove from processing set
                        st.session_state[processing_key].discard(file_key)
                except Exception as e:
                    failed_count += 1
                    failed_names.append(uploaded_file.name)
                    logger.error(f"  -> ❌ Error processing {uploaded_file.name}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    # Remove from processing set so it can be retried
                    st.session_state[processing_key].discard(file_key)
        
        # Show results
        if processed_count > 0:
            st.success(f"✅ Added {processed_count} file(s): {', '.join(processed_names[:3])}{'...' if len(processed_names) > 3 else ''}")
        if failed_count > 0:
            st.error(f"❌ Failed to process {failed_count} file(s): {', '.join(failed_names[:3])}{'...' if len(failed_names) > 3 else ''}")
        
        # Rerun to refresh UI and clear file uploader
        if processed_count > 0 or failed_count > 0:
            st.rerun()
    
    # Show project contents
    st.markdown("**Project Contents**")
    
    # Add Re-index button if there are documents
    if project.documents:
        not_indexed = [d for d in project.documents if not d.indexed]
        indexed = [d for d in project.documents if d.indexed]
        st.caption(f"📄 {len(indexed)} indexed, {len(not_indexed)} not indexed")
        
        if st.button("🔄 Re-index All Documents for Search", 
                    key=f"reindex_docs_{project.id}",
                    use_container_width=True,
                    help="Re-index all documents to make them searchable in Research Chat"):
            with st.spinner("Re-indexing documents..."):
                reindex_count = 0
                for doc in project.documents:
                    # Get document content from disk
                    doc_path = project_mgr._get_project_documents_dir(project.id) / doc.filename
                    if doc_path.exists():
                        try:
                            file_content = doc_path.read_bytes()
                            chunks = index_document_for_rag(
                                project.id, doc.id, file_content, 
                                doc.original_name, doc.file_type
                            )
                            if chunks > 0:
                                reindex_count += 1
                                logger.info(f"Re-indexed {doc.original_name}: {chunks} chunks")
                        except Exception as e:
                            logger.error(f"Failed to re-index {doc.original_name}: {e}")
                            st.error(f"Failed to re-index {doc.original_name}")
                    else:
                        logger.warning(f"Document file not found: {doc_path}")
                
                if reindex_count > 0:
                    st.success(f"✅ Re-indexed {reindex_count} document(s)")
                    # Check RAG stats
                    rag_mgr = get_rag_manager()
                    if rag_mgr:
                        stats = rag_mgr.get_stats()
                        st.info(f"RAG now has {stats.get('document_chunks', 0)} document chunks")
                else:
                    st.warning("No documents were re-indexed")
                st.rerun()
    
    total_items = len(project.documents) + len(project.sec_filings)
    
    if total_items == 0:
        st.caption("No documents yet. Upload files or add SEC filings.")
    else:
        # Bulk delete buttons
        if project.documents or project.sec_filings:
            # Check for confirmation state
            confirm_key = f"confirm_delete_all_docs_{project.id}"
            if confirm_key in st.session_state and st.session_state[confirm_key]:
                st.warning(f"⚠️ Delete all {len(project.documents)} document(s)?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Yes, Delete All Documents", key=f"confirm_yes_docs_{project.id}", 
                                use_container_width=True, type="primary"):
                        deleted_count = 0
                        for doc in project.documents[:]:  # Copy list to avoid modification during iteration
                            if project_mgr.remove_document(project.id, doc.id):
                                deleted_count += 1
                        st.session_state[confirm_key] = False
                        st.success(f"✅ Deleted {deleted_count} document(s)")
                        st.rerun()
                with col2:
                    if st.button("❌ Cancel", key=f"confirm_no_docs_{project.id}", 
                                use_container_width=True):
                        st.session_state[confirm_key] = False
                        st.rerun()
            else:
                # Show buttons only for items that exist
                if project.documents and project.sec_filings:
                    # Both exist - show two buttons side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🗑️ Delete All Documents", 
                                   key=f"delete_all_docs_{project.id}",
                                   use_container_width=True,
                                   type="secondary"):
                            st.session_state[confirm_key] = True
                            st.rerun()
                    with col2:
                        confirm_filings_key = f"confirm_delete_all_filings_{project.id}"
                        if st.button("🗑️ Delete All SEC Filings",
                                     key=f"delete_all_filings_{project.id}",
                                     use_container_width=True,
                                     type="secondary"):
                            st.session_state[confirm_filings_key] = True
                            st.rerun()
                elif project.documents:
                    # Only documents exist
                    if st.button("🗑️ Delete All Documents", 
                               key=f"delete_all_docs_{project.id}",
                               use_container_width=True,
                               type="secondary"):
                        st.session_state[confirm_key] = True
                        st.rerun()
                elif project.sec_filings:
                    # Only filings exist
                    confirm_filings_key = f"confirm_delete_all_filings_{project.id}"
                    if st.button("🗑️ Delete All SEC Filings",
                                 key=f"delete_all_filings_{project.id}",
                                 use_container_width=True,
                                 type="secondary"):
                        st.session_state[confirm_filings_key] = True
                        st.rerun()
            
            # Handle filings confirmation
            confirm_filings_key = f"confirm_delete_all_filings_{project.id}"
            if confirm_filings_key in st.session_state and st.session_state[confirm_filings_key]:
                st.warning(f"⚠️ Delete all {len(project.sec_filings)} SEC filing(s)?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Yes, Delete All SEC Filings", key=f"confirm_yes_filings_{project.id}", 
                                use_container_width=True, type="primary"):
                        deleted_count = 0
                        for filing in project.sec_filings[:]:
                            if project_mgr.remove_sec_filing(project.id, filing.accession_number):
                                deleted_count += 1
                        st.session_state[confirm_filings_key] = False
                        st.success(f"✅ Deleted {deleted_count} SEC filing(s)")
                        st.rerun()
                with col2:
                    if st.button("❌ Cancel", key=f"confirm_no_filings_{project.id}", 
                                use_container_width=True):
                        st.session_state[confirm_filings_key] = False
                        st.rerun()
            
            st.divider()
        
        # SEC Filings
        if project.sec_filings:
            st.markdown("*SEC Filings:*")
            for filing in project.sec_filings:
                col1, col2 = st.columns([5, 1])
                with col1:
                    icon = "✓" if filing.indexed else "○"
                    st.caption(f"{icon} {filing.display_name}")
                with col2:
                    if st.button("✕", key=f"remove_filing_{filing.accession_number}", help="Remove"):
                        project_mgr.remove_sec_filing(project.id, filing.accession_number)
                        st.rerun()
        
        # Documents
        if project.documents:
            st.markdown("*Documents:*")
            for doc in project.documents:
                col1, col2, col3 = st.columns([4, 1, 1])
                with col1:
                    icon = "✓" if doc.indexed else "○"
                    size_kb = doc.size_bytes / 1024
                    size_str = f"{size_kb:.1f}KB" if size_kb < 1024 else f"{size_kb/1024:.1f}MB"
                    st.caption(f"{icon} {doc.original_name} ({size_str})")
                with col2:
                    if st.button("👁", key=f"preview_doc_{doc.id}", help="Preview extracted text"):
                        st.session_state[f"show_preview_{doc.id}"] = True
                with col3:
                    if st.button("✕", key=f"remove_doc_{doc.id}", help="Remove"):
                        project_mgr.remove_document(project.id, doc.id)
                        st.rerun()
                
                # Show preview if requested
                if st.session_state.get(f"show_preview_{doc.id}"):
                    with st.expander(f"Preview: {doc.original_name}", expanded=True):
                        doc_path = project_mgr._get_project_documents_dir(project.id) / doc.filename
                        if doc_path.exists():
                            try:
                                file_content = doc_path.read_bytes()
                                processed = doc_processor.process(file_content, doc.original_name)
                                if processed.text:
                                    # Show first 2000 characters
                                    preview_text = processed.text[:2000]
                                    if len(processed.text) > 2000:
                                        preview_text += f"\n\n... ({len(processed.text) - 2000:,} more characters)"
                                    st.text_area(
                                        "Extracted Text",
                                        value=preview_text,
                                        height=200,
                                        disabled=True,
                                        key=f"preview_text_{doc.id}"
                                    )
                                    st.caption(f"Total: {len(processed.text):,} chars, {processed.word_count:,} words")
                                else:
                                    st.warning("No text could be extracted from this document")
                            except Exception as e:
                                st.error(f"Error reading document: {e}")
                        else:
                            st.error("Document file not found on disk")
                        
                        if st.button("Close Preview", key=f"close_preview_{doc.id}"):
                            st.session_state[f"show_preview_{doc.id}"] = False
                            st.rerun()
    
    # Add current SEC filings to project
    if "loaded_filings" in st.session_state and st.session_state.loaded_filings:
        st.divider()
        st.markdown("**Add Loaded Filings to Project**")
        
        for lf in st.session_state.loaded_filings:
            # Check if already in project
            already_added = any(
                f.accession_number == lf.filing_info.accession_number 
                for f in project.sec_filings
            )
            
            if not already_added:
                if st.button(
                    f"Add {lf.ticker} {lf.filing_info.display_name}",
                    key=f"add_filing_{lf.filing_info.accession_number}",
                    use_container_width=True
                ):
                    project_mgr.add_sec_filing(
                        project.id,
                        ticker=lf.ticker,
                        form_type=lf.filing_info.form_type,
                        filing_date=lf.filing_info.filing_date,
                        accession_number=lf.filing_info.accession_number,
                        display_name=lf.filing_info.display_name
                    )
                    st.rerun()


def get_active_project():
    """Get the currently active project."""
    project_mgr = get_project_manager()
    if not project_mgr:
        return None
    
    project_id = st.session_state.get("active_project_id")
    if not project_id:
        return None
    
    return project_mgr.get_project(project_id)


def get_full_document_context(project, threshold_kb: int = 100) -> tuple[str, list[str]]:
    """
    Get full document content for documents under the size threshold.
    
    Args:
        project: The project containing documents
        threshold_kb: Size threshold in KB. Documents larger than this use RAG.
        
    Returns:
        Tuple of (context_string, list_of_included_doc_names)
    """
    if not project or not project.documents:
        return "", []
    
    project_mgr = get_project_manager()
    doc_processor = get_document_processor()
    
    if not project_mgr or not doc_processor:
        return "", []
    
    threshold_bytes = threshold_kb * 1024
    context_parts = []
    included_docs = []
    
    for doc in project.documents:
        # Check if document is under threshold
        if doc.size_bytes <= threshold_bytes:
            # Get document content from disk
            doc_path = project_mgr._get_project_documents_dir(project.id) / doc.filename
            if doc_path.exists():
                try:
                    file_content = doc_path.read_bytes()
                    # Extract text
                    processed = doc_processor.process(file_content, doc.original_name)
                    
                    if processed.text:
                        context_parts.append(f"\n{'='*60}")
                        context_parts.append(f"DOCUMENT: {doc.original_name}")
                        context_parts.append(f"Size: {doc.size_bytes / 1024:.1f} KB")
                        context_parts.append(f"{'='*60}\n")
                        context_parts.append(processed.text)
                        included_docs.append(doc.original_name)
                        logger.info(f"Included full document: {doc.original_name} ({len(processed.text)} chars)")
                except Exception as e:
                    logger.error(f"Error reading document {doc.original_name}: {e}")
        else:
            logger.info(f"Document {doc.original_name} ({doc.size_bytes / 1024:.1f} KB) exceeds threshold, will use RAG")
    
    return "\n".join(context_parts), included_docs


def add_filing_to_active_project(
    ticker: str,
    form_type: str,
    filing_date: str,
    accession_number: str,
    display_name: str
) -> bool:
    """
    Add an SEC filing to the active project.
    
    Returns True if added successfully.
    """
    project_mgr = get_project_manager()
    if not project_mgr:
        return False
    
    project_id = st.session_state.get("active_project_id")
    
    # Auto-create project if needed
    if not project_id:
        project = project_mgr.get_or_create_default_project(ticker)
        st.session_state.active_project_id = project.id
        project_id = project.id
    
    result = project_mgr.add_sec_filing(
        project_id,
        ticker=ticker,
        form_type=form_type,
        filing_date=filing_date,
        accession_number=accession_number,
        display_name=display_name
    )
    
    return result is not None
