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
    
    uploaded_files = st.file_uploader(
        "Drag & drop files",
        type=["pdf", "docx", "doc", "txt", "md", "csv", "xlsx", "xls", "html"],
        accept_multiple_files=True,
        key=f"upload_{project.id}",
        help="Supported: PDF, Word, Text, Markdown, CSV, Excel, HTML"
    )
    
    if uploaded_files:
        doc_processor = get_document_processor()
        
        for uploaded_file in uploaded_files:
            # Check if already processing
            if f"processing_{uploaded_file.name}" in st.session_state:
                continue
            
            if doc_processor and doc_processor.is_supported(uploaded_file.name):
                file_content = uploaded_file.read()
                file_type = doc_processor.get_file_type(uploaded_file.name)
                
                # Add to project
                doc = project_mgr.add_document(
                    project.id,
                    file_content,
                    uploaded_file.name,
                    file_type
                )
                
                if doc:
                    st.success(f"Added: {uploaded_file.name}")
                else:
                    st.error(f"Failed to add: {uploaded_file.name}")
            else:
                st.error(f"Unsupported file type: {uploaded_file.name}")
        
        st.rerun()
    
    # Show project contents
    st.markdown("**Project Contents**")
    
    total_items = len(project.documents) + len(project.sec_filings)
    
    if total_items == 0:
        st.caption("No documents yet. Upload files or add SEC filings.")
    else:
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
                col1, col2 = st.columns([5, 1])
                with col1:
                    icon = "✓" if doc.indexed else "○"
                    size_kb = doc.size_bytes / 1024
                    size_str = f"{size_kb:.1f}KB" if size_kb < 1024 else f"{size_kb/1024:.1f}MB"
                    st.caption(f"{icon} {doc.original_name} ({size_str})")
                with col2:
                    if st.button("✕", key=f"remove_doc_{doc.id}", help="Remove"):
                        project_mgr.remove_document(project.id, doc.id)
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
