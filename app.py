"""
Financial Research Agent - Main Streamlit Application

A comprehensive financial research tool combining SEC filing analysis,
price data analysis, and web research in a unified conversational interface.

Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import sys
import logging

# Add project root to Python path to fix relative imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Financial Research Agent",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import application modules
from config import get_config, update_config
from core.agent import create_agent, FinancialResearchAgent
from core.conversation import ConversationManager
from ui.sidebar import render_sidebar, SidebarConfig
from ui.chat_interface import render_chat_interface
from ui.price_analysis_tab import render_price_analysis_tab


# Custom CSS
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: bold;
    }

    /* Tool call expander */
    .streamlit-expanderHeader {
        font-size: 0.9rem;
    }

    /* Sidebar styling - improved contrast */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
    }
    
    /* Sidebar headers - dark, readable text */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #262730 !important;
    }
    
    /* Sidebar markdown content */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] .stMarkdown strong {
        color: #262730 !important;
    }
    
    /* Sidebar input labels - ensure visibility */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label p {
        color: #262730 !important;
        font-weight: 500;
    }
    
    /* Sidebar selectbox and text input labels */
    [data-testid="stSidebar"] [data-baseweb="select"] label,
    [data-testid="stSidebar"] [data-baseweb="input"] label {
        color: #262730 !important;
    }
    
    /* Sidebar help text - slightly lighter but still readable */
    [data-testid="stSidebar"] [data-testid="stTooltipIcon"] {
        color: #4a5568 !important;
    }
    
    /* Sidebar expander headers */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        color: #262730 !important;
        font-weight: 500;
    }
    
    /* Sidebar expander content */
    [data-testid="stSidebar"] .streamlit-expanderContent {
        color: #262730 !important;
    }
    
    /* Sidebar divider */
    [data-testid="stSidebar"] hr {
        border-color: #e0e0e0;
    }
    
    /* Preserve status message colors (success, warning, info, error) */
    [data-testid="stSidebar"] .stSuccess,
    [data-testid="stSidebar"] .stWarning,
    [data-testid="stSidebar"] .stInfo,
    [data-testid="stSidebar"] .stError {
        /* Keep original colors for status messages */
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
    }

    /* Chat message content - improved readability (theme-aware) */
    .stChatMessage [data-testid="stMarkdownContainer"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        font-size: 1rem;
        line-height: 1.6;
    }

    /* Chat message paragraphs */
    .stChatMessage [data-testid="stMarkdownContainer"] p {
        margin-bottom: 0.75rem;
    }

    /* Chat message headers */
    .stChatMessage [data-testid="stMarkdownContainer"] h1,
    .stChatMessage [data-testid="stMarkdownContainer"] h2,
    .stChatMessage [data-testid="stMarkdownContainer"] h3 {
        margin-top: 1.25rem;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }

    .stChatMessage [data-testid="stMarkdownContainer"] h1 { font-size: 1.5rem; }
    .stChatMessage [data-testid="stMarkdownContainer"] h2 { font-size: 1.3rem; }
    .stChatMessage [data-testid="stMarkdownContainer"] h3 { font-size: 1.15rem; }

    /* Chat message lists */
    .stChatMessage [data-testid="stMarkdownContainer"] ul,
    .stChatMessage [data-testid="stMarkdownContainer"] ol {
        margin-left: 1.5rem;
        margin-bottom: 0.75rem;
        padding-left: 0;
    }

    .stChatMessage [data-testid="stMarkdownContainer"] li {
        margin-bottom: 0.4rem;
        line-height: 1.5;
    }

    /* Chat message code blocks - theme aware */
    .stChatMessage [data-testid="stMarkdownContainer"] pre {
        background-color: rgba(128, 128, 128, 0.1);
        border-radius: 6px;
        padding: 1rem;
        margin: 0.75rem 0;
        overflow-x: auto;
        font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
        font-size: 0.9rem;
        line-height: 1.4;
    }

    .stChatMessage [data-testid="stMarkdownContainer"] code {
        background-color: rgba(128, 128, 128, 0.1);
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
        font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
        font-size: 0.9em;
    }

    /* Inline code inside pre should not have extra background */
    .stChatMessage [data-testid="stMarkdownContainer"] pre code {
        background-color: transparent;
        padding: 0;
    }

    /* Chat message blockquotes - for citations */
    .stChatMessage [data-testid="stMarkdownContainer"] blockquote {
        border-left: 4px solid #4a90d9;
        margin: 0.75rem 0;
        padding: 0.5rem 1rem;
        background-color: rgba(74, 144, 217, 0.1);
        font-style: normal;
    }

    /* Chat message tables */
    .stChatMessage [data-testid="stMarkdownContainer"] table {
        border-collapse: collapse;
        margin: 0.75rem 0;
        width: 100%;
        font-size: 0.95rem;
    }

    .stChatMessage [data-testid="stMarkdownContainer"] th,
    .stChatMessage [data-testid="stMarkdownContainer"] td {
        border: 1px solid rgba(128, 128, 128, 0.3);
        padding: 0.5rem 0.75rem;
        text-align: left;
    }

    .stChatMessage [data-testid="stMarkdownContainer"] th {
        background-color: rgba(128, 128, 128, 0.1);
        font-weight: 600;
    }

    .stChatMessage [data-testid="stMarkdownContainer"] tr:nth-child(even) {
        background-color: rgba(128, 128, 128, 0.05);
    }

    /* Bold and emphasis - inherit color from theme */
    .stChatMessage [data-testid="stMarkdownContainer"] strong {
        font-weight: 600;
    }

    .stChatMessage [data-testid="stMarkdownContainer"] em {
        font-style: italic;
    }

    /* Links in chat */
    .stChatMessage [data-testid="stMarkdownContainer"] a {
        color: #4a90d9;
        text-decoration: none;
    }

    .stChatMessage [data-testid="stMarkdownContainer"] a:hover {
        text-decoration: underline;
    }

    /* Horizontal rule - theme aware */
    .stChatMessage [data-testid="stMarkdownContainer"] hr {
        border: none;
        border-top: 1px solid rgba(128, 128, 128, 0.3);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize all session state variables."""
    if "conversation_manager" not in st.session_state:
        cache_dir = Path(__file__).parent / "cache" / "conversations"
        st.session_state.conversation_manager = ConversationManager(cache_dir)
        st.session_state.conversation_manager.load_all_sessions()

    if "agent" not in st.session_state:
        st.session_state.agent = None

    if "config" not in st.session_state:
        st.session_state.config = None

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []


def create_or_update_agent(config: SidebarConfig) -> FinancialResearchAgent | None:
    """Create or update the agent based on config."""
    if not config.is_configured():
        return None

    # Check if we need to recreate the agent
    current_config = st.session_state.config
    if current_config is not None:
        same_provider = current_config.llm_provider == config.llm_provider
        same_key = current_config.api_key == config.api_key
        same_model = current_config.model == config.model

        if same_provider and same_key and same_model and st.session_state.agent:
            return st.session_state.agent

    # Create new agent
    try:
        agent = create_agent(
            provider=config.llm_provider,
            api_key=config.api_key,
            model=config.model,
            anthropic_key=config.api_key if config.llm_provider == "claude" else None,
            newsapi_key=config.newsapi_key
        )
        st.session_state.config = config
        st.session_state.agent = agent
        return agent
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        st.error(f"Failed to initialize agent: {e}")
        return None


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()

    # Render sidebar and get config
    config = render_sidebar()

    # Create/update agent
    agent = create_or_update_agent(config)

    # Main content
    st.title(":chart_with_upwards_trend: Financial Research Agent")
    st.markdown("*SEC filings, price analysis, and news research in one place*")

    # Create tabs
    tab_chat, tab_price = st.tabs([
        ":speech_balloon: Research Chat",
        ":chart: Price Analysis"
    ])

    # Research Chat Tab
    with tab_chat:
        render_chat_interface(
            agent=agent,
            conversation_manager=st.session_state.conversation_manager,
            ticker=config.ticker
        )

    # Price Analysis Tab
    with tab_price:
        render_price_analysis_tab(
            ticker=config.ticker,
            enable_upload=True,
            enable_yfinance=config.enable_price
        )

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.8rem;'>"
        "Financial Research Agent | SEC data from EDGAR | Price data from Yahoo Finance"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
