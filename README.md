# Financial Research Agent

A comprehensive financial research tool combining SEC filing analysis, stock price data analysis, and web research in a unified conversational interface powered by AI.

## Features

- **Projects**: Organize research into projects with SEC filings and uploaded documents (similar to Claude Desktop)
- **SEC Filing Analysis**: Extract and analyze information from 10-K, 10-Q, 8-K, and other SEC filings
- **Document Upload**: Drag-and-drop PDFs, Word docs, spreadsheets, and more into projects
- **Price Analysis**: Interactive charts, identify significant price events (all-time highs/lows, rallies, drawdowns)
- **News Search**: Search for relevant news articles using multiple backends (DuckDuckGo free search, Anthropic Claude, NewsAPI, cached historical data)
- **Conversational AI**: Ask questions about companies, financials, and market events using Claude, OpenAI, or Google Gemini
- **RAG (Retrieval-Augmented Generation)**: Semantic search over SEC filings and uploaded documents using ChromaDB
- **Secure API Key Storage**: Keys persist across browser sessions using OS keychain or encrypted local storage
- **Pre-built Workflows**: Automated analysis workflows for financial health, risk analysis, and competitive comparison

## Installation

1. Clone this repository:
```bash
git clone https://github.com/jingerzz/financial-research-agent.git
cd financial-research-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys (optional):
   - Create a `.streamlit/secrets.toml` file (see `.streamlit/secrets.toml.example`)
   - Or enter API keys directly in the Streamlit sidebar when running the app

4. (Recommended) Update SEC EDGAR user agent in `config.py`:
   - The SEC requires identification when accessing EDGAR
   - Change `user_agent` from `research@example.com` to your email

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Getting Started

1. **Configure API Key**: In the sidebar, select your LLM provider (Anthropic, OpenAI, or Google Gemini) and enter your API key. Click **Save Key** to persist it.
2. **Enter Ticker**: Type a stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
3. **Create or Select a Project**: Projects help organize your research. Add SEC filings and upload documents.
4. **Use the Interface**:
   - **Research Chat**: Ask questions about companies and financials
   - **Workflows**: Run pre-built analysis workflows
   - **Price Analysis**: View charts and analyze price history

### Example Questions

- "What are AAPL's main risk factors?"
- "Summarize MSFT's latest 10-K filing"
- "Compare AAPL, MSFT, and GOOGL"
- "Why did TSLA stock drop on [date]?"
- "What patterns do you see in the price rallies and drawdowns?"

## Project Structure

```
financial-research-agent/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── core/
│   ├── agent.py          # Main agent orchestrator
│   ├── conversation.py   # Conversation management
│   ├── credential_manager.py # Secure API key storage
│   ├── llm_provider.py   # Multi-LLM provider abstraction
│   └── tools.py          # Tool definitions and executor
├── data/
│   ├── sec_edgar.py      # SEC EDGAR filing retrieval
│   ├── price_analyzer.py # Stock price analysis
│   ├── news_search.py    # News search backends
│   ├── rag_manager.py    # RAG vector storage with ChromaDB
│   ├── chunking.py       # SEC filing chunking for RAG
│   ├── project_manager.py # Project CRUD and document management
│   └── document_processor.py # Text extraction from uploaded files
├── ui/
│   ├── chat_interface.py      # Chat UI component
│   ├── price_analysis_tab.py  # Price analysis UI
│   ├── workflows_tab.py       # Workflows UI
│   ├── sidebar.py             # Sidebar configuration
│   └── projects_panel.py      # Projects management UI
└── workflows/
    ├── base.py                # Base workflow class
    ├── financial_health.py    # Financial health analysis
    ├── risk_analysis.py       # Risk factor analysis
    └── competitive_comparison.py # Competitive analysis
```

## Requirements

- Python 3.8+
- API keys for:
  - Anthropic Claude (optional, for Claude AI)
  - OpenAI (optional, for GPT models)
  - Google AI (optional, for Gemini models)
  - NewsAPI (optional, for enhanced news search)
- Note: DuckDuckGo search works without any API key

## Features in Detail

### Projects
- Create named projects to organize research (e.g., "AAPL Deep Dive", "Tech Sector Analysis")
- Add SEC filings from EDGAR and uploaded documents to projects
- Drag-and-drop file upload for PDFs, Word documents, spreadsheets, and more
- All documents are indexed for semantic search
- Auto-create default project when loading a ticker
- Projects persist across sessions

### Secure API Key Storage
- Keys persist across browser refreshes (no more re-entering!)
- Storage priority: Environment vars > secrets.toml > OS Keychain > Encrypted file
- macOS: Uses Keychain
- Windows: Uses Credential Manager
- Linux: Uses Secret Service (GNOME Keyring, KWallet)
- Fallback: AES-256 encrypted local file
- Multi-user support (keys stored per-OS-user)

### SEC Filing Analysis
- Automatic retrieval of 10-K, 10-Q, 8-K, and other SEC filings
- Intelligent parsing and extraction of key information
- Risk factor identification
- Financial statement analysis

### Price Analysis
- Interactive Plotly charts
- Identification of significant events (all-time highs/lows, rallies, drawdowns)
- Statistical analysis (volatility, returns, win rates)
- Support for uploaded CSV/Excel files or Yahoo Finance data

### News Search
- Multiple backend support:
  - DuckDuckGo (free, no API key required - default)
  - Anthropic Claude with web search
  - OpenAI (uses training data for historical context)
  - NewsAPI.org
  - Cached historical market events
- Automatic fallback to available backends

### AI Chat
- Conversational interface for asking questions
- Context-aware responses using SEC filings, price data, and news
- Support for follow-up questions
- Tool visibility for transparency
- Multi-provider support: Anthropic Claude, OpenAI GPT, and Google Gemini
- **Custom system prompt**: Add your own instructions to customize the AI's behavior

### RAG (Retrieval-Augmented Generation)
- Semantic search over SEC filings and uploaded project documents
- **Hybrid mode**: Small documents (<1.5MB) are included in full; large files use RAG chunking
- ChromaDB vector database for fast similarity search
- Automatic chunking of filings by SEC sections (Item 1, 1A, 7, etc.)
- Sentence-transformers for local embeddings (no API calls required)
- Intelligent context retrieval for more accurate responses

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This tool is for research and educational purposes only. It is not intended as financial advice. Always consult with qualified financial professionals before making investment decisions.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Uses [edgartools](https://github.com/dgunning/edgartools) for SEC EDGAR access
- Uses [yfinance](https://github.com/ranaroussi/yfinance) for stock data
- Uses [ChromaDB](https://www.trychroma.com/) for vector storage
- Uses [sentence-transformers](https://www.sbert.net/) for embeddings
- Powered by [Anthropic Claude](https://www.anthropic.com/), [OpenAI](https://openai.com/), and [Google Gemini](https://ai.google.dev/)
