"""
Price Analysis Tab UI - Migrated from original stock analyzer app.

Provides interactive charts and price event analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from data.price_analyzer import StockPriceAnalyzer, AnalysisResults
from io import BytesIO


def create_combined_export(results: AnalysisResults, df: pd.DataFrame) -> BytesIO:
    """
    Create a combined Excel export with all analysis data.

    Returns:
        BytesIO buffer containing the Excel file
    """
    output = BytesIO()

    # Prepare statistics data
    stats = results.statistics
    ath = results.all_time_high
    atl = results.all_time_low
    ath_date = ath.date.strftime('%Y-%m-%d') if hasattr(ath.date, 'strftime') else str(ath.date)
    atl_date = atl.date.strftime('%Y-%m-%d') if hasattr(atl.date, 'strftime') else str(atl.date)

    stats_data = {
        "Metric": [
            "Ticker",
            "All-Time High",
            "All-Time High Date",
            "All-Time Low",
            "All-Time Low Date",
            "Mean Daily Return (%)",
            "Std Dev Daily Return (%)",
            "Max Daily Gain (%)",
            "Max Daily Loss (%)",
            "Annualized Return (%)",
            "Annualized Volatility (%)",
            "Total Trading Days",
            "Up Days",
            "Down Days",
            "Win Rate (%)",
            "Start Price",
            "End Price",
            "Total Return (%)"
        ],
        "Value": [
            results.ticker,
            ath.price,
            ath_date,
            atl.price,
            atl_date,
            stats.mean_daily_return,
            stats.std_daily_return,
            stats.max_daily_gain,
            stats.max_daily_loss,
            stats.annualized_return if stats.annualized_return else "N/A",
            stats.annualized_volatility if stats.annualized_volatility else "N/A",
            stats.total_days,
            stats.positive_days,
            stats.negative_days,
            stats.positive_days / stats.total_days * 100,
            stats.start_price,
            stats.end_price,
            stats.total_return
        ]
    }
    stats_df = pd.DataFrame(stats_data)

    # Prepare rallies data
    rallies_data = []
    for i, rally in enumerate(results.top_rallies, 1):
        date_str = rally.date.strftime('%Y-%m-%d') if hasattr(rally.date, 'strftime') else str(rally.date)
        rallies_data.append({
            "Rank": i,
            "Date": date_str,
            "Close": rally.price,
            "Previous_Close": rally.prev_price,
            "Pct_Change": rally.pct_change
        })
    rallies_df = pd.DataFrame(rallies_data)

    # Prepare drawdowns data
    drawdowns_data = []
    for i, dd in enumerate(results.top_drawdowns, 1):
        date_str = dd.date.strftime('%Y-%m-%d') if hasattr(dd.date, 'strftime') else str(dd.date)
        drawdowns_data.append({
            "Rank": i,
            "Date": date_str,
            "Close": dd.price,
            "Previous_Close": dd.prev_price,
            "Pct_Change": dd.pct_change
        })
    drawdowns_df = pd.DataFrame(drawdowns_data)

    # Prepare price data
    cols_to_export = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Pct_Change']
    cols_available = [c for c in cols_to_export if c in df.columns]
    price_df = df[cols_available].copy()

    # Write to Excel with multiple sheets
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        rallies_df.to_excel(writer, sheet_name='Top Rallies', index=False)
        drawdowns_df.to_excel(writer, sheet_name='Top Drawdowns', index=False)
        price_df.to_excel(writer, sheet_name='Price History', index=False)

    output.seek(0)
    return output


def render_export_button(results: AnalysisResults, df: pd.DataFrame, key: str = "export") -> None:
    """Render the combined export button."""
    export_data = create_combined_export(results, df)
    st.download_button(
        "Export Analysis (Excel)",
        data=export_data,
        file_name=f"{results.ticker}_price_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key
    )


def render_price_analysis_tab(
    ticker: str = "",
    enable_upload: bool = True,
    enable_yfinance: bool = True
) -> Optional[AnalysisResults]:
    """
    Render the price analysis tab.

    Args:
        ticker: Pre-filled ticker symbol
        enable_upload: Allow file upload
        enable_yfinance: Allow fetching from Yahoo Finance

    Returns:
        AnalysisResults if analysis was performed
    """
    if not PLOTLY_AVAILABLE:
        st.error("Plotly is required for price analysis. Install with: pip install plotly")
        return None

    st.subheader("Price History Analysis")

    # Data source selection
    col1, col2 = st.columns([2, 1])

    with col1:
        data_source = st.radio(
            "Data Source",
            options=["Yahoo Finance", "Upload File"] if enable_yfinance and enable_upload
                    else ["Yahoo Finance"] if enable_yfinance
                    else ["Upload File"],
            horizontal=True
        )

    with col2:
        num_events = st.slider("Events to show", 3, 10, 5)

    analyzer = None
    results = None

    if data_source == "Yahoo Finance":
        analyzer, results = render_yfinance_input(ticker, num_events)
    else:
        analyzer, results = render_file_upload_input(ticker, num_events)

    if analyzer is not None and results is not None:
        render_analysis_results(analyzer, results)

    return results


def render_yfinance_input(
    default_ticker: str,
    num_events: int
) -> tuple:
    """Render Yahoo Finance data input."""
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        ticker = st.text_input(
            "Ticker Symbol",
            value=default_ticker,
            placeholder="e.g., AAPL"
        ).upper().strip()

    with col2:
        period = st.selectbox(
            "Period",
            options=["1y", "2y", "5y", "10y", "max"],
            index=2  # 5y default
        )

    with col3:
        if st.button("Analyze", type="primary", disabled=not ticker):
            with st.spinner(f"Fetching {ticker} data..."):
                try:
                    analyzer = StockPriceAnalyzer.from_yfinance(ticker, period=period)
                    results = analyzer.run_full_analysis(num_events)
                    st.session_state.price_analyzer = analyzer
                    st.session_state.price_results = results
                    return analyzer, results
                except Exception as e:
                    st.error(f"Error fetching data: {e}")
                    return None, None

    # Return cached results if available
    if "price_analyzer" in st.session_state:
        return st.session_state.price_analyzer, st.session_state.price_results

    return None, None


def render_file_upload_input(
    default_ticker: str,
    num_events: int
) -> tuple:
    """Render file upload input."""
    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload price history",
            type=['csv', 'xlsx', 'xls'],
            help="File must have 'Date' and 'Close' columns"
        )

    with col2:
        ticker = st.text_input(
            "Ticker",
            value=default_ticker or "STOCK",
            placeholder="Ticker symbol"
        ).upper()

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            if 'Date' not in df.columns or 'Close' not in df.columns:
                st.error("File must contain 'Date' and 'Close' columns")
                return None, None

            analyzer = StockPriceAnalyzer(df=df, ticker=ticker)
            results = analyzer.run_full_analysis(num_events)
            return analyzer, results

        except Exception as e:
            st.error(f"Error reading file: {e}")

    return None, None


def render_analysis_results(
    analyzer: StockPriceAnalyzer,
    results: AnalysisResults
) -> None:
    """Render analysis results with charts and data."""
    # Key metrics
    st.markdown("### Key Metrics")
    render_key_metrics(results)

    # Tabs for different views
    tabs = st.tabs(["Price Chart", "Rallies & Drawdowns", "Statistics", "Raw Data", "Ask AI"])

    with tabs[0]:
        render_price_chart(analyzer.df, results)

    with tabs[1]:
        render_events_list(results, analyzer.df)

    with tabs[2]:
        render_statistics(results, analyzer.df)

    with tabs[3]:
        render_raw_data(analyzer.df, results)

    with tabs[4]:
        render_price_chat(analyzer, results)


def render_key_metrics(results: AnalysisResults) -> None:
    """Render key metrics as metric cards."""
    cols = st.columns(4)

    ath = results.all_time_high
    atl = results.all_time_low
    stats = results.statistics

    with cols[0]:
        ath_date = ath.date.strftime('%Y-%m-%d') if hasattr(ath.date, 'strftime') else str(ath.date)
        st.metric(
            "All-Time High",
            f"${ath.price:,.2f}",
            ath_date
        )

    with cols[1]:
        atl_date = atl.date.strftime('%Y-%m-%d') if hasattr(atl.date, 'strftime') else str(atl.date)
        st.metric(
            "All-Time Low",
            f"${atl.price:,.2f}",
            atl_date
        )

    with cols[2]:
        st.metric(
            "Total Return",
            f"{stats.total_return:+,.1f}%",
            f"{stats.total_days:,} days"
        )

    with cols[3]:
        win_rate = stats.positive_days / stats.total_days * 100
        st.metric(
            "Win Rate",
            f"{win_rate:.1f}%",
            f"{stats.positive_days:,} up days"
        )


def render_price_chart(df: pd.DataFrame, results: AnalysisResults) -> None:
    """Render interactive price chart with events marked."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f"{results.ticker} Price History", "Daily % Change"),
        row_heights=[0.7, 0.3]
    )

    # Main price line
    fig.add_trace(
        go.Scatter(
            x=df['Date'], y=df['Close'],
            mode='lines', name='Close Price',
            line=dict(color='#1976D2', width=1.5)
        ),
        row=1, col=1
    )

    # All-time high marker
    ath = results.all_time_high
    fig.add_trace(
        go.Scatter(
            x=[ath.date], y=[ath.price],
            mode='markers', name='All-Time High',
            marker=dict(color='#4CAF50', size=15, symbol='star'),
            hovertemplate=f"<b>ATH</b><br>${ath.price:.2f}<extra></extra>"
        ),
        row=1, col=1
    )

    # All-time low marker
    atl = results.all_time_low
    fig.add_trace(
        go.Scatter(
            x=[atl.date], y=[atl.price],
            mode='markers', name='All-Time Low',
            marker=dict(color='#F44336', size=15, symbol='star'),
            hovertemplate=f"<b>ATL</b><br>${atl.price:.2f}<extra></extra>"
        ),
        row=1, col=1
    )

    # Rally markers
    anomaly_dates = [a['date'] for a in results.anomalies]
    for rally in results.top_rallies:
        if rally.date not in anomaly_dates:
            fig.add_trace(
                go.Scatter(
                    x=[rally.date], y=[rally.price],
                    mode='markers',
                    marker=dict(color='#4CAF50', size=10, symbol='triangle-up'),
                    showlegend=False,
                    hovertemplate=f"<b>Rally</b><br>+{rally.pct_change:.2f}%<extra></extra>"
                ),
                row=1, col=1
            )

    # Drawdown markers
    for dd in results.top_drawdowns:
        if dd.date not in anomaly_dates:
            fig.add_trace(
                go.Scatter(
                    x=[dd.date], y=[dd.price],
                    mode='markers',
                    marker=dict(color='#F44336', size=10, symbol='triangle-down'),
                    showlegend=False,
                    hovertemplate=f"<b>Drawdown</b><br>{dd.pct_change:.2f}%<extra></extra>"
                ),
                row=1, col=1
            )

    # Daily change bars
    colors = ['#4CAF50' if x >= 0 else '#F44336' for x in df['Daily_Pct_Change'].fillna(0)]
    fig.add_trace(
        go.Bar(
            x=df['Date'], y=df['Daily_Pct_Change'],
            marker_color=colors, showlegend=False, name='Daily %'
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=650,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="% Change", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Distribution histogram
    render_returns_histogram(df)


def render_returns_histogram(df: pd.DataFrame) -> None:
    """Render histogram of daily returns."""
    fig = px.histogram(
        df, x='Daily_Pct_Change', nbins=100,
        title='Distribution of Daily Returns',
        color_discrete_sequence=['#1976D2']
    )

    mean = df['Daily_Pct_Change'].mean()
    std = df['Daily_Pct_Change'].std()

    fig.add_vline(x=mean, line_dash="dash", line_color="green",
                  annotation_text=f"Mean: {mean:.2f}%")
    fig.add_vline(x=mean + 2*std, line_dash="dot", line_color="orange",
                  annotation_text="+2s")
    fig.add_vline(x=mean - 2*std, line_dash="dot", line_color="orange",
                  annotation_text="-2s")

    fig.update_layout(height=350, xaxis_title="Daily % Change", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)


def render_events_list(results: AnalysisResults, df: pd.DataFrame) -> None:
    """Render list of rallies and drawdowns."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Top Rallies")
        for i, rally in enumerate(results.top_rallies, 1):
            with st.container():
                date_str = rally.date.strftime('%Y-%m-%d') if hasattr(rally.date, 'strftime') else str(rally.date)
                st.markdown(f"""
                **#{i}** - {date_str}

                Close: ${rally.price:,.2f} | Change: **+{rally.pct_change:.2f}%**
                """)
                st.divider()

    with col2:
        st.markdown("### Top Drawdowns")
        for i, dd in enumerate(results.top_drawdowns, 1):
            with st.container():
                date_str = dd.date.strftime('%Y-%m-%d') if hasattr(dd.date, 'strftime') else str(dd.date)
                st.markdown(f"""
                **#{i}** - {date_str}

                Close: ${dd.price:,.2f} | Change: **{dd.pct_change:.2f}%**
                """)
                st.divider()

    # Combined export button
    render_export_button(results, df, key="export_events")


def render_statistics(results: AnalysisResults, df: pd.DataFrame) -> None:
    """Render statistical summary."""
    stats = results.statistics

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Returns")
        st.metric("Mean Daily Return", f"{stats.mean_daily_return:.3f}%")
        st.metric("Best Day", f"+{stats.max_daily_gain:.2f}%")
        st.metric("Worst Day", f"{stats.max_daily_loss:.2f}%")
        if stats.annualized_return:
            st.metric("Annualized Return", f"{stats.annualized_return:.1f}%")

    with col2:
        st.markdown("### Volatility")
        st.metric("Daily Std Dev", f"{stats.std_daily_return:.3f}%")
        if stats.annualized_volatility:
            st.metric("Annualized Vol", f"{stats.annualized_volatility:.1f}%")

    with col3:
        st.markdown("### Period")
        st.metric("Trading Days", f"{stats.total_days:,}")
        st.metric("Up Days", f"{stats.positive_days:,}")
        st.metric("Down Days", f"{stats.negative_days:,}")

    # Combined export button
    render_export_button(results, df, key="export_stats")


def render_raw_data(df: pd.DataFrame, results: AnalysisResults) -> None:
    """Render raw data table."""
    cols_to_show = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Pct_Change']
    cols_available = [c for c in cols_to_show if c in df.columns]

    st.dataframe(
        df[cols_available].tail(100),
        use_container_width=True,
        hide_index=True
    )

    # Combined export button
    render_export_button(results, df, key="export_raw")


def render_price_chat(analyzer: StockPriceAnalyzer, results: AnalysisResults) -> None:
    """Render AI chat interface for asking questions about price analysis."""
    from data.news_search import NewsSearchManager

    st.markdown("### Ask AI About This Analysis")
    st.info("Ask questions about the price analysis. The AI can search the web for news context around major price events.")

    # Store analyzer and results in session state for chat context
    st.session_state.price_analyzer = analyzer
    st.session_state.price_results = results

    # Initialize chat history for price analysis
    if "price_chat_messages" not in st.session_state:
        st.session_state.price_chat_messages = []

    # Display chat history
    for msg in st.session_state.price_chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Suggested questions (only show if no messages yet)
    if not st.session_state.price_chat_messages:
        st.markdown("**Suggested questions:**")
        suggested = [
            "What news events caused the biggest rallies?",
            "What happened during the largest drawdowns?",
            "Is this stock more volatile than average?",
            "Summarize the key price events and their causes"
        ]

        cols = st.columns(2)
        for i, q in enumerate(suggested):
            with cols[i % 2]:
                if st.button(q, key=f"suggest_{i}", use_container_width=True):
                    st.session_state.price_pending_question = q
                    st.rerun()

    # Check for pending question from buttons
    if "price_pending_question" in st.session_state and st.session_state.price_pending_question:
        question = st.session_state.price_pending_question
        st.session_state.price_pending_question = None
    else:
        question = None

    # Chat input - always show it
    user_input = st.chat_input("Ask about this price analysis...", key="price_chat_input")

    if user_input or question:
        prompt = user_input or question

        # Add user message immediately
        st.session_state.price_chat_messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check if agent is available
        if "agent" not in st.session_state or st.session_state.agent is None:
            with st.chat_message("assistant"):
                error_msg = "Please configure an API key in the sidebar to use the AI chat feature."
                st.error(error_msg)
                st.session_state.price_chat_messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
            st.rerun()
            return

        agent = st.session_state.agent

        # Build context from analysis results AND actual price data
        stats = results.statistics
        ath = results.all_time_high
        atl = results.all_time_low

        ath_date = ath.date.strftime('%Y-%m-%d') if hasattr(ath.date, 'strftime') else str(ath.date)
        atl_date = atl.date.strftime('%Y-%m-%d') if hasattr(atl.date, 'strftime') else str(atl.date)

        rallies_str = "\n".join([
            f"  {i+1}. {r.date.strftime('%Y-%m-%d') if hasattr(r.date, 'strftime') else r.date}: +{r.pct_change:.2f}% (${r.price:.2f})"
            for i, r in enumerate(results.top_rallies)
        ])

        drawdowns_str = "\n".join([
            f"  {i+1}. {d.date.strftime('%Y-%m-%d') if hasattr(d.date, 'strftime') else d.date}: {d.pct_change:.2f}% (${d.price:.2f})"
            for i, d in enumerate(results.top_drawdowns)
        ])

        # Include sample of actual price data
        df = analyzer.df
        recent_data = df.tail(20).to_dict('records') if len(df) > 0 else []
        recent_data_str = "\n".join([
            f"  {row.get('Date', 'N/A')}: Close=${row.get('Close', 0):.2f}, Change={row.get('Daily_Pct_Change', 0):.2f}%"
            for row in recent_data[-10:]
        ])

        first_few = df.head(5).to_dict('records') if len(df) > 0 else []
        first_few_str = "\n".join([
            f"  {row.get('Date', 'N/A')}: Close=${row.get('Close', 0):.2f}"
            for row in first_few
        ])

        # Get response from agent with web search
        with st.chat_message("assistant"):
            with st.spinner("Searching for news context..."):
                # Search for news around price events
                news_context = ""
                try:
                    news_manager = NewsSearchManager()

                    # Check if the question is about news/events
                    news_keywords = ["news", "event", "happened", "caused", "why", "what", "reason", "explain"]
                    should_search = any(kw in prompt.lower() for kw in news_keywords)

                    if should_search:
                        # Search for news around top rallies and drawdowns
                        event_news = news_manager.search_for_price_events(
                            ticker=results.ticker,
                            rallies=results.top_rallies,
                            drawdowns=results.top_drawdowns,
                            max_events=3
                        )

                        # Format rally news
                        if event_news.get("rally_news"):
                            news_context += "\n**News Context for Major Rallies:**\n"
                            for event in event_news["rally_news"]:
                                news_context += f"\n*{event['date']} (+{event['pct_change']:.2f}%):*\n"
                                for news in event["news"][:2]:
                                    headline = news.get("headline", "")
                                    summary = news.get("summary", "")[:200]
                                    url = news.get("url", "")
                                    news_context += f"  - {headline}\n"
                                    if summary:
                                        news_context += f"    {summary}...\n"
                                    if url:
                                        news_context += f"    URL: {url}\n"

                        # Format drawdown news
                        if event_news.get("drawdown_news"):
                            news_context += "\n**News Context for Major Drawdowns:**\n"
                            for event in event_news["drawdown_news"]:
                                news_context += f"\n*{event['date']} ({event['pct_change']:.2f}%):*\n"
                                for news in event["news"][:2]:
                                    headline = news.get("headline", "")
                                    summary = news.get("summary", "")[:200]
                                    url = news.get("url", "")
                                    news_context += f"  - {headline}\n"
                                    if summary:
                                        news_context += f"    {summary}...\n"
                                    if url:
                                        news_context += f"    URL: {url}\n"

                        if news_context:
                            st.toast("Found news context for price events", icon="🔍")

                except Exception as e:
                    logger.error(f"News search error: {e}")

            with st.spinner("Generating response..."):
                context = f"""Here is the complete price analysis data for {results.ticker}:

**Summary Statistics:**
- Data Range: {results.data_range.get('start_date', 'N/A')} to {results.data_range.get('end_date', 'N/A')}
- Total Data Points: {len(df):,} trading days
- All-Time High: ${ath.price:.2f} on {ath_date}
- All-Time Low: ${atl.price:.2f} on {atl_date}
- Total Return: {stats.total_return:.2f}%
- Annualized Return: {stats.annualized_return:.2f}%
- Mean Daily Return: {stats.mean_daily_return:.4f}%
- Daily Volatility (Std Dev): {stats.std_daily_return:.4f}%
- Annualized Volatility: {stats.annualized_volatility:.2f}%
- Best Single Day: +{stats.max_daily_gain:.2f}%
- Worst Single Day: {stats.max_daily_loss:.2f}%
- Up Days: {stats.positive_days:,} ({stats.positive_days/stats.total_days*100:.1f}%)
- Down Days: {stats.negative_days:,} ({stats.negative_days/stats.total_days*100:.1f}%)

**Top Rallies (Biggest Up Days):**
{rallies_str}

**Top Drawdowns (Biggest Down Days):**
{drawdowns_str}
{news_context}
**Sample Price Data (First 5 days):**
{first_few_str}

**Sample Price Data (Last 10 days):**
{recent_data_str}

Based on this price analysis data{' and the web search results above' if news_context else ''}, please answer:
{prompt}

When citing news sources, include the URL if available."""

                try:
                    response = agent.chat(context)
                    st.markdown(response.content)
                    st.session_state.price_chat_messages.append({
                        "role": "assistant",
                        "content": response.content
                    })
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.price_chat_messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

    # Clear chat button
    if st.session_state.price_chat_messages:
        if st.button("Clear Chat", key="clear_price_chat"):
            st.session_state.price_chat_messages = []
            st.rerun()
