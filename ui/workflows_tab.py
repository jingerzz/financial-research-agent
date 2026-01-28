"""
Workflows Tab UI for Financial Research Agent.

Provides pre-built analysis workflows with progress tracking.
"""

import streamlit as st
from typing import Any, Callable, Dict, List, Optional

from core.agent import FinancialResearchAgent


def render_workflows_tab(
    agent: Optional[FinancialResearchAgent],
    ticker: str = ""
) -> None:
    """
    Render the workflows tab with pre-built analysis options.

    Args:
        agent: The configured agent
        ticker: Current ticker symbol
    """
    st.subheader("Pre-built Analysis Workflows")

    if not agent:
        st.warning("Configure an API key in the sidebar to run workflows.")
        return

    if not ticker:
        st.info("Enter a ticker symbol in the sidebar to run workflows.")
        return

    # Workflow cards
    col1, col2 = st.columns(2)

    with col1:
        render_workflow_card(
            title="Financial Health Analysis",
            description="Comprehensive analysis of financial statements including income, balance sheet, and cash flow. Calculates key ratios and identifies trends.",
            icon="chart_with_upwards_trend",
            workflow_id="financial_health",
            agent=agent,
            ticker=ticker
        )

        render_workflow_card(
            title="Competitive Comparison",
            description="Compare key metrics across peer companies. Input multiple tickers to analyze market position, valuation, and performance.",
            icon="bar_chart",
            workflow_id="competitive_comparison",
            agent=agent,
            ticker=ticker
        )

    with col2:
        render_workflow_card(
            title="Risk Factor Analysis",
            description="Extract and categorize risk factors from SEC filings. Compares year-over-year changes and highlights new risks.",
            icon="warning",
            workflow_id="risk_analysis",
            agent=agent,
            ticker=ticker
        )

        render_workflow_card(
            title="Price Event Investigation",
            description="Investigate significant price movements by correlating with news, SEC filings, and market events.",
            icon="search",
            workflow_id="price_investigation",
            agent=agent,
            ticker=ticker
        )


def render_workflow_card(
    title: str,
    description: str,
    icon: str,
    workflow_id: str,
    agent: FinancialResearchAgent,
    ticker: str
) -> None:
    """Render a single workflow card."""
    with st.container():
        st.markdown(f"### :{icon}: {title}")
        st.write(description)

        # Workflow-specific inputs
        if workflow_id == "competitive_comparison":
            peers = st.text_input(
                "Peer tickers (comma-separated)",
                placeholder="e.g., MSFT, GOOGL, META",
                key=f"peers_{workflow_id}"
            )

        if workflow_id == "price_investigation":
            event_date = st.date_input(
                "Date to investigate",
                key=f"date_{workflow_id}"
            )

        # Run button
        if st.button("Run Analysis", key=f"run_{workflow_id}", type="primary"):
            run_workflow(workflow_id, agent, ticker)

        st.divider()


def run_workflow(
    workflow_id: str,
    agent: FinancialResearchAgent,
    ticker: str
) -> None:
    """Execute a workflow and display results."""
    # Define workflow prompts
    workflow_prompts = {
        "financial_health": f"""Perform a comprehensive financial health analysis for {ticker}:

1. Get the latest financial statements (10-K or most recent 10-Q)
2. Calculate key financial ratios:
   - Liquidity: Current ratio, quick ratio
   - Profitability: ROE, ROA, gross margin, operating margin, net margin
   - Leverage: Debt-to-equity ratio
3. Analyze trends in revenue, net income, and cash flow
4. Identify strengths and concerns
5. Provide an overall financial health assessment

Format the results with clear sections and bullet points.""",

        "risk_analysis": f"""Analyze the risk factors for {ticker}:

1. Get the most recent 10-K filing
2. Extract the Risk Factors section (Item 1A)
3. Categorize the risks into:
   - Market/Economic risks
   - Operational risks
   - Regulatory/Legal risks
   - Technology risks
   - Financial risks
4. Identify the top 5 most significant risks
5. Note any new or emerging risks compared to typical industry risks

Provide a clear summary with risk categories and specific examples.""",

        "competitive_comparison": f"""Compare {ticker} against its main competitors:

1. Identify 2-3 key competitors in the same industry
2. Compare the following metrics:
   - Market capitalization
   - Revenue and revenue growth
   - Profitability (margins, ROE)
   - Valuation (P/E ratio, P/B ratio)
   - Debt levels
3. Analyze competitive positioning
4. Identify relative strengths and weaknesses

Present the comparison in a structured format with key takeaways.""",

        "price_investigation": f"""Investigate recent significant price movements for {ticker}:

1. Analyze price history to find the most significant recent events
2. For each major price move:
   - Identify the date and magnitude
   - Search for relevant news
   - Check for any SEC filings around that time
3. Correlate price movements with identified catalysts
4. Provide context on whether moves were company-specific or market-wide

Present findings chronologically with clear cause-effect analysis."""
    }

    prompt = workflow_prompts.get(workflow_id, f"Analyze {ticker}")

    # Execute with progress display
    with st.status("Running analysis...", expanded=True) as status:
        try:
            # Use the agent to run the workflow
            response = agent.chat(prompt)

            status.update(label="Analysis complete!", state="complete")

            # Display results
            st.markdown("### Results")
            st.markdown(response.content)

            # Show tool calls if any
            if response.tool_calls:
                with st.expander(f"Tools used ({len(response.tool_calls)})"):
                    for tc in response.tool_calls:
                        st.markdown(f"- **{tc.get('name')}**: {tc.get('arguments', {})}")

        except Exception as e:
            status.update(label="Error", state="error")
            st.error(f"Workflow error: {str(e)}")


def render_workflow_results(
    results: Dict[str, Any],
    workflow_id: str
) -> None:
    """Render formatted workflow results."""
    if workflow_id == "financial_health":
        render_financial_health_results(results)
    elif workflow_id == "risk_analysis":
        render_risk_analysis_results(results)
    elif workflow_id == "competitive_comparison":
        render_comparison_results(results)
    else:
        st.json(results)


def render_financial_health_results(results: Dict[str, Any]) -> None:
    """Render financial health analysis results."""
    st.markdown("### Financial Health Summary")

    if "ratios" in results:
        cols = st.columns(3)

        ratios = results["ratios"]

        with cols[0]:
            st.markdown("**Liquidity**")
            if "current_ratio" in ratios:
                st.metric("Current Ratio", f"{ratios['current_ratio']:.2f}")
            if "quick_ratio" in ratios:
                st.metric("Quick Ratio", f"{ratios['quick_ratio']:.2f}")

        with cols[1]:
            st.markdown("**Profitability**")
            if "roe" in ratios:
                st.metric("ROE", f"{ratios['roe']*100:.1f}%")
            if "net_margin" in ratios:
                st.metric("Net Margin", f"{ratios['net_margin']*100:.1f}%")

        with cols[2]:
            st.markdown("**Leverage**")
            if "debt_to_equity" in ratios:
                st.metric("Debt/Equity", f"{ratios['debt_to_equity']:.2f}")

    if "analysis" in results:
        st.markdown("### Analysis")
        st.write(results["analysis"])


def render_risk_analysis_results(results: Dict[str, Any]) -> None:
    """Render risk analysis results."""
    st.markdown("### Risk Factor Summary")

    if "categories" in results:
        for category, risks in results["categories"].items():
            with st.expander(f"{category} ({len(risks)} risks)"):
                for risk in risks:
                    st.markdown(f"- {risk}")

    if "top_risks" in results:
        st.markdown("### Top 5 Risks")
        for i, risk in enumerate(results["top_risks"], 1):
            st.markdown(f"**{i}.** {risk}")


def render_comparison_results(results: Dict[str, Any]) -> None:
    """Render competitive comparison results."""
    st.markdown("### Competitive Comparison")

    if "comparison" in results:
        import pandas as pd

        comparison_data = results["comparison"]
        df = pd.DataFrame(comparison_data).T

        st.dataframe(df, use_container_width=True)

    if "analysis" in results:
        st.markdown("### Analysis")
        st.write(results["analysis"])
