"""
Financial Health Analysis Workflow.

Comprehensive analysis of a company's financial health using SEC filings.
"""

from typing import Any, Dict, List

from .base import BaseWorkflow, WorkflowStep, StepStatus


class FinancialHealthWorkflow(BaseWorkflow):
    """
    Analyzes a company's financial health using SEC filings.

    Steps:
    1. Get company information
    2. Fetch latest financial statements
    3. Calculate financial ratios
    4. Analyze trends
    5. Generate health assessment
    """

    @property
    def id(self) -> str:
        return "financial_health"

    @property
    def name(self) -> str:
        return "Financial Health Analysis"

    @property
    def description(self) -> str:
        return "Comprehensive analysis of financial statements including ratios, trends, and overall health assessment"

    def define_steps(self, ticker: str, **kwargs) -> List[WorkflowStep]:
        """Define workflow steps."""
        return [
            WorkflowStep(
                id="company_info",
                name="Get Company Info",
                description=f"Fetch basic information about {ticker}"
            ),
            WorkflowStep(
                id="financials",
                name="Get Financial Statements",
                description="Retrieve latest balance sheet, income statement, and cash flow"
            ),
            WorkflowStep(
                id="ratios",
                name="Calculate Financial Ratios",
                description="Calculate liquidity, profitability, and leverage ratios"
            ),
            WorkflowStep(
                id="analysis",
                name="Analyze Financial Health",
                description="Generate comprehensive health assessment using LLM",
                metadata={"requires_agent": True}
            )
        ]

    def execute_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute a workflow step."""
        ticker = context["params"]["ticker"]

        if step.id == "company_info":
            return self._call_tool("get_company_info", {"ticker": ticker})

        elif step.id == "financials":
            return self._call_tool("get_financial_statements", {
                "ticker": ticker,
                "form_type": "10-K"
            })

        elif step.id == "ratios":
            return self._call_tool("calculate_financial_ratios", {
                "ticker": ticker,
                "ratios": [
                    "current_ratio", "quick_ratio",
                    "roe", "roa", "gross_margin", "operating_margin", "net_margin",
                    "debt_to_equity", "pe_ratio"
                ]
            })

        elif step.id == "analysis":
            # Use the agent to generate analysis
            company_info = context.get("company_info", {})
            financials = context.get("financials", {})
            ratios = context.get("ratios", {})

            prompt = f"""Analyze the financial health of {ticker} based on the following data:

## Company Information
{company_info}

## Financial Statements
{financials}

## Financial Ratios
{ratios}

Please provide:
1. **Liquidity Assessment**: Analyze the company's ability to meet short-term obligations
2. **Profitability Assessment**: Evaluate margins and returns
3. **Leverage Assessment**: Analyze debt levels and coverage
4. **Key Strengths**: 2-3 financial strengths
5. **Key Concerns**: 2-3 areas of concern
6. **Overall Health Rating**: Rate as Strong, Moderate, or Weak with justification

Format your response with clear sections and bullet points."""

            return self._call_agent(prompt)

    def summarize_results(self, context: Dict[str, Any]) -> str:
        """Generate workflow summary."""
        ticker = context["params"]["ticker"]
        ratios = context.get("ratios", {}).get("ratios", {})

        summary_parts = [f"Financial Health Analysis for {ticker}"]

        if ratios:
            if ratios.get("current_ratio"):
                summary_parts.append(f"Current Ratio: {ratios['current_ratio']:.2f}")
            if ratios.get("roe"):
                summary_parts.append(f"ROE: {ratios['roe']*100:.1f}%")
            if ratios.get("debt_to_equity"):
                summary_parts.append(f"D/E: {ratios['debt_to_equity']:.2f}")

        return " | ".join(summary_parts)
