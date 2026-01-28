"""
Risk Analysis Workflow.

Extracts and analyzes risk factors from SEC filings.
"""

from typing import Any, Dict, List

from .base import BaseWorkflow, WorkflowStep


class RiskAnalysisWorkflow(BaseWorkflow):
    """
    Analyzes company risk factors from SEC filings.

    Steps:
    1. Get company information
    2. Fetch risk factors section (Item 1A)
    3. Categorize and analyze risks
    4. Compare with previous year (if available)
    5. Generate risk assessment
    """

    @property
    def id(self) -> str:
        return "risk_analysis"

    @property
    def name(self) -> str:
        return "Risk Factor Analysis"

    @property
    def description(self) -> str:
        return "Extract, categorize, and analyze risk factors from SEC filings"

    def define_steps(self, ticker: str, **kwargs) -> List[WorkflowStep]:
        """Define workflow steps."""
        return [
            WorkflowStep(
                id="company_info",
                name="Get Company Info",
                description=f"Fetch basic information about {ticker}"
            ),
            WorkflowStep(
                id="risk_factors",
                name="Get Risk Factors",
                description="Extract Item 1A (Risk Factors) from latest 10-K"
            ),
            WorkflowStep(
                id="categorize",
                name="Categorize Risks",
                description="Categorize risks into market, operational, regulatory, etc."
            ),
            WorkflowStep(
                id="assessment",
                name="Risk Assessment",
                description="Generate comprehensive risk assessment",
                metadata={"requires_agent": True}
            )
        ]

    def execute_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute a workflow step."""
        ticker = context["params"]["ticker"]

        if step.id == "company_info":
            return self._call_tool("get_company_info", {"ticker": ticker})

        elif step.id == "risk_factors":
            return self._call_tool("get_sec_filing", {
                "ticker": ticker,
                "form_type": "10-K",
                "section": "Item 1A"
            })

        elif step.id == "categorize":
            risk_text = context.get("risk_factors", {}).get("text", "")

            if not risk_text or len(risk_text) < 100:
                return {"error": "Could not retrieve risk factors text"}

            # Use agent to categorize
            prompt = f"""Analyze and categorize the following risk factors for {ticker}.

RISK FACTORS TEXT (truncated to first 30,000 characters):
{risk_text[:30000]}

Please categorize the risks into these categories and list the top 3 risks in each:

1. **Market/Economic Risks**: Competition, market conditions, economic cycles
2. **Operational Risks**: Supply chain, manufacturing, key personnel
3. **Regulatory/Legal Risks**: Compliance, litigation, policy changes
4. **Technology Risks**: Cybersecurity, technology changes, IP
5. **Financial Risks**: Debt, currency, interest rates

For each category, identify and briefly describe the top 3 risks.
Format as a structured list."""

            return self._call_agent(prompt)

        elif step.id == "assessment":
            company_info = context.get("company_info", {})
            categorized = context.get("categorize", "")

            prompt = f"""Based on the risk analysis for {ticker}, provide a comprehensive risk assessment.

## Company Information
{company_info}

## Categorized Risks
{categorized}

Please provide:

1. **Top 5 Most Significant Risks**: The most material risks that could impact the company
2. **Industry-Specific Risks**: Risks that are specific to this company's industry
3. **Emerging Risks**: Any new or growing risks compared to typical disclosures
4. **Risk Mitigation**: Note any risk mitigation strategies mentioned
5. **Overall Risk Profile**: Rate as High, Moderate, or Low with justification

Be specific and cite examples from the risk factors where possible."""

            return self._call_agent(prompt)

    def summarize_results(self, context: Dict[str, Any]) -> str:
        """Generate workflow summary."""
        ticker = context["params"]["ticker"]
        return f"Risk Factor Analysis completed for {ticker}"
