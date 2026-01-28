"""
Competitive Comparison Workflow.

Compares financial metrics across multiple companies.
"""

from typing import Any, Dict, List

from .base import BaseWorkflow, WorkflowStep


class CompetitiveComparisonWorkflow(BaseWorkflow):
    """
    Compares key metrics across peer companies.

    Steps:
    1. Get information for all companies
    2. Gather financial metrics
    3. Perform comparative analysis
    4. Generate insights
    """

    @property
    def id(self) -> str:
        return "competitive_comparison"

    @property
    def name(self) -> str:
        return "Competitive Comparison"

    @property
    def description(self) -> str:
        return "Compare key financial metrics across peer companies"

    def define_steps(
        self,
        tickers: List[str],
        metrics: List[str] = None,
        **kwargs
    ) -> List[WorkflowStep]:
        """Define workflow steps."""
        tickers_str = ", ".join(tickers)

        return [
            WorkflowStep(
                id="company_info",
                name="Get Company Information",
                description=f"Fetch information for: {tickers_str}"
            ),
            WorkflowStep(
                id="metrics",
                name="Gather Financial Metrics",
                description="Collect financial ratios and metrics for comparison"
            ),
            WorkflowStep(
                id="comparison",
                name="Compare Companies",
                description="Generate side-by-side comparison table"
            ),
            WorkflowStep(
                id="analysis",
                name="Competitive Analysis",
                description="Generate insights and competitive positioning",
                metadata={"requires_agent": True}
            )
        ]

    def execute_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute a workflow step."""
        tickers = context["params"]["tickers"]
        metrics = context["params"].get("metrics", [
            "market_cap", "revenue", "net_income",
            "pe_ratio", "debt_to_equity", "roe",
            "gross_margin", "price_change_1y"
        ])

        if step.id == "company_info":
            info = {}
            for ticker in tickers:
                try:
                    result = self._call_tool("get_company_info", {"ticker": ticker})
                    info[ticker] = result
                except Exception as e:
                    info[ticker] = {"error": str(e)}
            return info

        elif step.id == "metrics":
            all_ratios = {}
            for ticker in tickers:
                try:
                    result = self._call_tool("calculate_financial_ratios", {
                        "ticker": ticker,
                        "ratios": metrics
                    })
                    all_ratios[ticker] = result
                except Exception as e:
                    all_ratios[ticker] = {"error": str(e)}
            return all_ratios

        elif step.id == "comparison":
            return self._call_tool("compare_companies", {
                "tickers": tickers,
                "metrics": metrics
            })

        elif step.id == "analysis":
            company_info = context.get("company_info", {})
            comparison = context.get("comparison", {})

            tickers_str = ", ".join(tickers)
            prompt = f"""Analyze the competitive positioning of {tickers_str} based on the following data:

## Company Information
{company_info}

## Comparison Data
{comparison}

Please provide:

1. **Market Position**: How do these companies rank by market cap and revenue?

2. **Valuation Comparison**: Compare P/E ratios and other valuation metrics. Which appears most/least expensive?

3. **Profitability Leaders**: Which company has the best margins and returns?

4. **Financial Strength**: Compare leverage and liquidity positions

5. **Growth Comparison**: Compare revenue/earnings growth and stock performance

6. **Competitive Advantages**: What are each company's relative strengths?

7. **Investment Considerations**: Key factors an investor should consider for each

Present your analysis with clear sections and include specific numbers where available.
Conclude with a summary table ranking the companies across key dimensions."""

            return self._call_agent(prompt)

    def summarize_results(self, context: Dict[str, Any]) -> str:
        """Generate workflow summary."""
        tickers = context["params"]["tickers"]
        return f"Competitive comparison completed for: {', '.join(tickers)}"
