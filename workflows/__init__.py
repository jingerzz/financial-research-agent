"""
Pre-built analysis workflows for the Financial Research Agent.
"""

from .base import BaseWorkflow, WorkflowStep, WorkflowResult
from .financial_health import FinancialHealthWorkflow
from .risk_analysis import RiskAnalysisWorkflow
from .competitive_comparison import CompetitiveComparisonWorkflow

__all__ = [
    'BaseWorkflow',
    'WorkflowStep',
    'WorkflowResult',
    'FinancialHealthWorkflow',
    'RiskAnalysisWorkflow',
    'CompetitiveComparisonWorkflow',
]
