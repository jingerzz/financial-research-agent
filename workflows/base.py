"""
Base Workflow Class for Financial Research Agent.

Provides a framework for building reusable, step-based analysis workflows.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional
import logging

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    id: str
    name: str
    description: str
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata
        }


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""
    workflow_id: str
    workflow_name: str
    success: bool
    steps: List[WorkflowStep]
    summary: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "success": self.success,
            "steps": [s.to_dict() for s in self.steps],
            "summary": self.summary,
            "data": self.data,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error
        }


class BaseWorkflow(ABC):
    """
    Abstract base class for analysis workflows.

    Workflows are reusable, step-based analysis patterns that combine
    multiple tool calls and LLM interactions into a coherent analysis.
    """

    def __init__(
        self,
        agent=None,
        tool_executor=None,
        progress_callback: Optional[Callable[[WorkflowStep], None]] = None
    ):
        """
        Initialize the workflow.

        Args:
            agent: Optional FinancialResearchAgent for LLM interactions
            tool_executor: Optional ToolExecutor for direct tool calls
            progress_callback: Optional callback for progress updates
        """
        self.agent = agent
        self.tool_executor = tool_executor or (agent.tools if agent else None)
        self.progress_callback = progress_callback
        self._steps: List[WorkflowStep] = []
        self._current_step: Optional[WorkflowStep] = None

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for this workflow."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this workflow."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this workflow does."""
        pass

    @abstractmethod
    def define_steps(self, **kwargs) -> List[WorkflowStep]:
        """
        Define the steps for this workflow.

        Args:
            **kwargs: Workflow-specific parameters

        Returns:
            List of WorkflowStep objects
        """
        pass

    @abstractmethod
    def execute_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """
        Execute a single workflow step.

        Args:
            step: The step to execute
            context: Shared context dictionary for passing data between steps

        Returns:
            Step result (also stored in step.result)
        """
        pass

    def summarize_results(self, context: Dict[str, Any]) -> str:
        """
        Generate a summary of the workflow results.

        Override in subclasses for custom summaries.
        """
        completed = sum(1 for s in self._steps if s.status == StepStatus.COMPLETED)
        total = len(self._steps)
        return f"Completed {completed}/{total} steps"

    def run(self, **kwargs) -> WorkflowResult:
        """
        Run the complete workflow.

        Args:
            **kwargs: Workflow-specific parameters

        Returns:
            WorkflowResult with all steps and data
        """
        result = WorkflowResult(
            workflow_id=self.id,
            workflow_name=self.name,
            success=False,
            steps=[],
            started_at=datetime.now()
        )

        try:
            # Define steps
            self._steps = self.define_steps(**kwargs)
            result.steps = self._steps

            # Execute steps
            context: Dict[str, Any] = {"params": kwargs}

            for step in self._steps:
                self._current_step = step
                step.status = StepStatus.RUNNING
                step.started_at = datetime.now()

                if self.progress_callback:
                    self.progress_callback(step)

                try:
                    step.result = self.execute_step(step, context)
                    step.status = StepStatus.COMPLETED
                    context[step.id] = step.result

                except Exception as e:
                    logger.exception(f"Step {step.id} failed")
                    step.status = StepStatus.FAILED
                    step.error = str(e)

                step.completed_at = datetime.now()

                if self.progress_callback:
                    self.progress_callback(step)

                # Stop on failure unless step allows continuation
                if step.status == StepStatus.FAILED and not step.metadata.get("continue_on_failure"):
                    break

            # Generate summary
            result.summary = self.summarize_results(context)
            result.data = {k: v for k, v in context.items() if k != "params"}
            result.success = all(
                s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
                for s in self._steps
            )

        except Exception as e:
            logger.exception(f"Workflow {self.id} failed")
            result.error = str(e)

        result.completed_at = datetime.now()
        return result

    def run_streaming(self, **kwargs) -> Generator[WorkflowStep, None, WorkflowResult]:
        """
        Run the workflow with streaming step updates.

        Yields WorkflowStep objects as they complete.
        Returns final WorkflowResult.
        """
        result = WorkflowResult(
            workflow_id=self.id,
            workflow_name=self.name,
            success=False,
            steps=[],
            started_at=datetime.now()
        )

        try:
            self._steps = self.define_steps(**kwargs)
            result.steps = self._steps
            context: Dict[str, Any] = {"params": kwargs}

            for step in self._steps:
                self._current_step = step
                step.status = StepStatus.RUNNING
                step.started_at = datetime.now()

                yield step  # Yield step starting

                try:
                    step.result = self.execute_step(step, context)
                    step.status = StepStatus.COMPLETED
                    context[step.id] = step.result

                except Exception as e:
                    step.status = StepStatus.FAILED
                    step.error = str(e)

                step.completed_at = datetime.now()
                yield step  # Yield step completed

                if step.status == StepStatus.FAILED and not step.metadata.get("continue_on_failure"):
                    break

            result.summary = self.summarize_results(context)
            result.data = {k: v for k, v in context.items() if k != "params"}
            result.success = all(
                s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
                for s in self._steps
            )

        except Exception as e:
            result.error = str(e)

        result.completed_at = datetime.now()
        return result

    def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Helper method to call a tool."""
        if self.tool_executor:
            result = self.tool_executor.execute(tool_name, arguments)
            if result.success:
                return result.data
            raise RuntimeError(result.error or f"Tool {tool_name} failed")
        raise RuntimeError("No tool executor available")

    def _call_agent(self, prompt: str) -> str:
        """Helper method to call the agent."""
        if self.agent:
            response = self.agent.chat(prompt)
            return response.content
        raise RuntimeError("No agent available")
