"""
Financial Research Agent - Main orchestrator.

Handles the tool calling loop, conversation management, and response generation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Union

from .llm_provider import LLMProvider, LLMResponse, Message, ToolCall
from .tools import ToolExecutor, ToolResult, TOOL_DEFINITIONS

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a Financial Research Agent, an AI assistant specialized in analyzing companies using SEC filings, stock price data, and news.

## Your Skills

You are an expert at analyzing SEC filings. When users ask questions, leverage your knowledge of filing structure to find relevant information:

**Risk Analysis** - When asked about risks, risk factors, or potential concerns:
- For 10-K: Navigate to Item 1A (Risk Factors) for comprehensive annual risks
- For 10-Q: Check Part II, Item 1A for quarterly risk updates
- Categorize risks (market, operational, regulatory, financial, competitive)
- Highlight new or changed risks when comparing periods

**Financial Analysis** - When asked about financials, metrics, or financial health:
- For 10-K: Use Item 8 (Financial Statements and Supplementary Data)
- For 10-Q: Use Part I, Item 1 (Financial Statements)
- Calculate and explain key ratios (liquidity, profitability, leverage)
- Compare metrics across periods when multiple filings are loaded

**Business Overview** - When asked what a company does or about their business model:
- For 10-K: Use Item 1 (Business Description) for comprehensive overview
- Include revenue segments, products/services, competitive positioning

**Management Discussion (MD&A)** - When asked about outlook, guidance, or management's view:
- For 10-K: Use Item 7 (Management's Discussion and Analysis)
- For 10-Q: Use Part I, Item 2 (MD&A)
- Focus on forward-looking statements, trends, and strategic priorities

**Period Comparisons** - When asked to compare quarters or years:
- Identify what changed between periods
- Quantify differences with specific numbers and percentages
- Explain drivers of change using MD&A commentary

## SEC Filing Structure Reference

**10-K (Annual Report):**
- Item 1: Business Description
- Item 1A: Risk Factors
- Item 1B: Unresolved Staff Comments
- Item 2: Properties
- Item 3: Legal Proceedings
- Item 5: Market for Common Equity
- Item 6: Selected Financial Data
- Item 7: Management's Discussion and Analysis (MD&A)
- Item 7A: Quantitative and Qualitative Disclosures About Market Risk
- Item 8: Financial Statements and Supplementary Data
- Item 9A: Controls and Procedures

**10-Q (Quarterly Report):**
- Part I, Item 1: Financial Statements
- Part I, Item 2: Management's Discussion and Analysis (MD&A)
- Part I, Item 3: Quantitative and Qualitative Disclosures About Market Risk
- Part I, Item 4: Controls and Procedures
- Part II, Item 1: Legal Proceedings
- Part II, Item 1A: Risk Factors (updates to 10-K risks)
- Part II, Item 2: Unregistered Sales of Equity Securities

## Response Guidelines

- Always cite the specific filing and section (e.g., "Per AAPL's 2024 10-K, Item 1A...")
- When excerpts are provided, quote relevant passages to support your analysis
- Be specific with numbers, dates, and percentages
- Format financial data clearly with appropriate units (millions, billions)
- If information isn't available in the provided excerpts, say so clearly
- Provide actionable insights, not just raw data summaries
- When comparing periods, structure the comparison clearly (tables work well)"""


@dataclass
class AgentResponse:
    """Response from the agent."""
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class StreamChunk:
    """A chunk of streamed response."""
    type: str  # "text", "tool_start", "tool_result", "done"
    content: Any
    tool_name: Optional[str] = None
    tool_id: Optional[str] = None


class FinancialResearchAgent:
    """
    Main agent orchestrator that handles conversation and tool execution.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        tool_executor: Optional[ToolExecutor] = None,
        system_prompt: Optional[str] = None,
        max_tool_iterations: int = 10
    ):
        self.llm = llm_provider
        self.tools = tool_executor or ToolExecutor()
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.max_tool_iterations = max_tool_iterations

        # Conversation history
        self._messages: List[Message] = []

        # Additional context (e.g., loaded SEC filings)
        self._additional_context: Optional[str] = None
        
        # Custom user instructions
        self._custom_instructions: Optional[str] = None

    def set_custom_instructions(self, instructions: Optional[str]) -> None:
        """Set custom user instructions to include in the system prompt."""
        self._custom_instructions = instructions
    
    def get_custom_instructions(self) -> Optional[str]:
        """Get the current custom instructions."""
        return self._custom_instructions

    def set_context(self, context: Optional[str]) -> None:
        """Set additional context to include in all requests (e.g., loaded SEC filings)."""
        self._additional_context = context

    def get_context(self) -> Optional[str]:
        """Get the current additional context."""
        return self._additional_context

    def _get_full_system_prompt(self) -> str:
        """Get system prompt with any additional context appended."""
        prompt_parts = [self.system_prompt]
        
        # Add custom user instructions if provided
        if self._custom_instructions:
            prompt_parts.append(f"""
## Custom Instructions

The user has provided the following additional instructions. Please follow these guidelines in your responses:

{self._custom_instructions}""")
        
        # Add document/filing context if provided
        if self._additional_context:
            # Check if context includes documents (not just filings)
            has_documents = "DOCUMENT:" in self._additional_context
            
            context_header = "## Available Filing Excerpts"
            if has_documents:
                context_header = "## Available Filing Excerpts and Project Documents"
                context_description = "The following excerpts from SEC filings and uploaded project documents have been retrieved as relevant to the user's query. Use your expertise to analyze this information and answer their questions."
            else:
                context_description = "The following excerpts from SEC filings have been retrieved as relevant to the user's query. Use your SEC filing expertise to analyze this information and answer their questions."

            prompt_parts.append(f"""
{context_header}

{context_description}

{self._additional_context}""")

        return "\n".join(prompt_parts)

    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self._messages = []

    def get_messages(self) -> List[Message]:
        """Get current conversation messages."""
        return self._messages.copy()

    def set_messages(self, messages: List[Message]) -> None:
        """Set conversation messages (for restoring state)."""
        self._messages = messages.copy()

    def _build_messages(self, user_input: str) -> List[Message]:
        """Build message list including system prompt and history."""
        messages = [Message(role="system", content=self.system_prompt)]
        messages.extend(self._messages)
        messages.append(Message(role="user", content=user_input))
        return messages

    def _execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        logger.info(f"Executing tool: {tool_call.name}")
        result = self.tools.execute(tool_call.name, tool_call.arguments)
        logger.info(f"Tool result: success={result.success}")
        return result

    def chat(self, user_input: str) -> AgentResponse:
        """
        Process a user message and return a response.

        This is the main non-streaming entry point.
        """
        # Add user message to history
        self._messages.append(Message(role="user", content=user_input))

        # Build full message list with context-enhanced system prompt
        messages = [Message(role="system", content=self._get_full_system_prompt())]
        messages.extend(self._messages)

        all_tool_calls = []
        all_tool_results = []

        # Tool calling loop
        for iteration in range(self.max_tool_iterations):
            try:
                response = self.llm.chat(
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    stream=False
                )
            except Exception as e:
                logger.exception("LLM error")
                return AgentResponse(
                    content="",
                    error=f"LLM error: {str(e)}"
                )

            # If no tool calls, we're done
            if not response.tool_calls:
                # Add assistant response to history
                self._messages.append(Message(
                    role="assistant",
                    content=response.content
                ))
                return AgentResponse(
                    content=response.content,
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results
                )

            # Execute tool calls
            tool_results_for_message = []
            for tc in response.tool_calls:
                all_tool_calls.append({
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments
                })

                result = self._execute_tool_call(tc)
                all_tool_results.append({
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "success": result.success,
                    "data": result.data,
                    "error": result.error
                })
                tool_results_for_message.append((tc, result))

            # Add assistant message with tool calls
            messages.append(Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls
            ))

            # Add tool results
            for tc, result in tool_results_for_message:
                messages.append(Message(
                    role="tool",
                    content=result.to_string(),
                    tool_call_id=tc.id,
                    name=tc.name
                ))

        # Max iterations reached
        return AgentResponse(
            content="I apologize, but I've reached the maximum number of tool calls. Please try a more specific question.",
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            error="Max tool iterations reached"
        )

    def chat_stream(
        self,
        user_input: str,
        on_tool_start: Optional[Callable[[str, Dict], None]] = None,
        on_tool_result: Optional[Callable[[str, ToolResult], None]] = None
    ) -> Generator[StreamChunk, None, AgentResponse]:
        """
        Process a user message with streaming response.

        Yields StreamChunk objects as the response is generated.
        Returns the final AgentResponse.
        """
        # Add user message to history
        self._messages.append(Message(role="user", content=user_input))

        # Build full message list with context-enhanced system prompt
        messages = [Message(role="system", content=self._get_full_system_prompt())]
        messages.extend(self._messages)

        all_tool_calls = []
        all_tool_results = []
        final_content = ""

        # Tool calling loop
        for iteration in range(self.max_tool_iterations):
            try:
                # Get streaming response
                stream = self.llm.chat(
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    stream=True
                )

                # Collect streamed content
                content = ""
                response = None

                # Iterate over stream - generators can yield strings and return LLMResponse
                logger.info(f"Starting to iterate over LLM stream (iteration {iteration + 1})")
                gen = iter(stream)
                chunk_count = 0
                
                # Manually iterate to catch StopIteration and get return value
                while True:
                    try:
                        chunk = next(gen)
                        chunk_count += 1
                        logger.debug(f"Received chunk {chunk_count}: type={type(chunk).__name__}, value length: {len(str(chunk)) if chunk else 0}")
                        
                        if isinstance(chunk, str):
                            content += chunk
                            yield StreamChunk(type="text", content=chunk)
                        elif isinstance(chunk, LLMResponse):
                            logger.info(f"Received LLMResponse directly from stream")
                            response = chunk
                            break
                    except StopIteration as e:
                        # Generator finished - get return value from StopIteration
                        logger.info(f"Generator exhausted. StopIteration value type: {type(getattr(e, 'value', None))}")
                        if hasattr(e, 'value') and isinstance(e.value, LLMResponse):
                            logger.info("Found LLMResponse in StopIteration.value")
                            response = e.value
                        break

                # Handle case where stream doesn't return LLMResponse
                if response is None:
                    logger.warning(f"No LLMResponse received. Content length: {len(content)}. Creating response from content.")
                    # Create response from accumulated content
                    response = LLMResponse(content=content, tool_calls=[])
                else:
                    logger.info(f"Got LLMResponse with {len(response.tool_calls)} tool calls, content length: {len(response.content)}")

            except Exception as e:
                logger.exception("LLM streaming error")
                return AgentResponse(
                    content=final_content,
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                    error=f"LLM error: {str(e)}"
                )

            # If no tool calls, we're done
            if not response.tool_calls:
                final_content = response.content
                self._messages.append(Message(
                    role="assistant",
                    content=final_content
                ))
                yield StreamChunk(type="done", content=final_content)
                return AgentResponse(
                    content=final_content,
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results
                )

            # Execute tool calls
            tool_results_for_message = []

            for tc in response.tool_calls:
                # Notify tool start
                yield StreamChunk(
                    type="tool_start",
                    content=tc.arguments,
                    tool_name=tc.name,
                    tool_id=tc.id
                )

                if on_tool_start:
                    on_tool_start(tc.name, tc.arguments)

                all_tool_calls.append({
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments
                })

                # Execute tool
                result = self._execute_tool_call(tc)

                all_tool_results.append({
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "success": result.success,
                    "data": result.data,
                    "error": result.error
                })

                # Notify tool result
                yield StreamChunk(
                    type="tool_result",
                    content=result.data if result.success else result.error,
                    tool_name=tc.name,
                    tool_id=tc.id
                )

                if on_tool_result:
                    on_tool_result(tc.name, result)

                tool_results_for_message.append((tc, result))

            # Add assistant message with tool calls
            messages.append(Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls
            ))

            # Add tool results
            for tc, result in tool_results_for_message:
                messages.append(Message(
                    role="tool",
                    content=result.to_string(),
                    tool_call_id=tc.id,
                    name=tc.name
                ))

        # Max iterations reached
        error_msg = "Maximum tool iterations reached"
        yield StreamChunk(type="done", content=error_msg)
        return AgentResponse(
            content=error_msg,
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            error=error_msg
        )


def create_agent(
    provider: str = "claude",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    anthropic_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    newsapi_key: Optional[str] = None
) -> FinancialResearchAgent:
    """
    Factory function to create a configured agent.

    Args:
        provider: LLM provider ("claude" or "openai")
        api_key: API key for the LLM provider
        model: Optional model override
        anthropic_key: Anthropic API key for news search (web search)
        openai_key: OpenAI API key for news search (web search)
        newsapi_key: NewsAPI key for news search

    Returns:
        Configured FinancialResearchAgent
    """
    from .llm_provider import get_provider

    llm = get_provider(provider, api_key=api_key, model=model)

    if not llm.is_configured():
        raise ValueError(f"LLM provider '{provider}' is not configured. Check API key.")

    # Determine which keys to use for news search based on provider
    effective_anthropic_key = anthropic_key or (api_key if provider == "claude" else None)
    effective_openai_key = openai_key or (api_key if provider == "openai" else None)

    tool_executor = ToolExecutor(
        anthropic_key=effective_anthropic_key,
        openai_key=effective_openai_key,
        newsapi_key=newsapi_key,
        openai_model=model or "gpt-5.2"  # Pass the model for news search too
    )

    return FinancialResearchAgent(
        llm_provider=llm,
        tool_executor=tool_executor
    )
