"""
Multi-LLM Provider abstraction for Claude and OpenAI.

Provides a unified interface for tool calling with both providers.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class Message:
    """Represents a message in the conversation."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool responses
    name: Optional[str] = None  # Tool name for tool responses


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    raw_response: Any = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, Generator[str, None, LLMResponse]]:
        """
        Send a chat request to the LLM.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions
            stream: Whether to stream the response

        Returns:
            LLMResponse or Generator yielding chunks then final response
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the provider is properly configured."""
        pass


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider with tool calling support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.client = None

        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.warning("anthropic package not installed")

    def is_configured(self) -> bool:
        return self.client is not None

    def _convert_messages_to_claude(self, messages: List[Message]) -> tuple:
        """Convert generic messages to Claude format."""
        system_prompt = None
        claude_messages = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                claude_messages.append({
                    "role": "user",
                    "content": msg.content
                })
            elif msg.role == "assistant":
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments
                        })
                claude_messages.append({
                    "role": "assistant",
                    "content": content if content else msg.content
                })
            elif msg.role == "tool":
                # Claude expects tool results in user messages
                claude_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content
                    }]
                })

        return system_prompt, claude_messages

    def _convert_tools_to_claude(self, tools: List[Dict]) -> List[Dict]:
        """Convert generic tool definitions to Claude format."""
        claude_tools = []
        for tool in tools:
            claude_tools.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {})
            })
        return claude_tools

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, Generator[str, None, LLMResponse]]:
        if not self.is_configured():
            raise RuntimeError("Claude provider not configured. Set API key.")

        system_prompt, claude_messages = self._convert_messages_to_claude(messages)
        claude_tools = self._convert_tools_to_claude(tools) if tools else None

        request_kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": claude_messages,
        }

        if system_prompt:
            request_kwargs["system"] = system_prompt

        if claude_tools:
            request_kwargs["tools"] = claude_tools

        if stream:
            return self._stream_chat(request_kwargs)
        else:
            return self._sync_chat(request_kwargs)

    def _sync_chat(self, request_kwargs: Dict) -> LLMResponse:
        """Synchronous chat request."""
        response = self.client.messages.create(**request_kwargs)

        content = ""
        tool_calls = []

        for block in response.content:
            if hasattr(block, 'text'):
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input
                ))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason or "stop",
            raw_response=response
        )

    def _stream_chat(self, request_kwargs: Dict) -> Generator[str, None, LLMResponse]:
        """Streaming chat request."""
        content = ""
        tool_calls = []
        current_tool = None

        with self.client.messages.stream(**request_kwargs) as stream:
            for event in stream:
                if hasattr(event, 'type'):
                    if event.type == 'content_block_start':
                        if hasattr(event, 'content_block'):
                            if event.content_block.type == 'tool_use':
                                current_tool = {
                                    'id': event.content_block.id,
                                    'name': event.content_block.name,
                                    'input': ''
                                }
                    elif event.type == 'content_block_delta':
                        if hasattr(event, 'delta'):
                            if hasattr(event.delta, 'text'):
                                content += event.delta.text
                                yield event.delta.text
                            elif hasattr(event.delta, 'partial_json'):
                                if current_tool:
                                    current_tool['input'] += event.delta.partial_json
                    elif event.type == 'content_block_stop':
                        if current_tool:
                            try:
                                args = json.loads(current_tool['input']) if current_tool['input'] else {}
                            except json.JSONDecodeError:
                                args = {}
                            tool_calls.append(ToolCall(
                                id=current_tool['id'],
                                name=current_tool['name'],
                                arguments=args
                            ))
                            current_tool = None

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="stop"
        )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "anthropic",
            "model": self.model,
            "max_tokens": self.max_tokens
        }


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider with tool calling support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1",
        max_tokens: int = 4096
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.client = None

        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("openai package not installed")

    def is_configured(self) -> bool:
        return self.client is not None

    def _convert_messages_to_openai(self, messages: List[Message]) -> List[Dict]:
        """Convert generic messages to OpenAI format."""
        openai_messages = []

        for msg in messages:
            if msg.role == "tool":
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content
                })
            elif msg.role == "assistant" and msg.tool_calls:
                openai_messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments)
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                })
            else:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        return openai_messages

    def _convert_tools_to_openai(self, tools: List[Dict]) -> List[Dict]:
        """Convert generic tool definitions to OpenAI format."""
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {})
                }
            })
        return openai_tools

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, Generator[str, None, LLMResponse]]:
        if not self.is_configured():
            raise RuntimeError("OpenAI provider not configured. Set API key.")

        openai_messages = self._convert_messages_to_openai(messages)
        openai_tools = self._convert_tools_to_openai(tools) if tools else None

        request_kwargs = {
            "model": self.model,
            "messages": openai_messages,
        }

        # Modern OpenAI models use max_completion_tokens instead of max_tokens
        # All recent models (gpt-4.1, gpt-5.x, gpt-4o, o1, o3) use max_completion_tokens
        model_lower = self.model.lower()

        # Only legacy models use max_tokens
        uses_legacy_max_tokens = (
            model_lower in ['gpt-3.5-turbo', 'gpt-4'] or
            model_lower.startswith('gpt-3.5')
        )

        logger.info(f"Model: {self.model}, uses_legacy_max_tokens: {uses_legacy_max_tokens}")

        if uses_legacy_max_tokens:
            request_kwargs["max_tokens"] = self.max_tokens
        else:
            # All modern models use max_completion_tokens
            request_kwargs["max_completion_tokens"] = self.max_tokens

        if openai_tools:
            request_kwargs["tools"] = openai_tools

        if stream:
            return self._stream_chat(request_kwargs)
        else:
            return self._sync_chat(request_kwargs)

    def _sync_chat(self, request_kwargs: Dict) -> LLMResponse:
        """Synchronous chat request."""
        response = self.client.chat.completions.create(**request_kwargs)

        message = response.choices[0].message
        content = message.content or ""
        tool_calls = []

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                ))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason or "stop",
            raw_response=response
        )

    def _stream_chat(self, request_kwargs: Dict) -> Generator[str, None, LLMResponse]:
        """Streaming chat request."""
        request_kwargs["stream"] = True

        try:
            logger.info(f"Creating OpenAI streaming request with model: {request_kwargs.get('model')}")
            logger.info(f"Request kwargs keys: {list(request_kwargs.keys())}")
            stream = self.client.chat.completions.create(**request_kwargs)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")

        content = ""
        tool_calls_data = {}
        chunk_count = 0
        finish_reason = None

        try:
            for chunk in stream:
                chunk_count += 1

                if not chunk.choices:
                    logger.debug(f"Chunk {chunk_count} has no choices")
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                # Track finish reason
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                    logger.info(f"Received finish_reason: {finish_reason}")

                if delta:
                    if delta.content:
                        content += delta.content
                        yield delta.content

                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_data:
                                tool_calls_data[idx] = {
                                    'id': '',
                                    'name': '',
                                    'arguments': ''
                                }
                            if tc.id:
                                tool_calls_data[idx]['id'] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    tool_calls_data[idx]['name'] = tc.function.name
                                if tc.function.arguments:
                                    tool_calls_data[idx]['arguments'] += tc.function.arguments

        except Exception as e:
            logger.error(f"Error processing OpenAI stream: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        logger.info(f"OpenAI stream finished. Chunks: {chunk_count}, Content: {len(content)} chars, Tool calls: {len(tool_calls_data)}, Finish reason: {finish_reason}")

        tool_calls = []
        for idx in sorted(tool_calls_data.keys()):
            tc_data = tool_calls_data[idx]
            logger.info(f"Tool call {idx}: name={tc_data['name']}, args_len={len(tc_data['arguments'])}")
            try:
                args = json.loads(tc_data['arguments']) if tc_data['arguments'] else {}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool args: {e}")
                args = {}
            tool_calls.append(ToolCall(
                id=tc_data['id'],
                name=tc_data['name'],
                arguments=args
            ))

        result = LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason or "stop"
        )

        # Yield the final response as well so it can be captured
        yield result

        return result

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "openai",
            "model": self.model,
            "max_tokens": self.max_tokens
        }


class GeminiProvider(LLMProvider):
    """Google Gemini provider with tool calling support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-pro",
        max_tokens: int = 4096
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.client = None

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai
            except ImportError:
                logger.warning("google-generativeai package not installed")

    def is_configured(self) -> bool:
        return self.client is not None

    def _convert_tools_to_gemini(self, tools: List[Dict]) -> List:
        """Convert generic tool definitions to Gemini format."""
        from google.generativeai.types import FunctionDeclaration, Tool

        function_declarations = []
        for tool in tools:
            # Build parameters schema
            params = tool.get("parameters", {})

            function_declarations.append(
                FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=params
                )
            )

        return [Tool(function_declarations=function_declarations)]

    def _convert_messages_to_gemini(self, messages: List[Message]) -> tuple:
        """Convert generic messages to Gemini format."""
        system_instruction = None
        gemini_messages = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "user":
                gemini_messages.append({
                    "role": "user",
                    "parts": [msg.content]
                })
            elif msg.role == "assistant":
                parts = []
                if msg.content:
                    parts.append(msg.content)
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        from google.generativeai.types import FunctionCall
                        parts.append(FunctionCall(
                            name=tc.name,
                            args=tc.arguments
                        ))
                gemini_messages.append({
                    "role": "model",
                    "parts": parts if parts else [msg.content or ""]
                })
            elif msg.role == "tool":
                from google.generativeai.types import FunctionResponse
                # Parse the tool result content
                try:
                    result_data = json.loads(msg.content)
                except (json.JSONDecodeError, TypeError):
                    result_data = {"result": msg.content}

                gemini_messages.append({
                    "role": "user",
                    "parts": [FunctionResponse(
                        name=msg.name,
                        response=result_data
                    )]
                })

        return system_instruction, gemini_messages

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, Generator[str, None, LLMResponse]]:
        if not self.is_configured():
            raise RuntimeError("Gemini provider not configured. Set API key.")

        system_instruction, gemini_messages = self._convert_messages_to_gemini(messages)

        # Create the model with system instruction if present
        model_kwargs = {}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction

        model = self.client.GenerativeModel(
            model_name=self.model,
            **model_kwargs
        )

        # Prepare generation config
        generation_config = {
            "max_output_tokens": self.max_tokens,
        }

        # Start chat with history
        chat = model.start_chat(history=gemini_messages[:-1] if gemini_messages else [])

        # Get the last user message
        last_message = gemini_messages[-1]["parts"] if gemini_messages else [""]

        # Prepare tool config if tools are provided
        gemini_tools = self._convert_tools_to_gemini(tools) if tools else None

        if stream:
            return self._stream_chat(chat, last_message, gemini_tools, generation_config)
        else:
            return self._sync_chat(chat, last_message, gemini_tools, generation_config)

    def _sync_chat(self, chat, message_parts, tools, generation_config) -> LLMResponse:
        """Synchronous chat request."""
        send_kwargs = {
            "generation_config": generation_config
        }
        if tools:
            send_kwargs["tools"] = tools

        response = chat.send_message(message_parts, **send_kwargs)

        content = ""
        tool_calls = []

        for part in response.parts:
            if hasattr(part, 'text') and part.text:
                content += part.text
            elif hasattr(part, 'function_call') and part.function_call:
                fc = part.function_call
                tool_calls.append(ToolCall(
                    id=f"call_{fc.name}_{len(tool_calls)}",
                    name=fc.name,
                    arguments=dict(fc.args) if fc.args else {}
                ))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="stop",
            raw_response=response
        )

    def _stream_chat(self, chat, message_parts, tools, generation_config) -> Generator[str, None, LLMResponse]:
        """Streaming chat request."""
        send_kwargs = {
            "generation_config": generation_config,
            "stream": True
        }
        if tools:
            send_kwargs["tools"] = tools

        try:
            logger.info(f"Creating Gemini streaming request with model: {self.model}")
            response = chat.send_message(message_parts, **send_kwargs)
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Gemini API call failed: {str(e)}")

        content = ""
        tool_calls = []
        chunk_count = 0

        try:
            for chunk in response:
                chunk_count += 1

                for part in chunk.parts:
                    if hasattr(part, 'text') and part.text:
                        content += part.text
                        yield part.text
                    elif hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        tool_calls.append(ToolCall(
                            id=f"call_{fc.name}_{len(tool_calls)}",
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {}
                        ))

        except Exception as e:
            logger.error(f"Error processing Gemini stream: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        logger.info(f"Gemini stream finished. Chunks: {chunk_count}, Content: {len(content)} chars, Tool calls: {len(tool_calls)}")

        result = LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="stop"
        )

        # Yield the final response as well so it can be captured
        yield result

        return result

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "gemini",
            "model": self.model,
            "max_tokens": self.max_tokens
        }


def get_provider(
    provider_type: str = "claude",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> LLMProvider:
    """
    Factory function to create an LLM provider.

    Args:
        provider_type: "claude" or "openai"
        api_key: API key for the provider
        model: Optional model override
        **kwargs: Additional provider-specific arguments

    Returns:
        LLMProvider instance
    """
    if provider_type.lower() == "claude":
        return ClaudeProvider(
            api_key=api_key,
            model=model or "claude-sonnet-4-20250514",
            **kwargs
        )
    elif provider_type.lower() == "openai":
        return OpenAIProvider(
            api_key=api_key,
            model=model or "gpt-4.1",
            **kwargs
        )
    elif provider_type.lower() == "gemini":
        return GeminiProvider(
            api_key=api_key,
            model=model or "gemini-2.5-pro",
            **kwargs
        )
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
