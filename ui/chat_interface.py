"""
Chat Interface UI for Financial Research Agent.

Provides a conversational interface with streaming responses and tool visibility.
"""

import streamlit as st
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.agent import FinancialResearchAgent, AgentResponse, StreamChunk
from core.conversation import ConversationManager, Message
from ui.sidebar import get_loaded_filings_context, get_rag_context
from config import get_config

logger = logging.getLogger(__name__)


def render_chat_interface(
    agent: Optional[FinancialResearchAgent],
    conversation_manager: ConversationManager,
    ticker: str = ""
) -> None:
    """
    Render the chat interface.

    Args:
        agent: The configured agent (may be None if not configured)
        conversation_manager: Conversation state manager
        ticker: Current ticker symbol for context
    """
    # Initialize session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "pending_tool_calls" not in st.session_state:
        st.session_state.pending_tool_calls = []
    if "chat_processing" not in st.session_state:
        st.session_state.chat_processing = False

    # Header with quick actions
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.subheader("Research Chat")

    with col2:
        if st.button("Clear Chat", type="secondary"):
            st.session_state.chat_messages = []
            st.session_state.chat_processing = False
            conversation_manager.clear_current_session()
            if agent:
                agent.reset_conversation()
            st.rerun()

    with col3:
        show_tools = st.checkbox("Show Tools", value=True, help="Show tool execution details")

    st.divider()

    # Create a container for chat messages to keep them separate from input
    chat_container = st.container()

    # Display chat messages in container (limit to last 50 to prevent rendering issues)
    messages_to_show = st.session_state.chat_messages[-50:] if len(st.session_state.chat_messages) > 50 else st.session_state.chat_messages

    if len(st.session_state.chat_messages) > 50:
        st.caption(f"Showing last 50 of {len(st.session_state.chat_messages)} messages. Clear chat to reset.")

    with chat_container:
        for msg in messages_to_show:
            try:
                render_message(msg, show_tools)
            except Exception as e:
                logger.error(f"Error rendering message: {e}")
                st.error(f"Error displaying message: {str(e)[:100]}")

    # Chat input
    prompt = st.chat_input(
        "Ask about a company..." if agent else "Configure API key to start chatting",
        disabled=agent is None
    )

    if prompt and agent:
        # Inject loaded filings and project documents context into agent using RAG or full context
        try:
            config = get_config()
            loaded_count = len(st.session_state.get("loaded_filings", []))
            
            # Set custom instructions if provided
            custom_prompt = st.session_state.get("custom_system_prompt", "")
            if custom_prompt:
                agent.set_custom_instructions(custom_prompt)
                logger.info(f"Custom instructions set: {len(custom_prompt)} chars")
            else:
                agent.set_custom_instructions(None)
            
            # Check if there's an active project with documents
            active_project_id = st.session_state.get("active_project_id")
            has_project_documents = False
            active_project = None
            if active_project_id:
                from ui.projects_panel import get_active_project, get_full_document_context
                active_project = get_active_project()
                if active_project and len(active_project.documents) > 0:
                    has_project_documents = True
                    doc_names = [d.original_name for d in active_project.documents]
                    logger.info(f"Active project '{active_project.name}' has {len(active_project.documents)} document(s): {doc_names}")

            # HYBRID MODE: Use full document for small files, RAG for large files
            context_parts = []
            full_doc_names = []
            rag_source_info = []
            
            # Step 1: Get full document content for small documents
            if has_project_documents and config.rag.full_doc_threshold_kb > 0:
                full_doc_context, full_doc_names = get_full_document_context(
                    active_project, 
                    threshold_kb=config.rag.full_doc_threshold_kb
                )
                if full_doc_context:
                    context_parts.append(full_doc_context)
                    logger.info(f"Included {len(full_doc_names)} full document(s): {full_doc_names}")
            
            # Step 2: Use RAG for large documents and filings
            use_rag_for_remaining = config.rag.enabled and (
                loaded_count > 0 or 
                (has_project_documents and len(full_doc_names) < len(active_project.documents))
            )
            
            if use_rag_for_remaining:
                # Log RAG stats for debugging
                from ui.sidebar import get_rag_manager
                rag_mgr = get_rag_manager()
                
                # Use RAG to retrieve relevant chunks
                rag_context, chunk_count = get_rag_context(prompt)
                if rag_context:
                    context_parts.append(rag_context)
                    if loaded_count > 0:
                        rag_source_info.append(f"{loaded_count} filing(s)")
                    # Count documents that weren't included in full
                    large_docs = len(active_project.documents) - len(full_doc_names) if active_project else 0
                    if large_docs > 0:
                        rag_source_info.append(f"{large_docs} large document(s)")
                    logger.info(f"RAG retrieved {chunk_count} chunks for: {rag_source_info}")
            
            # Step 3: Fall back to full filing context if no RAG results
            if not context_parts and loaded_count > 0:
                filings_context = get_loaded_filings_context()
                if filings_context:
                    context_parts.append(filings_context)
                    logger.info(f"Using full filing context: {len(filings_context)} chars")
            
            # Combine all context and set on agent
            if context_parts:
                combined_context = "\n\n".join(context_parts)
                agent.set_context(combined_context)
                
                # Show user-friendly toast
                source_parts = []
                if full_doc_names:
                    source_parts.append(f"{len(full_doc_names)} full doc(s)")
                if rag_source_info:
                    source_parts.append(f"RAG from {', '.join(rag_source_info)}")
                
                if source_parts:
                    st.toast(f"Using: {', '.join(source_parts)}", icon="📄")
                logger.info(f"Total context: {len(combined_context)} chars")
            else:
                agent.set_context(None)
                if has_project_documents:
                    st.toast("⚠️ Could not load document content. Try re-uploading.", icon="⚠️")
                    logger.warning("Have project documents but could not generate context")
        except Exception as e:
            logger.error(f"Error setting context: {e}")
            import traceback
            logger.error(traceback.format_exc())

        # Add user message
        user_msg = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.chat_messages.append(user_msg)

        # Display user message immediately
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                response_container = st.empty()
                tool_container = st.container()

                # Show loading indicator
                with response_container:
                    status = st.status("Processing your request...", expanded=True)
                    with status:
                        st.write("Initializing agent...")

                try:
                    logger.info(f"User prompt: {prompt[:100]}...")
                    logger.info(f"Agent type: {type(agent)}")
                    logger.info(f"Agent LLM: {type(agent.llm) if hasattr(agent, 'llm') else 'N/A'}")

                    # Show debug info in UI
                    with response_container:
                        st.write("Starting request...")

                    # Use streaming response
                    response = process_chat_stream(
                        agent,
                        prompt,
                        response_container,
                        tool_container,
                        show_tools
                    )

                    logger.info(f"Got response object: {type(response)}")
                    logger.info(f"Response content length: {len(response.content) if response and response.content else 0}")
                    logger.info(f"Response error: {response.error if response else 'N/A'}")
                    logger.info(f"Response tool_calls: {len(response.tool_calls) if response and response.tool_calls else 0}")

                    # Clear status and show response
                    response_container.empty()
                    with response_container:
                        if response and response.content:
                            st.markdown(response.content)
                        elif response and response.error:
                            st.error(f"Error: {response.error}")
                        else:
                            st.warning("No response received")

                    # Add assistant message to history
                    assistant_msg = {
                        "role": "assistant",
                        "content": response.content if response else "No response",
                        "tool_calls": response.tool_calls if response else [],
                        "tool_results": response.tool_results if response else [],
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.chat_messages.append(assistant_msg)

                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    logger.error(f"Chat error: {error_details}")
                    response_container.empty()
                    with response_container:
                        st.error(f"Error: {str(e)}")
                        with st.expander("Error Details"):
                            st.code(error_details)

                    # Still add an error message to chat history
                    assistant_msg = {
                        "role": "assistant",
                        "content": f"Error: {str(e)}",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.chat_messages.append(assistant_msg)

    elif prompt and not agent:
        st.error("Please configure an API key in the sidebar.")

    # Show placeholder if no messages and no agent
    if not st.session_state.chat_messages:
        if agent is None:
            st.info("Configure your API key in the sidebar to start chatting.")
        else:
            # Check if filings are loaded
            loaded_count = len(st.session_state.get("loaded_filings", []))
            if loaded_count > 0:
                st.info(f"📄 **{loaded_count} filing(s) loaded** - Ask questions about the loaded SEC filings!\n\n"
                       f"Examples:\n"
                       f"- \"What are the main risk factors?\"\n"
                       f"- \"Summarize the business description\"\n"
                       f"- \"Compare revenue between years\"\n"
                       f"- \"What changed in the risk factors?\"")
            else:
                st.info(f"Ask me anything about {'your companies' if not ticker else ticker}! For example:\n\n"
                       f"- \"What are the main risk factors?\"\n"
                       f"- \"Summarize the latest 10-K\"\n"
                       f"- \"Why did the stock drop on [date]?\"\n\n"
                       f"💡 **Tip**: Load SEC filings in the sidebar to ask questions about specific filings.")


def render_message(msg: Dict[str, Any], show_tools: bool = True) -> None:
    """Render a single chat message."""
    role = msg.get("role", "user")

    with st.chat_message(role):
        content = msg.get("content", "")
        # Use markdown for consistent rendering
        st.markdown(content)

        # Show tool calls for assistant messages
        if role == "assistant" and show_tools:
            tool_calls = msg.get("tool_calls", [])
            tool_results = msg.get("tool_results", [])

            if tool_calls:
                with st.expander(f"Tool Calls ({len(tool_calls)})", expanded=False):
                    for i, tc in enumerate(tool_calls):
                        st.markdown(f"**{tc.get('name', 'Unknown')}**")
                        st.json(tc.get('arguments', {}))

                        # Show result if available
                        if i < len(tool_results):
                            result = tool_results[i]
                            if result.get('success'):
                                st.success("Success")
                                with st.expander("Result"):
                                    if isinstance(result.get('data'), dict):
                                        st.json(result['data'])
                                    else:
                                        st.write(result.get('data'))
                            else:
                                st.error(result.get('error', 'Unknown error'))

                        if i < len(tool_calls) - 1:
                            st.divider()


def process_chat_stream(
    agent: FinancialResearchAgent,
    prompt: str,
    response_container,
    tool_container,
    show_tools: bool
) -> AgentResponse:
    """
    Process chat with streaming and display updates.

    Args:
        agent: The agent to use
        prompt: User prompt
        response_container: Streamlit container for response text
        tool_container: Streamlit container for tool displays
        show_tools: Whether to show tool execution details

    Returns:
        Final AgentResponse
    """
    accumulated_text = ""
    tool_displays = {}

    try:
        logger.info(f"Starting chat stream for prompt: {prompt[:50]}...")
        logger.info(f"Agent: {agent}, LLM provider: {agent.llm_provider if hasattr(agent, 'llm_provider') else 'N/A'}")
        
        # Try streaming first
        logger.info("Calling agent.chat_stream...")
        stream = agent.chat_stream(prompt)
        logger.info(f"Got stream generator: {type(stream)}")

        chunk_count = 0
        response_container.markdown("_Waiting for response..._")
        
        try:
            for chunk in stream:
                chunk_count += 1
                chunk_type = getattr(chunk, 'type', None)

                # Handle StreamChunk objects (use duck typing to avoid isinstance issues with module reloading)
                if chunk_type is not None:
                    if chunk_type == "text":
                        accumulated_text += chunk.content
                        response_container.markdown(accumulated_text + " ")

                    elif chunk_type == "tool_start" and show_tools:
                        with tool_container:
                            tool_key = getattr(chunk, 'tool_id', None) or getattr(chunk, 'tool_name', 'unknown')
                            tool_displays[tool_key] = st.status(
                                f"Calling {getattr(chunk, 'tool_name', 'tool')}...",
                                expanded=False
                            )
                            with tool_displays[tool_key]:
                                st.json(chunk.content)

                    elif chunk_type == "tool_result" and show_tools:
                        tool_key = getattr(chunk, 'tool_id', None) or getattr(chunk, 'tool_name', 'unknown')
                        if tool_key in tool_displays:
                            tool_displays[tool_key].update(
                                label=f"{getattr(chunk, 'tool_name', 'tool')} completed",
                                state="complete"
                            )

                    elif chunk_type == "done":
                        # Use accumulated text if available, otherwise use chunk content
                        final_text = accumulated_text or getattr(chunk, 'content', '')
                        if final_text:
                            response_container.markdown(final_text)
                            accumulated_text = final_text

                # Handle AgentResponse objects (final response)
                elif hasattr(chunk, 'content') and hasattr(chunk, 'tool_calls'):
                    logger.info(f"Received final AgentResponse with content length: {len(chunk.content)}")
                    response_container.markdown(chunk.content)
                    return chunk
        except StopIteration as e:
            logger.info(f"Generator exhausted via StopIteration: {e}")
            # Try to get return value (use duck typing)
            if hasattr(e, 'value') and hasattr(e.value, 'content') and hasattr(e.value, 'tool_calls'):
                logger.info("Got AgentResponse from StopIteration.value")
                response_container.markdown(e.value.content)
                return e.value
        except Exception as stream_error:
            logger.error(f"Error iterating stream: {stream_error}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        logger.info(f"Stream ended. Accumulated {len(accumulated_text)} chars, {chunk_count} chunks")
        
        # If we got some text, use it
        if accumulated_text:
            logger.info(f"Creating response from accumulated text: {len(accumulated_text)} chars")
            response_container.markdown(accumulated_text)
            return AgentResponse(content=accumulated_text)
        elif chunk_count == 0:
            # No chunks at all - try non-streaming as fallback
            logger.warning("No chunks received from stream. Trying non-streaming fallback...")
            try:
                response_container.markdown("_Trying alternative method..._")
                response = agent.chat(prompt)
                logger.info(f"Non-streaming response: {len(response.content)} chars")
                response_container.markdown(response.content)
                return response
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                error_msg = f"Failed to get response. Error: {str(fallback_error)}"
                response_container.error(error_msg)
                return AgentResponse(content="", error=error_msg)
        else:
            logger.warning("No content accumulated from stream")
            return AgentResponse(content="No response received. Please check your API key and try again.")

    except GeneratorExit:
        logger.info("GeneratorExit - stream interrupted")
        return AgentResponse(content=accumulated_text)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Streaming error: {error_trace}")
        # Fall back to non-streaming
        try:
            logger.info("Falling back to non-streaming chat")
            response = agent.chat(prompt)
            response_container.markdown(response.content)
            return response
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            error_msg = f"Error: {str(e)}. Fallback also failed: {str(fallback_error)}"
            response_container.error(error_msg)
            return AgentResponse(content="", error=error_msg)


def render_chat_history_sidebar(conversation_manager: ConversationManager) -> Optional[str]:
    """
    Render chat history in the sidebar.

    Returns session_id if a session is selected, None otherwise.
    """
    sessions = conversation_manager.list_sessions()

    if not sessions:
        return None

    st.sidebar.subheader("Chat History")

    for session in sessions[:5]:  # Show last 5 sessions
        col1, col2 = st.sidebar.columns([4, 1])

        with col1:
            title = session.get('title', 'Untitled')[:30]
            if st.button(title, key=f"session_{session['id']}", use_container_width=True):
                return session['id']

        with col2:
            if st.button("x", key=f"delete_{session['id']}"):
                conversation_manager.delete_session(session['id'])
                st.rerun()

    return None


def format_tool_result_for_display(result: Dict[str, Any]) -> str:
    """Format a tool result for display in the chat."""
    if not result.get('success'):
        return f"Error: {result.get('error', 'Unknown error')}"

    data = result.get('data', {})

    if isinstance(data, str):
        # Truncate long strings
        if len(data) > 500:
            return data[:500] + "..."
        return data

    if isinstance(data, dict):
        # Format key metrics nicely
        if 'ratios' in data:
            lines = [f"**{data.get('ticker', 'Company')} Financial Ratios**\n"]
            for key, value in data['ratios'].items():
                if value is not None and not key.endswith('_pct'):
                    lines.append(f"- {key.replace('_', ' ').title()}: {value}")
            return "\n".join(lines[:15])

        if 'comparison' in data:
            return "Company comparison data retrieved successfully."

        if 'filings' in data:
            filings = data['filings']
            return f"Found {len(filings)} filing(s). Latest: {filings[0].get('filing_date', 'N/A')}"

        # Default dict formatting
        return json.dumps(data, indent=2, default=str)[:1000]

    return str(data)[:500]
