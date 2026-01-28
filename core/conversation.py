"""
Conversation State Management for Financial Research Agent.

Handles session persistence, message history, and conversation context.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For assistant messages with tool calls
    tool_calls: Optional[List[Dict]] = None

    # For tool response messages
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
        if self.tool_calls:
            data["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_name:
            data["tool_name"] = self.tool_name
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        data = data.copy()
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class ToolExecution:
    """Record of a tool execution."""
    tool_call_id: str
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    success: bool
    error: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
            "success": self.success,
            "error": self.error,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ConversationSession:
    """A conversation session with history and metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    title: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    tool_executions: List[ToolExecution] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        """Add a message to the session."""
        self.messages.append(message)
        self.updated_at = datetime.now()

    def add_tool_execution(self, execution: ToolExecution) -> None:
        """Add a tool execution record."""
        self.tool_executions.append(execution)

    def get_display_messages(self) -> List[Message]:
        """Get messages suitable for display (excludes tool messages)."""
        return [m for m in self.messages if m.role in ("user", "assistant")]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "title": self.title,
            "messages": [m.to_dict() for m in self.messages],
            "tool_executions": [t.to_dict() for t in self.tool_executions],
            "context": self.context
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSession":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            title=data.get("title"),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            tool_executions=[],  # Don't restore tool executions for now
            context=data.get("context", {})
        )


class ConversationManager:
    """
    Manages conversation sessions and persistence.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize the conversation manager.

        Args:
            storage_dir: Optional directory for persisting sessions
        """
        self.storage_dir = Path(storage_dir) if storage_dir else None
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._sessions: Dict[str, ConversationSession] = {}
        self._current_session_id: Optional[str] = None

    def create_session(self, title: Optional[str] = None) -> ConversationSession:
        """Create a new conversation session."""
        session = ConversationSession(title=title)
        self._sessions[session.id] = session
        self._current_session_id = session.id
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def get_current_session(self) -> Optional[ConversationSession]:
        """Get the current active session."""
        if self._current_session_id:
            return self._sessions.get(self._current_session_id)
        return None

    def set_current_session(self, session_id: str) -> bool:
        """Set the current active session."""
        if session_id in self._sessions:
            self._current_session_id = session_id
            return True
        return False

    def get_or_create_session(self) -> ConversationSession:
        """Get current session or create a new one."""
        session = self.get_current_session()
        if session is None:
            session = self.create_session()
        return session

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions with basic info."""
        return [
            {
                "id": s.id,
                "title": s.title or f"Session {s.created_at.strftime('%Y-%m-%d %H:%M')}",
                "created_at": s.created_at.isoformat(),
                "message_count": len(s.get_display_messages())
            }
            for s in sorted(
                self._sessions.values(),
                key=lambda x: x.updated_at,
                reverse=True
            )
        ]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            if self._current_session_id == session_id:
                self._current_session_id = None
            if self.storage_dir:
                session_file = self.storage_dir / f"{session_id}.json"
                if session_file.exists():
                    session_file.unlink()
            return True
        return False

    def add_user_message(self, content: str, session_id: Optional[str] = None) -> Message:
        """Add a user message to a session."""
        session = self.get_session(session_id) if session_id else self.get_or_create_session()
        message = Message(role="user", content=content)
        session.add_message(message)
        return message

    def add_assistant_message(
        self,
        content: str,
        tool_calls: Optional[List[Dict]] = None,
        session_id: Optional[str] = None
    ) -> Message:
        """Add an assistant message to a session."""
        session = self.get_session(session_id) if session_id else self.get_or_create_session()
        message = Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls
        )
        session.add_message(message)
        return message

    def add_tool_message(
        self,
        content: str,
        tool_call_id: str,
        tool_name: str,
        session_id: Optional[str] = None
    ) -> Message:
        """Add a tool result message to a session."""
        session = self.get_session(session_id) if session_id else self.get_or_create_session()
        message = Message(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            tool_name=tool_name
        )
        session.add_message(message)
        return message

    def record_tool_execution(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
        error: Optional[str] = None,
        execution_time: Optional[float] = None,
        session_id: Optional[str] = None
    ) -> ToolExecution:
        """Record a tool execution."""
        session = self.get_session(session_id) if session_id else self.get_or_create_session()
        execution = ToolExecution(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            success=success,
            error=error,
            execution_time=execution_time
        )
        session.add_tool_execution(execution)
        return execution

    def save_session(self, session_id: Optional[str] = None) -> bool:
        """Save a session to disk."""
        if not self.storage_dir:
            return False

        session = self.get_session(session_id) if session_id else self.get_current_session()
        if not session:
            return False

        try:
            session_file = self.storage_dir / f"{session.id}.json"
            with open(session_file, 'w') as f:
                json.dump(session.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            return False

    def load_session(self, session_id: str) -> Optional[ConversationSession]:
        """Load a session from disk."""
        if not self.storage_dir:
            return None

        session_file = self.storage_dir / f"{session_id}.json"
        if not session_file.exists():
            return None

        try:
            with open(session_file, 'r') as f:
                data = json.load(f)
            session = ConversationSession.from_dict(data)
            self._sessions[session.id] = session
            return session
        except Exception as e:
            logger.error(f"Error loading session: {e}")
            return None

    def load_all_sessions(self) -> int:
        """Load all sessions from disk. Returns count of loaded sessions."""
        if not self.storage_dir:
            return 0

        count = 0
        for session_file in self.storage_dir.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                session = ConversationSession.from_dict(data)
                self._sessions[session.id] = session
                count += 1
            except Exception as e:
                logger.warning(f"Error loading session {session_file}: {e}")

        return count

    def clear_current_session(self) -> None:
        """Clear the current session's messages."""
        session = self.get_current_session()
        if session:
            session.messages = []
            session.tool_executions = []
            session.updated_at = datetime.now()

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a value from the current session's context."""
        session = self.get_current_session()
        if session:
            return session.context.get(key, default)
        return default

    def set_context(self, key: str, value: Any) -> None:
        """Set a value in the current session's context."""
        session = self.get_or_create_session()
        session.context[key] = value
        session.updated_at = datetime.now()

    def get_llm_messages(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get messages in LLM-compatible format for the agent.

        Returns messages suitable for passing to the LLM provider.
        """
        session = self.get_session(session_id) if session_id else self.get_current_session()
        if not session:
            return []

        llm_messages = []
        for msg in session.messages:
            if msg.role == "tool":
                llm_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                    "name": msg.tool_name
                })
            elif msg.role == "assistant" and msg.tool_calls:
                llm_messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": msg.tool_calls
                })
            else:
                llm_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        return llm_messages
