"""
state.py - Workflow execution state

All state lives here. Nodes read it, mutate it, return it.
Nothing else holds state between turns.
"""

from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class ExecutionLogEntry(TypedDict):
    node_id: str
    node_type: str
    action: str
    detail: Optional[str]


class WorkflowState(TypedDict):
    # ── Core I/O ──────────────────────────────────────────────
    input: Dict[str, Any]          # original workflow input
    current_output: Optional[str]  # last node's text output
    final_output: Optional[str]    # set only by output node

    # ── Conversation history ───────────────────────────────────
    # Full turns between assistant and user. Kept compact.
    conversation_history: List[Message]
    conversation_summary: Optional[str]  # summary when history grows large
    turn_count: int                # how many user→assistant turns so far
    max_turns: int                 # upper limit; 0 = unlimited

    # ── Pause / resume signals ─────────────────────────────────
    # The executor reads these to decide whether to keep streaming
    # or pause and wait for external input.
    pause_type: Optional[Literal["user_input", "hitl"]]
    # pause_type == "user_input"  → normal user reply expected
    # pause_type == "hitl"        → human reviewer must approve/edit/reject
    # pause_type == None          → graph is running normally

    pause_prompt: Optional[str]    # message shown to the human
    timeout_at: Optional[float]    # unix timestamp; None = no timeout

    # ── User / HITL reply (set externally before resuming) ─────
    user_response: Optional[str]   # plain text reply from user
    hitl_decision: Optional[Dict[str, Any]]  # {type, edited_prompt?, reason?}
    pending_hitl_prompt: Optional[str]       # the prompt awaiting approval

    # ── Structured variables extracted from conversation ───────
    extracted_variables: Dict[str, Any]

    # ── Graph housekeeping ─────────────────────────────────────
    current_node_id: Optional[str]
    current_output_node_type: Optional[str]  # "prompt" | "interactive" | "input" | "output"
    execution_log: List[ExecutionLogEntry]

    # ── End-conversation flags ─────────────────────────────────
    end_conversation: bool   # any node may set True to stop cleanly
    end_reason: Optional[str]


def initial_state(input_data: Dict[str, Any], max_turns: int = 20) -> WorkflowState:
    """Return a fresh, fully-initialised state for a new execution."""
    return WorkflowState(
        input=input_data,
        current_output=None,
        final_output=None,
        conversation_history=[],
        conversation_summary=None,
        turn_count=0,
        max_turns=max_turns,
        pause_type=None,
        pause_prompt=None,
        timeout_at=None,
        user_response=None,
        hitl_decision=None,
        pending_hitl_prompt=None,
        extracted_variables={},
        current_node_id=None,
        current_output_node_type=None,
        execution_log=[],
        end_conversation=False,
        end_reason=None,
    )