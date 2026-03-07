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
    
    input: Dict[str, Any]         
    current_output: Optional[str]  # last node's text output
    final_output: Optional[str]   
    
    conversation_history: List[Message]
    conversation_summary: Optional[str]  # summary when history grows large
    turn_count: int              
    max_turns: int                 

    pause_type: Optional[Literal["user_input", "hitl"]]
    # "user_input"  → normal user reply expected
    # "hitl"        → human reviewer must approve/edit/reject
    # None          → graph is running normally

    pause_prompt: Optional[str]    # message shown to the human
    timeout_at: Optional[float]

    user_response: Optional[str]   # plain text reply from user
    hitl_decision: Optional[Dict[str, Any]]  # {type, edited_prompt?, reason?}
    pending_hitl_prompt: Optional[str]       # the prompt awaiting approval

    extracted_variables: Dict[str, Any]

    current_node_id: Optional[str]
    current_output_node_type: Optional[str] 
    execution_log: List[ExecutionLogEntry]

    end_conversation: bool  #any node can set it 
    end_reason: Optional[str]


def initial_state(input_data: Dict[str, Any], max_turns: int = 20) -> WorkflowState:
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