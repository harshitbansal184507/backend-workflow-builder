"""
compiler.py - Compile a JSON workflow definition into a LangGraph graph.

HOW PAUSE/RESUME WORKS (final, correct architecture)
──────────────────────────────────────────────────────
Interactive nodes use interrupt_before at compile time.

  graph.compile(
      checkpointer=MemorySaver(),
      interrupt_before=["wait_reply", "wait_details", ...]  ← all interactive node IDs
  )

This tells LangGraph: "always pause BEFORE executing these nodes".

Execution flow for an interactive node:
  1. graph.stream(state, cfg)
     → Graph runs until it's about to execute an interactive node
     → Emits {"__interrupt__": ...} chunk and stops
     → Executor yields "waiting_for_input" to frontend

  2. User types reply

  3. executor calls:
       graph.update_state(cfg, {"user_response": "user text"})
       graph.stream(None, cfg)   ← None = resume from checkpoint
     → The interactive node NOW runs with user_response in state
     → Node reads user_response, appends to history, clears it, returns
     → Graph continues to next node

WHY THIS IS CORRECT vs previous attempts
──────────────────────────────────────────
- interrupt() inside nodes: pauses inside node, but stream_mode="updates"
  doesn't reliably emit the __interrupt__ chunk → graph appears to complete early

- update_state + stream(None) without interrupt_before: resumes AFTER the node,
  skipping all processing logic → user reply never added to conversation_history

- interrupt_before: pauses BEFORE the node, then node runs FULLY with the
  injected user_response → all processing is correct, history is complete

Node design
────────────
interactive: pure function. Checks user_response in state.
  - user_response present → process it, clear it, continue
  - user_response absent  → should not happen (interrupt_before prevents this)

prompt: pure function. Calls LLM with full conversation_history.

All nodes are simple, stateless, testable.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from backend.state import WorkflowState, ExecutionLogEntry
from backend.ai_helpers import (
    extract_variables,
    maybe_compress_history,
    pick_next_node,
    run_prompt,
    should_end_conversation,
    validate_input,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log(state: WorkflowState, node_id: str, node_type: str, action: str, detail: str = "") -> None:
    state["execution_log"].append(
        ExecutionLogEntry(node_id=node_id, node_type=node_type, action=action, detail=detail)
    )


def _fill_template(template: str, state: WorkflowState) -> str:
    combined = {**state["input"], **state["extracted_variables"]}
    for var in re.findall(r"\{\{(\w+)\}\}", template):
        val = combined.get(var)
        if val is not None:
            template = template.replace(f"{{{{{var}}}}}", str(val))
    return template


# ─────────────────────────────────────────────────────────────────────────────
# Node factories — all pure functions, no interrupt() calls
# ─────────────────────────────────────────────────────────────────────────────

def make_input_node(node_id: str):
    def node(state: WorkflowState) -> WorkflowState:
        state["current_node_id"] = node_id
        state["current_output_node_type"] = "input"
        state["current_output"] = None
        _log(state, node_id, "input", "started")
        return state
    return node


def make_output_node(node_id: str):
    def node(state: WorkflowState) -> WorkflowState:
        state["current_node_id"] = node_id
        state["current_output_node_type"] = "output"
        state["final_output"] = state["current_output"]
        state["end_conversation"] = True
        state["end_reason"] = "workflow_complete"
        _log(state, node_id, "output", "completed")
        return state
    return node


def make_prompt_node(
    *,
    node_id: str,
    system_prompt: str,
    user_prompt_template: str,
    model: str,
    temperature: float,
    require_approval: bool,
    extraction_plan: List[Dict[str, Any]],
    check_end_signal: bool,
):
    def node(state: WorkflowState) -> WorkflowState:
        state["current_node_id"] = node_id
        state["current_output_node_type"] = "prompt"

        # Max turn guard
        if state["max_turns"] > 0 and state["turn_count"] >= state["max_turns"]:
            output = "Thank you for your time! We've reached the end of our conversation."
            state["current_output"] = output
            state["end_conversation"] = True
            state["end_reason"] = "max_turns_reached"
            state["conversation_history"].append({"role": "assistant", "content": output})
            _log(state, node_id, "prompt", "max_turns_reached")
            return state

        # Check if the latest user message signals end of conversation
        if check_end_signal:
            user_msgs = [m for m in state["conversation_history"] if m["role"] == "user"]
            if user_msgs and should_end_conversation(user_msgs[-1]["content"], state["conversation_history"]):
                output = "Understood! Thank you for your time. Have a great day!"
                state["current_output"] = output
                state["end_conversation"] = True
                state["end_reason"] = "user_ended"
                state["conversation_history"].append({"role": "assistant", "content": output})
                _log(state, node_id, "prompt", "user_ended")
                return state

        # HITL: pause for human approval (uses pause_type signal, handled by executor)
        if require_approval and not state.get("hitl_decision"):
            filled = _fill_template(user_prompt_template, state)
            state["pending_hitl_prompt"] = filled
            state["pause_type"] = "hitl"
            state["pause_prompt"] = filled
            _log(state, node_id, "prompt", "waiting_hitl")
            return state

        # Apply HITL decision if present
        # Use active_template (not reassigning the closure param — causes UnboundLocalError)
        active_template = user_prompt_template
        if state.get("hitl_decision"):
            decision = state["hitl_decision"]
            dtype = decision.get("type", "approve")
            if dtype == "reject":
                state["current_output"] = f"[Rejected: {decision.get('reason', '')}]"
                state["hitl_decision"] = None
                state["pause_type"] = None
                _log(state, node_id, "prompt", "hitl_rejected")
                return state
            if dtype == "edit":
                active_template = decision.get("edited_prompt", user_prompt_template)
            state["hitl_decision"] = None
            state["pause_type"] = None

        # Compress history if it's grown large
        history, summary = maybe_compress_history(
            state["conversation_history"],
            state["conversation_summary"],
        )
        state["conversation_history"] = history
        state["conversation_summary"] = summary

        # Call LLM
        filled_prompt = _fill_template(active_template, state)
        output = run_prompt(
            system_prompt=system_prompt,
            user_prompt=filled_prompt,
            conversation_history=state["conversation_history"],
            conversation_summary=state["conversation_summary"],
            model=model,
            temperature=temperature,
        )

        state["conversation_history"].append({"role": "user", "content": filled_prompt})
        state["conversation_history"].append({"role": "assistant", "content": output})
        state["turn_count"] += 1
        state["current_output"] = output

        if extraction_plan:
            extracted = extract_variables(state["conversation_history"], extraction_plan)
            state["extracted_variables"].update(extracted)

        _log(state, node_id, "prompt", "completed", detail=f"turns={state['turn_count']}")
        return state

    return node


def make_interactive_node(
    *,
    node_id: str,
    prompt_message: str,
    validation_rule: Optional[str],
    timeout_seconds: int,
    check_end_signal: bool,
):
    """
    Pure function — no interrupt() call needed.

    This node is listed in interrupt_before at compile time, so LangGraph
    always pauses BEFORE running it. The executor injects user_response via
    update_state(), then resumes with stream(None). This node then runs with
    user_response already populated in state.

    State contract:
      On entry:  state["user_response"] = "what the user typed"
                 state["pause_prompt"]  = prompt shown to user (set by executor)
      On exit:   state["user_response"] = None (consumed)
                 state["current_output"] = user's text
                 state["conversation_history"] has the new user message appended
    """
    def node(state: WorkflowState) -> WorkflowState:
        state["current_node_id"] = node_id
        state["current_output_node_type"] = "interactive"

        # Max turn guard
        if state["max_turns"] > 0 and state["turn_count"] >= state["max_turns"]:
            state["end_conversation"] = True
            state["end_reason"] = "max_turns_reached"
            hist = state["conversation_history"]
            state["current_output"] = hist[-1]["content"] if hist else ""
            _log(state, node_id, "interactive", "max_turns_reached")
            return state

        user_response = state.get("user_response")

        # Should always be present (interrupt_before guarantees we paused before
        # this node and executor injected user_response before resuming)
        if not user_response:
            # Defensive: treat as no input, end gracefully
            state["end_conversation"] = True
            state["end_reason"] = "no_response"
            state["current_output"] = "No response received."
            _log(state, node_id, "interactive", "no_response")
            return state

        # Timeout check
        if state.get("timeout_at") and time.time() > state["timeout_at"]:
            state["end_conversation"] = True
            state["end_reason"] = "timeout"
            state["current_output"] = "Session timed out."
            state["user_response"] = None
            _log(state, node_id, "interactive", "timeout")
            return state

        # End-signal check
        if check_end_signal and should_end_conversation(user_response, state["conversation_history"]):
            output = "Understood! Thank you for your time. Have a great day!"
            state["conversation_history"].append({"role": "user", "content": user_response})
            state["conversation_history"].append({"role": "assistant", "content": output})
            state["current_output"] = output
            state["end_conversation"] = True
            state["end_reason"] = "user_ended"
            state["user_response"] = None
            _log(state, node_id, "interactive", "user_ended")
            return state

        # Validation
        if validation_rule and not validate_input(user_response, validation_rule):
            # Signal executor to ask again — set pause_type so executor knows
            # to re-pause before this node next time
            state["pause_type"] = "user_input"
            state["pause_prompt"] = f"{prompt_message}\n(Requirement: {validation_rule}. Please try again.)"
            state["user_response"] = None
            _log(state, node_id, "interactive", "validation_failed")
            return state

        # Accept the response
        state["conversation_history"].append({"role": "user", "content": user_response})
        state["current_output"] = user_response
        state["turn_count"] += 1
        state["user_response"] = None
        state["timeout_at"] = None
        state["pause_type"] = None

        _log(state, node_id, "interactive", "input_received", detail=user_response[:100])
        return state

    return node


# ─────────────────────────────────────────────────────────────────────────────
# Conditional router
# ─────────────────────────────────────────────────────────────────────────────

def make_conditional_router(*, node_id: str, candidate_edges: List[Dict], model: str):
    def router(state: WorkflowState) -> str:
        if state.get("end_conversation"):
            return candidate_edges[0]["target"]

        user_msgs = [m for m in state["conversation_history"] if m["role"] == "user"]
        last_user = user_msgs[-1]["content"] if user_msgs else ""

        return pick_next_node(
            current_node_id=node_id,
            last_user_message=last_user,
            conversation_history=state["conversation_history"],
            extracted_variables=state["extracted_variables"],
            candidate_edges=candidate_edges,
            model=model,
        )
    return router


# ─────────────────────────────────────────────────────────────────────────────
# Main compiler
# ─────────────────────────────────────────────────────────────────────────────

def compile_workflow(
    workflow_data: Dict[str, Any],
    routing_model: str = "gpt-4o-mini",
    check_end_signal: bool = True,
) -> Any:
    nodes_data: List[Dict] = workflow_data.get("nodes", [])
    edges_data: List[Dict] = workflow_data.get("edges", [])

    direct_edges: Dict[str, List[str]] = {}
    conditional_edges_map: Dict[str, List[Dict]] = {}

    for edge in edges_data:
        src, tgt = edge["source"], edge["target"]
        if edge.get("type") == "conditional":
            cond = edge.get("data", {}).get("condition", "")
            conditional_edges_map.setdefault(src, []).append({"condition": cond, "target": tgt})
        else:
            direct_edges.setdefault(src, []).append(tgt)

    graph = StateGraph(WorkflowState)

    # Collect interactive node IDs for interrupt_before
    interactive_node_ids: List[str] = []

    for node in nodes_data:
        nid, ntype, data = node["id"], node["type"], node.get("data", {})

        if ntype == "input":
            graph.add_node(nid, make_input_node(node_id=nid))

        elif ntype == "output":
            graph.add_node(nid, make_output_node(node_id=nid))

        elif ntype == "prompt":
            ep = data.get("variableExtractionPlan", {})
            extraction_plan = ep.get("output", []) if isinstance(ep, dict) else (ep if isinstance(ep, list) else [])
            graph.add_node(nid, make_prompt_node(
                node_id=nid,
                system_prompt=data.get("systemPrompt", "You are a helpful assistant."),
                user_prompt_template=data.get("userPrompt", ""),
                model=data.get("model", "gpt-4o-mini"),
                temperature=float(data.get("temperature", 0.7)),
                require_approval=bool(data.get("requireApproval", False)),
                extraction_plan=extraction_plan,
                check_end_signal=check_end_signal,
            ))

        elif ntype == "interactive":
            interactive_node_ids.append(nid)
            graph.add_node(nid, make_interactive_node(
                node_id=nid,
                prompt_message=data.get("promptMessage", "Please provide your input:"),
                validation_rule=data.get("validation") or None,
                timeout_seconds=int(data.get("timeout", 300)),
                check_end_signal=check_end_signal,
            ))

        else:
            graph.add_node(nid, lambda s, _nid=nid: {**s, "current_node_id": _nid})

    # Entry point
    input_nodes = [n for n in nodes_data if n["type"] == "input"]
    if not input_nodes:
        raise ValueError("Workflow must have at least one input node.")
    graph.set_entry_point(input_nodes[0]["id"])

    # Wire edges
    for node in nodes_data:
        nid = node["id"]
        if nid in conditional_edges_map:
            candidates = conditional_edges_map[nid]
            graph.add_conditional_edges(
                nid,
                make_conditional_router(node_id=nid, candidate_edges=candidates, model=routing_model),
                [c["target"] for c in candidates],
            )
        elif nid in direct_edges:
            for tgt in direct_edges[nid]:
                graph.add_edge(nid, tgt)
        elif node["type"] == "output":
            graph.add_edge(nid, END)

    compiled = graph.compile(
        checkpointer=MemorySaver(),
        interrupt_before=interactive_node_ids if interactive_node_ids else None,
    )

    # Build a map of {node_id: prompt_message} for all interactive nodes
    # Executor uses this to tell the frontend what prompt to show
    node_prompts: Dict[str, str] = {}
    for node in nodes_data:
        if node["type"] == "interactive":
            nid = node["id"]
            node_prompts[nid] = node.get("data", {}).get("promptMessage", "Please provide your input:")

    # Return both — executor needs the prompt map to show correct prompts
    return compiled, node_prompts