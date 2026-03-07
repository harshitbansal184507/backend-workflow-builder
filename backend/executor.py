from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from backend.state import WorkflowState, initial_state


class PauseStore:
    TTL_HOURS = 24

    def __init__(self, directory: str = "workflow_states"):
        self._dir = Path(directory)
        self._dir.mkdir(exist_ok=True)

    def save(self, exec_id: str, pause_type: str, pause_prompt: str) -> None:
        (self._dir / f"{exec_id}.json").write_text(json.dumps({
            "pause_type": pause_type,
            "pause_prompt": pause_prompt,
            "saved_at": datetime.now().isoformat(),
        }))

    def load(self, exec_id: str) -> Optional[Dict]:
        path = self._dir / f"{exec_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            if datetime.now() - datetime.fromisoformat(data["saved_at"]) > timedelta(hours=self.TTL_HOURS):
                self.delete(exec_id)
                return None
            return data
        except Exception:
            return None

    def delete(self, exec_id: str) -> None:
        (self._dir / f"{exec_id}.json").unlink(missing_ok=True)

    def cleanup(self) -> int:
        removed = 0
        for path in self._dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                if datetime.now() - datetime.fromisoformat(data["saved_at"]) > timedelta(hours=self.TTL_HOURS):
                    path.unlink(missing_ok=True)
                    removed += 1
            except Exception:
                pass
        return removed


StateStore = PauseStore  


class WorkflowExecutor:

    def __init__(
        self,
        graph: Any,
        exec_id: Optional[str] = None,
        store: Optional[PauseStore] = None,
        node_prompts: Optional[Dict[str, str]] = None,
    ):
        self.graph = graph
        self.exec_id = exec_id or f"exec_{uuid.uuid4().hex[:12]}"
        self.store = store or PauseStore()
        self.node_prompts = node_prompts or {}  # {node_id: prompt_message}
        self._cfg = {"configurable": {"thread_id": self.exec_id}}


    async def start(self, input_data: Dict[str, Any], max_turns: int = 20) -> AsyncIterator[Dict[str, Any]]:
        state = initial_state(input_data, max_turns=max_turns)
        yield self._event("workflow_started", message="Workflow started")
        async for event in self._run(state):
            yield event

    async def resume_user(self, user_text: str) -> AsyncIterator[Dict[str, Any]]:
        pause_info = self.store.load(self.exec_id)
        if not pause_info or pause_info.get("pause_type") != "user_input":
            yield self._error("Not currently waiting for user input.")
            return

        import time as _time
        self.graph.update_state(self._cfg, {
            "user_response": user_text,
            "timeout_at": _time.time() + 300,
        })

        yield self._event("input_received", message="Input received, resuming…")
        async for event in self._run(None):
            yield event

    async def resume_hitl(self, decision: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        pause_info = self.store.load(self.exec_id)
        if not pause_info or pause_info.get("pause_type") != "hitl":
            yield self._error("Not currently waiting for HITL decision.")
            return

        dtype = decision.get("type")
        if dtype not in ("approve", "edit", "reject"):
            yield self._error(f"Invalid decision type: {dtype!r}.")
            return

        self.graph.update_state(self._cfg, {
            "hitl_decision": decision,
            "pause_type": None,
        })

        yield self._event("hitl_decision_received", message=f"HITL decision: {dtype}")
        async for event in self._run(None):
            yield event


    async def _run(self, input_or_none) -> AsyncIterator[Dict[str, Any]]:
       
        try:
            last_state: Dict = {}

            for chunk in self.graph.stream(input_or_none, self._cfg, stream_mode="updates"):
                for node_name, updated_state in chunk.items():
                    if node_name.startswith("__"):
                        continue

                    last_state = updated_state

                    if updated_state.get("end_conversation"):
                        final = updated_state.get("final_output") or updated_state.get("current_output")

                        # Show the final message if it came from a prompt node
                        if (updated_state.get("current_output")
                                and updated_state.get("current_output_node_type") == "prompt"):
                            yield self._event(
                                "node_completed",
                                message=updated_state["current_output"],
                                extra={"node": node_name},
                            )

                        reason = updated_state.get("end_reason", "unknown")
                        ev_type = "workflow_completed" if reason == "workflow_complete" else "workflow_ended"
                        yield self._event(
                            ev_type,
                            message="Workflow completed." if ev_type == "workflow_completed" else f"Ended: {reason}",
                            extra={
                                "reason": reason,
                                "final_output": final,
                                "extracted_variables": updated_state.get("extracted_variables", {}),
                                "turn_count": updated_state.get("turn_count", 0),
                            },
                        )
                        self.store.delete(self.exec_id)
                        return

                    # Emit assistant message from prompt nodes
                    if (updated_state.get("current_output")
                            and updated_state.get("current_output_node_type") == "prompt"):
                        yield self._event(
                            "node_completed",
                            message=updated_state["current_output"],
                            extra={
                                "node": node_name,
                                "current_state": {
                                    "current_node": updated_state.get("current_node_id"),
                                    "turn_count": updated_state.get("turn_count", 0),
                                    "extracted_variables": updated_state.get("extracted_variables", {}),
                                },
                            },
                        )

         
            snapshot = self.graph.get_state(self._cfg)
            current_values = snapshot.values if snapshot else {}
            next_nodes = snapshot.next if snapshot else ()

            if next_nodes:
                next_node = next_nodes[0]

                if current_values.get("pause_type") == "hitl":
                    prompt = current_values.get("pause_prompt", "Please review the prompt.")
                    self.store.save(self.exec_id, "hitl", prompt)
                    yield self._event(
                        "hitl_request",
                        message=prompt,
                        extra={"pending_prompt": prompt, "node": next_node},
                    )
                    return

                prompt = self.node_prompts.get(next_node, "Please provide your input:")
                self.store.save(self.exec_id, "user_input", prompt)
                yield self._event(
                    "waiting_for_input",
                    message=prompt,
                    extra={
                        "node": next_node,
                        "current_output": current_values.get("current_output"),
                    },
                )
                return

            yield self._event(
                "workflow_completed",
                message="Workflow completed.",
                extra={
                    "final_output": current_values.get("final_output"),
                    "extracted_variables": current_values.get("extracted_variables", {}),
                    "turn_count": current_values.get("turn_count", 0),
                },
            )
            self.store.delete(self.exec_id)

        except Exception as exc:
            import traceback
            yield {
                "type": "workflow_error",
                "execution_id": self.exec_id,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "timestamp": _ts(),
            }

    def _event(self, event_type: str, *, message: str = "", extra: Optional[Dict] = None) -> Dict:
        ev = {
            "type": event_type,
            "execution_id": self.exec_id,
            "message": message,
            "timestamp": _ts(),
        }
        if extra:
            ev.update(extra)
        return ev

    def _error(self, message: str) -> Dict:
        return {
            "type": "error",
            "execution_id": self.exec_id,
            "message": message,
            "timestamp": _ts(),
        }

#timestamp helper
def _ts() -> str:
    return datetime.now().isoformat()