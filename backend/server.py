from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.compiler import compile_workflow
from backend.executor import StateStore, WorkflowExecutor

load_dotenv()

app = FastAPI(title="Workflow Builder API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_store = StateStore()

class ClientSession:

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.graph: Optional[Any] = None
        self.node_prompts: Dict[str, str] = {}
        self.executor: Optional[WorkflowExecutor] = None

    async def send(self, payload: Dict[str, Any]) -> None:
        await self.ws.send_json(payload)


class ConnectionManager:
    def __init__(self):
        self._sessions: Dict[str, ClientSession] = {}

    async def connect(self, ws: WebSocket, client_id: str) -> ClientSession:
        await ws.accept()
        session = ClientSession(ws)
        self._sessions[client_id] = session
        return session

    def disconnect(self, client_id: str) -> None:
        self._sessions.pop(client_id, None)

    @property
    def active_count(self) -> int:
        return len(self._sessions)


manager = ConnectionManager()


@app.get("/")
async def root():
    return {
        "name": "Workflow Builder API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "active_connections": manager.active_count,
    }


class WorkflowDef(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    name: str = "workflow"


class ValidateRequest(BaseModel):
    workflow: WorkflowDef


@app.post("/api/validate")
async def validate(req: ValidateRequest):
    w = req.workflow
    input_nodes = [n for n in w.nodes if n["type"] == "input"]
    output_nodes = [n for n in w.nodes if n["type"] == "output"]
    errors = []
    if not input_nodes:
        errors.append("Workflow needs at least one input node.")
    if not output_nodes:
        errors.append("Workflow needs at least one output node.")
    node_ids = {n["id"] for n in w.nodes}
    for e in w.edges:
        if e["source"] not in node_ids:
            errors.append(f"Edge source '{e['source']}' not found in nodes.")
        if e["target"] not in node_ids:
            errors.append(f"Edge target '{e['target']}' not found in nodes.")
    return {"valid": not errors, "errors": errors}


@app.post("/api/cleanup")
async def cleanup():
    removed = _store.cleanup()
    return {"removed_states": removed}


@app.websocket("/ws/{client_id}")
async def ws_endpoint(websocket: WebSocket, client_id: str):
    session = await manager.connect(websocket, client_id)

    await session.send({
        "type": "connected",
        "client_id": client_id,
        "message": "Connected to Workflow Builder",
    })

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "set_workflow":
                try:
                    graph, node_prompts = compile_workflow(msg["workflow"])
                    session.graph = graph
                    session.node_prompts = node_prompts
                    await session.send({"type": "workflow_set", "message": "Workflow ready."})
                except Exception as exc:
                    await session.send({"type": "error", "message": f"Compile error: {exc}"})

            elif mtype == "execute":
                if session.graph is None:
                    await session.send({"type": "error", "message": "Set a workflow first."})
                    continue

                exec_id = msg.get("execution_id") or f"exec_{uuid.uuid4().hex[:12]}"
                input_data = msg.get("input", {})
                max_turns = int(msg.get("max_turns", 20))

                executor = WorkflowExecutor(session.graph, exec_id, store=_store, node_prompts=session.node_prompts)
                session.executor = executor

                async for event in executor.start(input_data, max_turns=max_turns):
                    await session.send(event)

            elif mtype == "respond":
                exec_id = msg.get("execution_id", "")
                user_text = msg.get("input", "").strip()

                if not exec_id:
                    await session.send({"type": "error", "message": "execution_id required."})
                    continue
                if not user_text:
                    await session.send({"type": "error", "message": "Empty response not accepted."})
                    continue

                if session.executor is None or session.executor.exec_id != exec_id:
                    if session.graph is None:
                        await session.send({"type": "error", "message": "No workflow loaded."})
                        continue
                    session.executor = WorkflowExecutor(session.graph, exec_id, store=_store, node_prompts=session.node_prompts)

                async for event in session.executor.resume_user(user_text):
                    await session.send(event)

            elif mtype == "hitl_decision":
                exec_id = msg.get("execution_id", "")
                decision = msg.get("decision", {})

                if not exec_id or not decision:
                    await session.send({
                        "type": "error",
                        "message": "execution_id and decision required.",
                    })
                    continue

                if session.executor is None or session.executor.exec_id != exec_id:
                    if session.graph is None:
                        await session.send({"type": "error", "message": "No workflow loaded."})
                        continue
                    session.executor = WorkflowExecutor(session.graph, exec_id, store=_store, node_prompts=session.node_prompts)

                async for event in session.executor.resume_hitl(decision):
                    await session.send(event)

            elif mtype == "get_state":
                exec_id = msg.get("execution_id", "")
                if not exec_id:
                    await session.send({"type": "error", "message": "execution_id required."})
                    continue

                state = _store.load(exec_id)
                if state:
                    await session.send({
                        "type": "state_loaded",
                        "execution_id": exec_id,
                        "current_node": state.get("current_node_id"),
                        "turn_count": state.get("turn_count", 0),
                        "extracted_variables": state.get("extracted_variables", {}),
                        "pause_type": state.get("pause_type"),
                    })
                else:
                    await session.send({
                        "type": "error",
                        "message": f"No state found for {exec_id}.",
                    })

            elif mtype == "ping":
                await session.send({"type": "pong"})

            else:
                await session.send({"type": "error", "message": f"Unknown message type: {mtype}"})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await session.send({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        manager.disconnect(client_id)


if __name__ == "__main__":
    import uvicorn

    print("Cleaning up old states…")
    _store.cleanup()

    print("Starting Workflow Builder server on :8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")