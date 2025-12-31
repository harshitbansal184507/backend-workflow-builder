from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import os
from dotenv import load_dotenv

from workflow_compiler import compile_workflow
from graph_executor import execute_workflow

load_dotenv()

app = FastAPI(title="Workflow Builder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class Node(BaseModel):
    id: str
    type: str
    data: Dict[str, Any]
    position: Dict[str, float]


class Edge(BaseModel):
    id: str
    source: str
    target: str
    type: Optional[str] = "default"
    data : dict[str, Any] = {}


class WorkflowDefinition(BaseModel):
    nodes: List[Node]
    edges: List[Edge]


class ExecuteRequest(BaseModel):
    workflow: WorkflowDefinition
    input: Dict[str, Any]


class ExecuteResponse(BaseModel):
    success: bool
    output: Any
    execution_log: List[Dict[str, Any]]
    error: Optional[str] = None


# Endpoints
@app.get("/")
async def root():
    return {
        "message": "Workflow Builder API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    openai_key = os.getenv("OPENAI_API_KEY")
    return {
        "status": "healthy",
        "openai_configured": bool(openai_key)
    }


@app.post("/api/workflow/validate")
async def validate_workflow(workflow: WorkflowDefinition):
    """Validate workflow structure"""
    try:
        # Check for at least one input and one output node
        input_nodes = [n for n in workflow.nodes if n.type == "input"]
        output_nodes = [n for n in workflow.nodes if n.type == "output"]
        
        if not input_nodes:
            return {
                "valid": False,
                "error": "Workflow must have at least one input node"
            }
        
        if not output_nodes:
            return {
                "valid": False,
                "error": "Workflow must have at least one output node"
            }
        
        # Check if all nodes are connected
        node_ids = {n.id for n in workflow.nodes}
        edge_sources = {e.source for e in workflow.edges}
        edge_targets = {e.target for e in workflow.edges}
        
        # All nodes except input should have incoming edges
        for node in workflow.nodes:
            if node.type != "input" and node.id not in edge_targets:
                return {
                    "valid": False,
                    "error": f"Node {node.id} has no incoming connections"
                }
        
        return {"valid": True, "message": "Workflow is valid"}
        
    except Exception as e:
        return {"valid": False, "error": str(e)}


@app.post("/api/workflow/execute", response_model=ExecuteResponse)
async def execute_workflow_endpoint(request: ExecuteRequest):
    """Execute a workflow with given input"""
    try:
        # Validate API key
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY not configured"
            )
        
        # Compile workflow to LangGraph
        graph = compile_workflow(request.workflow.model_dump())
        
        # Execute workflow
        result = execute_workflow(graph, request.input)
        
        return ExecuteResponse(
            success=True,
            output=result.get("final_output"),
            execution_log=result.get("execution_log", [])
        )
        
    except Exception as e:
        return ExecuteResponse(
            success=False,
            output=None,
            execution_log=[],
            error=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)