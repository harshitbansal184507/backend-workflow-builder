from typing import Dict, Any
import traceback


def execute_workflow(graph, input_data: Dict[str, Any]) -> Dict[str, Any]:
  
    try:
        initial_state = {
            "input": input_data,
            "messages": [],
            "current_output": None,
            "final_output": None,
            "execution_log": []
        }
        
        print(f"[Executor] Starting workflow with input: {input_data}")
        
        result = graph.invoke(initial_state)
        
        print(f"[Executor] Workflow completed successfully")
        
        return {
            "final_output": result.get("final_output"),
            "execution_log": result.get("execution_log", []),
            "messages": result.get("messages", [])
        }
    except Exception as e:
        print(f"[Executor] Error: {str(e)}")
        print(f"[Executor] Traceback: {traceback.format_exc()}")
        raise