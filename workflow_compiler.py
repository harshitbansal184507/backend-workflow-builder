from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from IPython.display import display, Image
import os
import re
from datetime import datetime


class WorkflowState(TypedDict):
    input: Dict[str, Any]
    messages: List[Dict[str, Any]]
    current_output: Any
    final_output: Any
    execution_log: List[Dict[str, Any]]


def visualize_workflow(workflow: StateGraph, workflow_name: str = "workflow") -> str:
    try:
        compiled = workflow.compile()
        graph = compiled.get_graph()
        png_bytes = graph.draw_mermaid_png()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"saved_workflows/{workflow_name}_{timestamp}.png"
        
        with open(filename, "wb") as f:
            f.write(png_bytes)
        
        return filename
    except Exception as e:
        return None


def evaluate_condition_with_llm(condition: str, output: Any, model: str = "gpt-4o-mini") -> bool:
    """
    Evaluate if output satisfies the condition using LLM.
    
    Args:
        condition: The condition to check (e.g., "TECHNICAL", "sentiment is positive")
        output: The actual output to evaluate
        model: LLM model to use for evaluation
    
    Returns:
        True if condition is satisfied, False otherwise
    """
    output_str = str(output).strip()
    
    
    # First try exact/substring match for simple classification
    if condition.upper() in output_str.upper() or output_str.upper() == condition.upper():
        return True
    
    # Fall back to LLM evaluation for complex conditions
    system_prompt = """You are a condition evaluator for a workflow system. 
Your job is to determine if a given condition is satisfied by the provided output.

Analyze the output carefully and determine if it meets the condition.
Respond with ONLY "TRUE" or "FALSE" - nothing else.

Examples:
- Condition: "user approved" | Output: "Yes, I approve this" → TRUE
- Condition: "contains error" | Output: "Everything looks good" → FALSE
- Condition: "sentiment is positive" | Output: "I love this product!" → TRUE
- Condition: "TECHNICAL" | Output: "TECHNICAL" → TRUE
- Condition: "BILLING" | Output: "BILLING" → TRUE
- Condition: "TECHNICAL" | Output: "This is a technical issue" → TRUE"""

    user_prompt = f"""Condition: {condition}

Output to evaluate:
{output_str}

Does the output satisfy the condition? Respond with only TRUE or FALSE."""

    try:
        llm = ChatOpenAI(
            model=model,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = llm.invoke(messages)
        result = response.content.strip().upper()
        
        if "TRUE" in result:
            return True
        elif "FALSE" in result:
            return False
        else:
            return False
            
    except Exception as e:
        return False


def compile_workflow(workflow_data: Dict[str, Any], 
                     visualize: bool = True,
                     condition_eval_model: str = "gpt-4o-mini") -> StateGraph:
    """
    Compile a workflow from JSON definition into an executable LangGraph.
    """
    nodes = workflow_data.get("nodes", [])
    edges = workflow_data.get("edges", [])
    workflow_name = workflow_data.get("name", "workflow")
    
    workflow = StateGraph(WorkflowState)
    
    edge_map = {}  
    conditional_edges = {}  
    
    # Parse edges
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        edge_type = edge.get("type", "default")
        
        if edge_type == "conditional":
            
            condition = edge.get("data", {}).get("condition", "")
            if source not in conditional_edges:
                conditional_edges[source] = []
            conditional_edges[source].append({
                "condition": condition,
                "target": target
            })
        else:
            if source not in edge_map:
                edge_map[source] = []
            edge_map[source].append(target)
    
    node_functions = {}
    
    # Create node functions
    for node in nodes:
        node_id = node["id"]
        node_type = node["type"]
        node_data = node.get("data", {})
        
        if node_type == "input":
            def create_input_node(nid):
                def input_node(state: WorkflowState) -> WorkflowState:
                 
                    state["execution_log"].append({
                        "node": nid,
                        "type": "input",
                        "output": state["input"]
                    })
                    state["current_output"] = state["input"]
                    return state
                return input_node
            
            node_functions[node_id] = create_input_node(node_id)
        
        elif node_type == "prompt":
            system_prompt = node_data.get("systemPrompt", "You are a helpful assistant.")
            user_prompt_template = node_data.get("userPrompt", "{{input}}")
            model_name = node_data.get("model", "gpt-4o")
            temperature = node_data.get("temperature", 0.7)
            
            def create_prompt_node(sys_prompt, usr_template, mdl, temp, nid):
                def prompt_node(state: WorkflowState) -> WorkflowState:
                  
                    
                    user_prompt = usr_template
                    
                    # Replace template variables
                    variables = re.findall(r'\{\{(\w+)\}\}', user_prompt)
                    for var in variables:
                        value = state["input"].get(var)
                        if value is None and state.get("current_output"):
                            if isinstance(state["current_output"], dict):
                                value = state["current_output"].get(var)
                            else:
                                value = state["current_output"]
                        
                        if value is not None:
                            user_prompt = user_prompt.replace(f"{{{{{var}}}}}", str(value))
                    
                 
                    
                    llm = ChatOpenAI(
                        model=mdl,
                        temperature=temp,
                        api_key=os.getenv("OPENAI_API_KEY")
                    )
                    
                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    response = llm.invoke(messages)
                    output = response.content
                    
                    
                    state["current_output"] = output
                    state["messages"].append({
                        "node": nid,
                        "input": user_prompt,
                        "output": output
                    })
                    state["execution_log"].append({
                        "node": nid,
                        "type": "prompt",
                        "input": user_prompt,
                        "output": output
                    })
                    
                    return state
                
                return prompt_node
            
            node_functions[node_id] = create_prompt_node(
                system_prompt, user_prompt_template, model_name, temperature, node_id
            )
        
        elif node_type == "output":
            def create_output_node(nid):
                def output_node(state: WorkflowState) -> WorkflowState:
                  
                    state["final_output"] = state["current_output"]
                    state["execution_log"].append({
                        "node": nid,
                        "type": "output",
                        "output": state["current_output"]
                    })
                    return state
                
                return output_node
            
            node_functions[node_id] = create_output_node(node_id)
    
    # Add nodes to workflow
    for node_id, node_func in node_functions.items():
        workflow.add_node(node_id, node_func)
    
    # Set entry point
    for node in nodes:
        if node["type"] == "input":
            workflow.set_entry_point(node["id"])
            break
    
    # Add edges
    for node in nodes:
        node_id = node["id"]
        
        if node_id in conditional_edges:
            conditions_list = conditional_edges[node_id]
            
            def make_router(conds_list, nid, eval_model):
                def router(state: WorkflowState) -> str:
                    output = state.get("current_output", "")
                    
                  
                    
                    for cond_obj in conds_list:
                        condition = cond_obj["condition"]
                        target = cond_obj["target"]
                        
                        # FIXED: Pass parameters in correct order
                        if evaluate_condition_with_llm(condition, output, eval_model):
                            return target
                    
                    # Default fallback
                    default_target = conds_list[0]["target"]
                    return default_target
                
                return router
            
            possible_targets = [c["target"] for c in conditions_list]
            
            workflow.add_conditional_edges(
                node_id,
                make_router(conditions_list, node_id, condition_eval_model),
                possible_targets
            )
        
        elif node_id in edge_map:
            for target in edge_map[node_id]:
                workflow.add_edge(node_id, target)
        
        elif node["type"] == "output":
            workflow.add_edge(node_id, END)
    
    if visualize:
        visualize_workflow(workflow, workflow_name)
    
    return workflow.compile()