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
   
    output_str = str(output).strip()
    
  
    
    system_prompt = """You are a condition evaluator for a workflow system. 
Your job is to determine if a given condition is satisfied by the provided output.

Analyze the output carefully and determine if it meets the condition.
Respond with ONLY "TRUE" or "FALSE" - nothing else.

Examples:
- Condition: "user approved" | Output: "Yes, I approve this" → TRUE
- Condition: "contains error" | Output: "Everything looks good" → FALSE
- Condition: "sentiment is positive" | Output: "I love this product!" → TRUE
- Condition: "price is above 100" | Output: "The price is $150" → TRUE"""

    user_prompt = f"""Condition: {condition}

Output to evaluate:
{output_str}

Does the output satisfy the condition? Respond with only TRUE or FALSE."""

    try:
        # Initialize LLM with lower temperature for consistent evaluation
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
        
        # Parse the response
        if "TRUE" in result:
            print(f"[LLM Condition Eval] Result: TRUE ✓")
            return True
        elif "FALSE" in result:
            print(f"[LLM Condition Eval] Result: FALSE ✗")
            return False
        else:
            print(f"[LLM Condition Eval] Unexpected response: '{result}', defaulting to FALSE")
            return False
            
    except Exception as e:
        print(f"[LLM Condition Eval] Error: {e}, defaulting to FALSE")
        return False


def compile_workflow(workflow_data: Dict[str, Any], 
                     visualize: bool = True,
                     condition_eval_model: str = "gpt-4o-mini") -> StateGraph:
  
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
            print(f"  Conditional: {source} --[{condition}]--> {target}")
        else:
            if source not in edge_map:
                edge_map[source] = []
            edge_map[source].append(target)
            print(f"  Direct: {source} --> {target}")
    
    # Create node functions
    node_functions = {}
    
    for node in nodes:
        node_id = node["id"]
        node_type = node["type"]
        node_data = node.get("data", {})
        
        if node_type == "input":
            def input_node(state: WorkflowState) -> WorkflowState:
                print(f"\n[Node: {node_id}] Processing input...")
                state["execution_log"].append({
                    "node": node_id,
                    "type": "input",
                    "output": state["input"]
                })
                state["current_output"] = state["input"]
                return state
            
            node_functions[node_id] = input_node
        
        elif node_type == "prompt":
            system_prompt = node_data.get("systemPrompt", "You are a helpful assistant.")
            user_prompt_template = node_data.get("userPrompt", "{{input}}")
            model_name = node_data.get("model", "gpt-4o")
            temperature = node_data.get("temperature", 0.7)
            
            def create_prompt_node(sys_prompt, usr_template, mdl, temp, nid):
                def prompt_node(state: WorkflowState) -> WorkflowState:
                    print(f"\n[Node: {nid}] Executing prompt node...")
                    
                    user_prompt = usr_template
                    
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
                    
                    print(f"  Prompt: {user_prompt[:100]}...")
                    
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
                    
                    print(f"  Output: {output[:100]}...")
                    
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
                    print(f"\n[Node: {nid}] Setting final output...")
                    state["final_output"] = state["current_output"]
                    state["execution_log"].append({
                        "node": nid,
                        "type": "output",
                        "output": state["current_output"]
                    })
                    return state
                
                return output_node
            
            node_functions[node_id] = create_output_node(node_id)
    
    # Add nodes to graph
    for node_id, node_func in node_functions.items():
        workflow.add_node(node_id, node_func)
    
    # Set entry point
    for node in nodes:
        if node["type"] == "input":
            workflow.set_entry_point(node["id"])
            print(f"\n✓ Entry point set: {node['id']}")
            break
    
    # Add edges
    for node in nodes:
        node_id = node["id"]
        
        # Handle conditional edges with LLM evaluation
        if node_id in conditional_edges:
            conditions_list = conditional_edges[node_id]
            
            def make_router(conds_list, nid, eval_model):
                def router(state: WorkflowState) -> str:
                    output = state.get("current_output", "")
                    print(f"\n[Router: {nid}] Evaluating conditions with LLM...")
                    print(f"  Current output: '{str(output)[:150]}'")
                    
                    for cond_obj in conds_list:
                        condition = cond_obj["condition"]
                        target = cond_obj["target"]
                        
                        if evaluate_condition_with_llm(condition, output, eval_model):
                            print(f"  ✓ Condition '{condition}' satisfied -> routing to {target}")
                            return target
                    
                    # If no match, use first target as default
                    default_target = conds_list[0]["target"]
                    print(f"  ⚠ No condition matched, using default: {default_target}")
                    return default_target
                
                return router
            
            possible_targets = [c["target"] for c in conditions_list]
            
            workflow.add_conditional_edges(
                node_id,
                make_router(conditions_list, node_id, condition_eval_model),
                possible_targets
            )
            print(f"✓ Added conditional edges for {node_id}")
        
        # Handle regular edges
        elif node_id in edge_map:
            for target in edge_map[node_id]:
                workflow.add_edge(node_id, target)
        
        # Handle end nodes
        elif node["type"] == "output":
            workflow.add_edge(node_id, END)
    
    
    if visualize:
        visualize_workflow(workflow, workflow_name)
    
    return workflow.compile()