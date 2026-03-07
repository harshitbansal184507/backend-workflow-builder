
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI


def _llm(model: str = "gpt-4o-mini", temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature, api_key=os.getenv("OPENAI_API_KEY"))

def run_prompt(
    system_prompt: str,
    user_prompt: str,
    conversation_history: List[Dict],
    conversation_summary: Optional[str],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> str:
  
    messages: List[Dict] = [{"role": "system", "content": system_prompt}]

    if conversation_summary:
        messages.append({
            "role": "system",
            "content": f"[Earlier conversation summary]: {conversation_summary}"
        })

    # Include recent history
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_prompt})

    response = _llm(model, temperature).invoke(messages)
    return response.content.strip()


SUMMARY_THRESHOLD = 8  
KEEP_RECENT       = 4  


def maybe_compress_history(
    history: List[Dict],
    existing_summary: Optional[str],
    model: str = "gpt-4o-mini",
) -> tuple[List[Dict], Optional[str]]:
   
    if len(history) <= SUMMARY_THRESHOLD:
        return history, existing_summary

    old_part = history[:-KEEP_RECENT]
    keep_part = history[-KEEP_RECENT:]

    turns_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in old_part)
    prior = f"Prior summary: {existing_summary}\n\n" if existing_summary else ""

    prompt = (
        f"{prior}Summarise this conversation in 2-3 sentences. "
        f"Keep: key facts, decisions, extracted data. Drop pleasantries.\n\n{turns_text}"
    )

    try:
        new_summary = _llm(model).invoke([{"role": "user", "content": prompt}]).content.strip()
    except Exception:
        new_summary = existing_summary  #keep old summary

    return keep_part, new_summary


def extract_variables(
    conversation_history: List[Dict],
    extraction_plan: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
  
    if not extraction_plan:
        return {}

    recent = conversation_history[-6:]
    turns_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)

    spec = json.dumps([
        {"name": v.get("title"), "description": v.get("description"), "type": v.get("type", "string")}
        for v in extraction_plan
    ], indent=2)

    prompt = (
        f"Extract variables from this conversation. Return ONLY a JSON object (no markdown).\n\n"
        f"Variables:\n{spec}\n\nConversation:\n{turns_text}\n\n"
        f"Use null for variables not mentioned."
    )

    try:
        raw = _llm(model).invoke([{"role": "user", "content": prompt}]).content.strip()
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        return {k: v for k, v in json.loads(raw).items() if v is not None}
    except Exception:
        return {}


def pick_next_node(
    current_node_id: str,
    last_user_message: str,
    conversation_history: List[Dict],
    extracted_variables: Dict[str, Any],
    candidate_edges: List[Dict[str, Any]],   #[{condition, target}]
    model: str = "gpt-4o-mini",
) -> str:
 
    if not candidate_edges:
        raise ValueError("pick_next_node called with no candidate edges")

    if len(candidate_edges) == 1:
        return candidate_edges[0]["target"]

    recent = conversation_history[-4:]
    history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)

    numbered = "\n".join(
        f"{i+1}. Condition: \"{e['condition']}\" → {e['target']}"
        for i, e in enumerate(candidate_edges)
    )

    prompt = (
        f"You are deciding which workflow branch to take.\n\n"
        f"Current node: {current_node_id}\n"
        f"Latest user message: {last_user_message}\n"
        f"Extracted variables: {json.dumps(extracted_variables)}\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"Possible transitions:\n{numbered}\n\n"
        f"Which number best matches the situation? Reply with ONLY the number."
    )

    try:
        raw = _llm(model).invoke([{"role": "user", "content": prompt}]).content.strip()
        match = re.search(r"\d+", raw)
        idx = int(match.group()) - 1 if match else 0
        idx = max(0, min(idx, len(candidate_edges) - 1))
        return candidate_edges[idx]["target"]
    except Exception:
        return candidate_edges[0]["target"]


def should_end_conversation(
    user_message: str,
    conversation_history: List[Dict],
    model: str = "gpt-4o-mini",
) -> bool:
 
    prompt = (
        f"Does the following message clearly signal the user wants to END or STOP "
        f"the conversation (goodbye, not interested, leave me alone, stop, etc.)?\n\n"
        f"Message: {user_message}\n\n"
        f"Answer only 'yes' or 'no'."
    )
    try:
        raw = _llm(model).invoke([{"role": "user", "content": prompt}]).content.strip().lower()
        return raw.startswith("yes")
    except Exception:
        return False



def validate_input(user_input: str, rule: str, model: str = "gpt-4o-mini") -> bool:
    prompt = (
        f"Does this input satisfy the requirement? Answer only 'yes' or 'no'.\n"
        f"Requirement: {rule}\nInput: {user_input}"
    )
    try:
        raw = _llm(model).invoke([{"role": "user", "content": prompt}]).content.strip().lower()
        return raw.startswith("yes")
    except Exception:
        return True  