# mcp_server.py
import os
import json
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import httpx
from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()

AZURE_ENDPOINT = "AZURE_OPENAI_ENDPOINT"
AZURE_KEY = "AZURE_OPENAI_KEY"
AZURE_DEPLOYMENT = "AZURE_OPENAI_DEPLOYMENT"

if not AZURE_ENDPOINT or not AZURE_KEY or not AZURE_DEPLOYMENT:
    print("WARNING: Azure OpenAI environment variables not set. LLM calls will raise errors if attempted.")

app = FastAPI(title="MCP Server (Common + Atlas combined)")

class AbilityRequest(BaseModel):
    payload: Dict[str, Any]
    context: Dict[str, Any] = {}

def call_azure_chat_system(messages, max_tokens=512):
    
    if not (AZURE_ENDPOINT and AZURE_KEY and AZURE_DEPLOYMENT):
        raise RuntimeError("Azure OpenAI config not found in environment variables.")

    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version=2023-05-15"
    headers = {
        "api-key": AZURE_KEY,
        "Content-Type": "application/json"
    }
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0
    }
    r = httpx.post(url, json=body, headers=headers, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

@app.post("/ability/{ability_name}")
async def ability(ability_name: str, req: AbilityRequest, request: Request):
   
    mcp_client = request.headers.get("X-MCP-Client", "COMMON")
    payload = req.payload
    context = req.context or {}

    try:
        if ability_name == "accept_payload":
            return {"status": "ok", "mcp_client": mcp_client, "result": payload}

        if ability_name == "parse_request_text":
            text = payload.get("query", "")
            summary = {"summary": text.strip()}
            return {"status": "ok", "mcp_client": mcp_client, "result": summary}

        if ability_name == "extract_entities":
            text = payload.get("query", "")
            prompt = [
                {"role": "system", "content": "Extract product names, account IDs and ISO dates from the user query. Return JSON."},
                {"role": "user", "content": text}
            ]
            llm_output = call_azure_chat_system(prompt, max_tokens=200)
            try:
                parsed = json.loads(llm_output)
            except Exception:
                parsed = {"raw": llm_output}
            return {"status": "ok", "mcp_client": mcp_client, "result": parsed}

        if ability_name == "normalize_fields":
            p = payload.copy()
            if "priority" in p and isinstance(p["priority"], str):
                p["priority"] = p["priority"].strip().upper()
            return {"status": "ok", "mcp_client": mcp_client, "result": p}

        if ability_name == "enrich_records":
            prompt = [
                {"role": "system", "content": "You are a system that enriches customer records with SLA and historical ticket counts."},
                {"role": "user", "content": f"Customer email: {payload.get('email')}, ticket: {payload.get('ticket_id')}"}
            ]
            llm_output = call_azure_chat_system(prompt, max_tokens=150)
            try:
                parsed = json.loads(llm_output)
            except Exception:
                parsed = {"enrichment_note": llm_output}
            return {"status": "ok", "mcp_client": mcp_client, "result": parsed}

        if ability_name == "add_flags_calculations":
            p = payload.copy()
            priority = p.get("priority", "NORMAL").upper()
            risk = "HIGH" if priority in ["HIGH", "P1", "P0"] else "LOW"
            return {"status": "ok", "mcp_client": mcp_client, "result": {"risk": risk}}

        if ability_name == "clarify_question":
            prompt = [
                {"role": "system", "content": "You ask a single, concise clarification question for missing info."},
                {"role": "user", "content": f"Payload: {json.dumps(payload)}"}
            ]
            q = call_azure_chat_system(prompt, max_tokens=120)
            return {"status": "ok", "mcp_client": mcp_client, "result": {"clarify_question": q}}

        if ability_name == "extract_answer":
            answer = context.get("human_reply", "No reply provided")
            return {"status": "ok", "mcp_client": mcp_client, "result": {"answer": answer}}

        if ability_name == "store_answer":
            return {"status": "ok", "mcp_client": mcp_client, "result": {"stored": payload.get("answer", None)}}

        if ability_name == "knowledge_base_search":
            user_q = payload.get("query", "")
            prompt = [
                {"role": "system", "content": "Act as a knowledge base. If you have relevant KB items, return a JSON list of {title, snippet}."},
                {"role": "user", "content": user_q}
            ]
            kb = call_azure_chat_system(prompt, max_tokens=200)
            try:
                parsed = json.loads(kb)
            except Exception:
                parsed = {"kb_raw": kb}
            return {"status": "ok", "mcp_client": mcp_client, "result": parsed}

        if ability_name == "store_data":
            return {"status": "ok", "mcp_client": mcp_client, "result": {"stored_data": payload}}

        if ability_name == "solution_evaluation":
            candidates = payload.get("candidates", [])
            prompt = [
                {"role": "system", "content": "Score these solutions 1-100 and provide short justification in JSON: [{solution,score,reason}]"},
                {"role": "user", "content": json.dumps(candidates)}
            ]
            out = call_azure_chat_system(prompt, max_tokens=250)
            try:
                parsed = json.loads(out)
            except Exception:
                parsed = {"eval_raw": out}
            return {"status": "ok", "mcp_client": mcp_client, "result": parsed}

        if ability_name == "escalation_decision":
            eval_result = payload.get("evaluation", [])
            prompt = [
                {"role": "system", "content": "Decide whether to escalate when top score < 90. Return JSON {escalate: bool, reason: str}"},
                {"role": "user", "content": json.dumps(eval_result)}
            ]
            out = call_azure_chat_system(prompt, max_tokens=120)
            try:
                parsed = json.loads(out)
            except Exception:
                parsed = {"decision_raw": out}
            return {"status": "ok", "mcp_client": mcp_client, "result": parsed}

        if ability_name == "update_payload":
            return {"status": "ok", "mcp_client": mcp_client, "result": payload}

        if ability_name == "update_ticket":
            return {"status": "ok", "mcp_client": mcp_client, "result": {"ticket_updated": True}}

        if ability_name == "close_ticket":
            return {"status": "ok", "mcp_client": mcp_client, "result": {"ticket_closed": True}}

        if ability_name == "response_generation":
            prompt = [
                {"role": "system", "content": "Produce a short, empathetic customer reply with steps and next actions."},
                {"role": "user", "content": json.dumps(payload)}
            ]
            resp = call_azure_chat_system(prompt, max_tokens=200)
            return {"status": "ok", "mcp_client": mcp_client, "result": {"response": resp}}

        if ability_name == "execute_api_calls":
            return {"status": "ok", "mcp_client": mcp_client, "result": {"api_calls_executed": True}}

        if ability_name == "trigger_notifications":
            return {"status": "ok", "mcp_client": mcp_client, "result": {"notifications_sent": True}}

        if ability_name == "output_payload":
            return {"status": "ok", "mcp_client": mcp_client, "result": payload}

        raise HTTPException(status_code=404, detail=f"Ability {ability_name} not implemented")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
