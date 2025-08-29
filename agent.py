# agent.py
import os
import json
import time
import yaml
import httpx
from datetime import datetime

CONFIG_PATH = "config.yaml"
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
LOG_PATH = "agent_run_log.txt"
FINAL_PAYLOAD_PATH = "response payload.json"

def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)

class MCPClient:
    def __init__(self, base_url=MCP_SERVER_URL):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)

    def call_ability(self, ability_name, payload, context, mcp_client_hint):
        url = f"{self.base_url}/ability/{ability_name}"
        headers = {"X-MCP-Client": mcp_client_hint}
        body = {"payload": payload, "context": context}
        r = self.client.post(url, json=body, headers=headers)
        r.raise_for_status()
        return r.json()

class LangieAgent:
    def __init__(self, cfg, mcp_client: MCPClient):
        self.cfg = cfg
        self.mcp = mcp_client
        self.state = {}  
        self.logs = []

    def log(self, message):
        ts = datetime.utcnow().isoformat() + "Z"
        entry = f"{ts} - {message}"
        print(entry)
        self.logs.append(entry)

    def run(self, input_payload):
        
        for k in self.cfg["input_schema"]:
            if k not in input_payload:
                input_payload[k] = None

        for stage in self.cfg["stages"]:
            name = stage["name"]
            mode = stage["mode"]
            self.log(f"=== Stage: {name} (mode={mode}) ===")
            if mode == "deterministic" or mode == "human":
                for ability in stage["abilities"]:
                    self.execute_ability(ability)
            elif mode == "non-deterministic":
                for ability in stage["abilities"]:
                    self.execute_ability(ability)
            else:
                self.log(f"Unknown mode {mode} - executing abilities sequentially")
                for ability in stage["abilities"]:
                    self.execute_ability(ability)

        with open(FINAL_PAYLOAD_PATH, "w") as f:
            json.dump(self.state, f, indent=2)
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(self.logs))
        return self.state

    def execute_ability(self, ability_name):
        mcp_map = self.cfg.get("ability_to_mcp", {})
        mcp_client_hint = mcp_map.get(ability_name, "COMMON")
        
        payload = {"query": self.state.get("query"), **self.state}
        context = {}
        
        if ability_name == "accept_payload":
            payload = self.initial_input.copy()
        if ability_name == "clarify_question":
            pass
        if ability_name == "extract_answer":
            context["human_reply"] = self.state.get("human_reply", None)

        self.log(f"Calling ability {ability_name} via MCP={mcp_client_hint}")
        try:
            resp = self.mcp.call_ability(ability_name, payload, context, mcp_client_hint)
        except Exception as e:
            self.log(f"Ability {ability_name} failed: {e}")
            return

        result = resp.get("result")
        self.apply_result_to_state(ability_name, result)
        self.log(f"Ability {ability_name} returned via {resp.get('mcp_client')}: {result}")

    def apply_result_to_state(self, ability, result):
        if ability == "accept_payload":
            self.state.update(result)
        elif ability == "parse_request_text":
            self.state["summary"] = result
        elif ability == "extract_entities":
            self.state["entities"] = result
        elif ability == "normalize_fields":
            self.state.update(result)
        elif ability == "enrich_records":
            self.state.setdefault("enrichment", {}).update(result)
        elif ability == "add_flags_calculations":
            self.state.setdefault("flags", {}).update(result)
        elif ability == "clarify_question":
            self.state["clarify_question"] = result.get("clarify_question")
            self.state["human_reply"] = "Here is the requested order id: ORD-12345"
        elif ability == "extract_answer":
            self.state["human_answer"] = result.get("answer")
        elif ability == "store_answer":
            self.state["stored_answer"] = result.get("stored")
        elif ability == "knowledge_base_search":
            self.state["kb"] = result
        elif ability == "store_data":
            self.state["kb_stored"] = result
        elif ability == "solution_evaluation":
            self.state["evaluations"] = result
            try:
                top = max(result, key=lambda x: x.get("score", 0))
                self.state["top_solution"] = top
            except Exception:
                self.state["top_solution"] = result
        elif ability == "escalation_decision":
            self.state["escalation_decision"] = result
        elif ability == "update_payload":
            self.state.update(result)
        elif ability == "update_ticket":
            self.state["ticket_update"] = result
        elif ability == "close_ticket":
            self.state["ticket_close"] = result
        elif ability == "response_generation":
            self.state["customer_response"] = result.get("response")
        elif ability == "execute_api_calls":
            self.state["api_actions"] = result
        elif ability == "trigger_notifications":
            self.state["notifications"] = result
        elif ability == "output_payload":
            self.state["output_payload"] = result
        else:
            self.state.setdefault("ability_outputs", {})[ability] = result

def demo_run(sample_input_path="sample_input.json"):
    cfg = load_config(CONFIG_PATH)
    with open(sample_input_path) as f:
        data = json.load(f)
    client = MCPClient()
    agent = LangieAgent(cfg, client)
    agent.initial_input = data
    agent.state = data.copy()  
    final_state = agent.run(data)
    print("=== FINAL STATE ===")
    print(json.dumps(final_state, indent=2))

if __name__ == "__main__":
    demo_run()
