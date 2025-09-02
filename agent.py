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

class AgentLogger:
    def __init__(self):
        self.logs = []

    def log(self, msg: str):
        ts = datetime.utcnow().isoformat() + "Z"
        entry = f"{ts} - {msg}"
        print(entry)
        self.logs.append(entry)

    def save(self):
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(self.logs))

def make_node(ability, cfg, mcp, logger):
    """Return a LangGraph node function for an ability."""
    mcp_map = cfg.get("ability_to_mcp", {})

    def node(state: dict):
        payload = {"query": state.get("query"), **state}
        context = {}

        if ability == "extract_answer":
            context["human_reply"] = state.get("human_reply")

        mcp_client_hint = mcp_map.get(ability, "COMMON")

        logger.log(f"Calling ability {ability} via MCP={mcp_client_hint}")
        try:
            resp = mcp.call_ability(ability, payload, context, mcp_client_hint)
            result = resp.get("result")
            logger.log(f"Ability {ability} returned: {result}")
        except Exception as e:
            logger.log(f"Ability {ability} failed: {e}")
            return state

        state = state.copy()
        state[ability] = result
        return state

    return node

def build_graph(cfg, mcp, logger):
    graph = StateGraph(dict)

    for stage in cfg["stages"]:
        for ability in stage["abilities"]:
            graph.add_node(ability, make_node(ability, cfg, mcp, logger))

    prev = None
    first_ability = None

    for stage in cfg["stages"]:
        abilities = stage["abilities"]

        for ability in abilities:
            if not first_ability:
                first_ability = ability  
            if prev:
                graph.add_edge(prev, ability)
            prev = ability

    if first_ability:
        graph.add_edge(START, first_ability)

    if "solution_evaluation" in graph.nodes and "escalation_decision" in graph.nodes:
        graph.add_edge("solution_evaluation", "escalation_decision")
    if "escalation_decision" in graph.nodes and "update_payload" in graph.nodes:
        graph.add_edge("escalation_decision", "update_payload")

    if prev:
        graph.add_edge(prev, END)

    return graph

def run_agent(sample_input="sample_input.json"):
    cfg = load_config(CONFIG_PATH)
    with open(sample_input) as f:
        input_payload = json.load(f)

    mcp = MCPClient()
    logger = AgentLogger()

    graph = build_graph(cfg, mcp, logger)
    app = graph.compile()

    final_state = app.invoke(input_payload)

    with open(FINAL_PAYLOAD_PATH, "w") as f:
        json.dump(final_state, f, indent=2)
    logger.save()

    print("=== FINAL STATE ===")
    print(json.dumps(final_state, indent=2))

if __name__ == "__main__":
    run_agent()


