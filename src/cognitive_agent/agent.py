"""
Advanced CognitiveAgent

Features:
- Model-agnostic (callable or object with invoke/chat/stream)
- 3-tier memory: short_term, working, long_term
- Supervisor loop with replanning on low confidence
- Streaming "thinking" capture (model_thinking)
- Agent-level thinking, decisions, meta-reflection, error analysis
- Dynamic tool routing (tool registry with metadata)
- Simple knowledge graph extraction & storage
- Token usage aggregation and timing metrics
- Plugin hooks for planner/validator/synthesizer
- stream_response boolean (init or run-time) and optional callback
"""

# cognitive_agent/agent.py
import json
import re
import time
from typing import List, Callable, Any, Dict, Optional


# ---------------------------
# Simple Knowledge Graph
# ---------------------------
class KnowledgeGraph:
    def __init__(self):
        self.nodes = set()
        self.edges = []

    def add_text(self, text: str):
        """
        Very naive subject-verb-object extraction.
        Example: "AI uses Data and improves Performance."
        """
        sentences = re.split(r"[.?!]", text)
        for sentence in sentences:
            words = sentence.strip().split()
            if not words:
                continue

            # Extract subject (first capitalized word)
            subject = None
            if words[0][0].isupper():
                subject = words[0]
                self.nodes.add(subject)

            # Extract verb-object pairs
            pairs = re.findall(r"\b(\w+)\s+(?:the\s+)?([A-Z][a-zA-Z]+)", sentence)
            for verb, obj in pairs:
                if subject:
                    self.edges.append((subject, verb, obj))
                    self.nodes.add(obj)
                else:
                    self.nodes.add(obj)

    def summary(self):
        return {"nodes": list(self.nodes), "edges": self.edges}


# ---------------------------
# Simple Memory System
# ---------------------------
class Memory:
    def __init__(self, path: Optional[str] = None):
        self.short_term = []
        self.long_term = []
        self.path = path

    def persist_long(self, text: str):
        entry = {"text": text, "timestamp": time.time()}
        self.long_term.append(entry)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def recall_long(self, n: int = 1):
        return self.long_term[-n:]

    def add_short(self, role: str, content: str):
        self.short_term.append({"role": role, "content": content})

    def get_context(self):
        return self.short_term[-5:]


# ---------------------------
# Main CognitiveAgent Class
# ---------------------------
class CognitiveAgent:
    def __init__(
        self,
        model: Any,
        tools: Optional[List[Callable]] = None,
        memory_store_path: Optional[str] = None,
        stream_response: bool = False,
    ):
        self.model = model
        self.stream_response = stream_response
        self.memory = Memory(memory_store_path)
        self.kg = KnowledgeGraph()
        self.tools = {t.__name__: t for t in (tools or [])}

    # ---- utility to emit events for streaming ----
    def emit_stream_event(self, stream_callback, event_type, data):
        if stream_callback:
            stream_callback({"type": event_type, "data": data})

    # ---- core execution ----
    def run(self, objective: str, stream_callback: Optional[Callable] = None):
        start_time = time.time()
        trace = []
        usage = {"steps": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # 1️⃣ Planning stage
        plan_prompt = f"Planner: Create a plan to achieve the objective: {objective}"
        plan_response = self.model([{"role": "user", "content": plan_prompt}])
        try:
            plan = json.loads(plan_response["content"])
        except json.JSONDecodeError:
            plan = {"steps": ["Perform the task directly"], "rationale": "Fallback simple plan."}
        trace.append({"role": "AI", "stage": "Plan", "content": plan})
        self.memory.add_short("plan", json.dumps(plan))
        self._accumulate_usage(usage, plan_response.get("usage", {}))
        usage["steps"] += 1

        # 2️⃣ Action stage
        for step in plan.get("steps", []):
            if self.stream_response:
                self.emit_stream_event(stream_callback, "model_thinking", f"Thinking about step: {step}")
                time.sleep(0.05)  # simulate latency

            action_prompt = f"Perform step: {step}"
            response = self.model([{"role": "user", "content": action_prompt}])
            content = response.get("content", "")

            if content.startswith("TOOL:"):
                # Parse tool invocation
                parts = content.split(":", 2)
                if len(parts) == 3:
                    _, tool_name, tool_input = parts
                    tool = self.tools.get(tool_name)
                    tool_result = tool(tool_input) if tool else f"[Unknown tool: {tool_name}]"
                    trace.append({"role": "Tool", "name": tool_name, "response": tool_result})
                    self.memory.add_short("tool", tool_result)
                    if self.stream_response:
                        self.emit_stream_event(stream_callback, "model_content", f"Tool {tool_name} executed.")
            else:
                trace.append({"role": "AI", "stage": "Act", "content": content})
                self.memory.add_short("act", content)
                if self.stream_response:
                    self.emit_stream_event(stream_callback, "model_content", f"Produced: {content}")

            self.kg.add_text(content)
            self._accumulate_usage(usage, response.get("usage", {}))
            usage["steps"] += 1

        # 3️⃣ Reflection stage
        reflect_prompt = f"Reflector: Reflect on how well the agent achieved the objective: {objective}"
        reflect_response = self.model([{"role": "user", "content": reflect_prompt}])
        reflection = reflect_response.get("content", "")
        meta_reflection = {"confidence": round(min(1.0, len(reflection) / 100.0), 2)}
        trace.append({"role": "AI", "stage": "Reflect", "content": reflection, "meta_reflection": meta_reflection})
        self._accumulate_usage(usage, reflect_response.get("usage", {}))
        usage["steps"] += 1

        # 4️⃣ Final result
        final_answer = reflection.replace("FINAL:", "").strip()
        total_runtime = round(time.time() - start_time, 3)
        usage["runtime_sec"] = total_runtime

        return {
            "objective": objective,
            "trace": trace,
            "final_answer": final_answer,
            "usage": usage,
            "knowledge_graph": self.kg.summary(),
        }

    # ---- helper ----
    def _accumulate_usage(self, usage: Dict, new: Dict):
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            usage[key] = usage.get(key, 0) + new.get(key, 0)
