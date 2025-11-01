# tests/test_agent.py
import unittest
import json
import tempfile
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cognitive_agent.agent import CognitiveAgent


# ------------------------
# Dummy Model
# ------------------------
class DummyModel:
    """A minimal mock model with deterministic behavior."""
    def __call__(self, messages):
        last = messages[-1]["content"]

        # --- Planner phase ---
        if "Planner" in last or "Decompose" in last:
            return {
                "content": json.dumps({
                    "steps": [
                        "Gather sample data",
                        "TOOL:echo_tool:process sample data",
                        "Summarize insights"
                    ],
                    "rationale": "Simple 3-step plan"
                }),
                "usage": {"prompt_tokens": 25, "completion_tokens": 15, "total_tokens": 40}
            }

        # --- Reflection phase ---
        elif "Reflector" in last:
            return {
                "content": "FINAL: The agent successfully completed the task.",
                "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
            }

        # --- Action phase ---
        elif "Perform step" in last:
            if "TOOL:" in last:
                return {
                    "content": "TOOL:echo_tool:Hello from the cognitive agent",
                    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
                }
            else:
                return {
                    "content": "Direct reasoning result for step.",
                    "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25}
                }

        # Default
        return {"content": "FINAL: Task complete", "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}}


# ------------------------
# Simple Tool
# ------------------------
def echo_tool(text: str) -> str:
    return f"ECHO: {text}"


# ------------------------
# Tests
# ------------------------
class TestCognitiveAgent(unittest.TestCase):
    def setUp(self):
        self.tempfile = tempfile.NamedTemporaryFile(delete=False)
        self.agent = CognitiveAgent(
            model=DummyModel(),
            tools=[echo_tool],
            memory_store_path=self.tempfile.name,
            stream_response=True
        )

    def tearDown(self):
        if hasattr(self, 'tempfile') and self.tempfile:
            self.tempfile.close()
        if hasattr(self, 'tempfile') and os.path.exists(self.tempfile.name):
            os.unlink(self.tempfile.name)

    def test_basic_run(self):
        result = self.agent.run("Test simple reasoning task")
        self.assertIn("final_answer", result)
        self.assertIn("agent", result["final_answer"].lower())
        print("\n✅ test_basic_run passed.")

    def test_tool_invocation(self):
        result = self.agent.run("Run a task that uses echo_tool")
        trace = result["trace"]
        tool_used = any(t.get("role") == "Tool" and "ECHO:" in t.get("response", "") for t in trace)
        self.assertTrue(tool_used)
        print("\n✅ test_tool_invocation passed.")

    def test_memory_persistence(self):
        self.agent.memory.persist_long("Previous run summary")
        recall = self.agent.memory.recall_long(1)
        self.assertEqual(recall[-1]["text"], "Previous run summary")
        print("\n✅ test_memory_persistence passed.")

    def test_knowledge_graph_building(self):
        self.agent.kg.add_text("AI uses Data and improves Performance.")
        summary = self.agent.kg.summary()
        self.assertIn("AI", summary["nodes"])
        self.assertTrue(any("uses" in rel for _, rel, _ in summary["edges"]))
        print("\n✅ test_knowledge_graph_building passed.")

    def test_streaming_callback(self):
        events = []

        def stream_callback(event):
            events.append(event)

        _ = self.agent.run("Streaming test", stream_callback=stream_callback)
        event_types = {e["type"] for e in events}
        self.assertTrue(any(t in event_types for t in ["model_thinking", "model_content"]))
        print("\n✅ test_streaming_callback passed.")

    def test_confidence_and_reflection(self):
        result = self.agent.run("Analyze confidence test")
        reflect_stage = [t for t in result["trace"] if t.get("stage") == "Reflect"]
        self.assertTrue(reflect_stage)
        meta = reflect_stage[-1].get("meta_reflection", {})
        self.assertIn("confidence", meta)
        print("\n✅ test_confidence_and_reflection passed.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
