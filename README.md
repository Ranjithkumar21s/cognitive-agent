# 🧠 Cognitive Agent

**Cognitive Agent** is a lightweight, model-agnostic framework for building intelligent, reasoning-driven agents.  
It supports **planning**, **acting**, **reflection**, **tool use**, and **supervised replanning** — without requiring LangChain or any external orchestration library.

This package is designed for developers who want full control and transparency over the agent reasoning pipeline.

---

## 🚀 Features

- **Model-Agnostic Core** – Works with any callable LLM (OpenAI, Gemini, Anthropic, Ollama, or your custom model).
- **3-Tier Memory System**
  - `short_term` – recent context
  - `working` – temporary per-run state
  - `long_term` – persistent summary storage
- **Supervisor Loop** – Automatic replanning if confidence is low.
- **Tool Integration** – Register and dynamically select from a set of external functions.
- **Knowledge Graph Extraction** – Builds a simple knowledge graph from text.
- **Streaming Support** – Stream model thinking and responses with callbacks.
- **Metrics & Token Accounting** – Aggregates usage, time, and step statistics.
- **Plugin-Ready Hooks** – Customizable planner and reflector logic.

---

## 🧩 Installation

```bash
pip install cognitive-agent
