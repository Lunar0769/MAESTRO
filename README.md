# Maestro — Dual-Model Local AI Orchestration

Runs entirely on your machine using Ollama. No API keys needed.

## Models

| Role | Model |
|------|-------|
| Planner + Critic A | `gemma4:e4b` |
| Builder + Critic B | `qwen3:8b` |

## How it works

```
gemma4  →  Specification (requirements, architecture)
qwen3   →  Code (implements all requirements)
gemma4  →  Critique A (planner perspective)
qwen3   →  Critique B (executor perspective)
         ↑
         └── loop until BOTH approve or max iterations reached
```

Both models must give a production readiness score ≥ 80 with no CRITICAL/HIGH issues before the output is accepted.

## Setup

```bash
# 1. Install Ollama from https://ollama.ai
# 2. Pull the models
ollama pull gemma4:e4b
ollama pull qwen3:8b

# 3. Install Python deps
pip install -r requirements.txt

# 4. Run
python -m maestro run "create a Python class for a banking system"
```

## Output

- Generated code → `output/`
- Execution logs → `runs/`

## Config

Edit `config.yaml` to change models, iteration limit, or score threshold.
