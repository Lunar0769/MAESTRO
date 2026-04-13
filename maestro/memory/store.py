"""
Persistent memory store — survives across runs.
Saved to memory/knowledge.json
"""
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

_PATH = Path("memory/knowledge.json")
_PATH.parent.mkdir(exist_ok=True)

_EMPTY = {
    "mistakes":       [],   # recurring issues the system has seen
    "best_practices": [],   # patterns that led to high scores
    "prompt_rules":   [],   # dynamic rules injected into every prompt
    "run_log":        [],   # every completed run (task, score, iterations)
}


def _load() -> dict:
    if _PATH.exists():
        try:
            return json.loads(_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {k: list(v) for k, v in _EMPTY.items()}


def _save(data: dict):
    _PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── public API ────────────────────────────────────────────────────────────────

def get_context() -> dict:
    """Return the full memory store."""
    return _load()


def add_mistake(pattern: str, fix: str):
    """Record a recurring mistake and its fix."""
    data = _load()
    for m in data["mistakes"]:
        if m["pattern"].lower() == pattern.lower():
            m["frequency"] = m.get("frequency", 1) + 1
            m["fix"] = fix
            _save(data)
            return
    data["mistakes"].append({"pattern": pattern, "fix": fix, "frequency": 1})
    _save(data)


def add_best_practice(pattern: str, solution: str):
    """Record a pattern that produced high-quality output."""
    data = _load()
    for bp in data["best_practices"]:
        if bp["pattern"].lower() == pattern.lower():
            bp["frequency"] = bp.get("frequency", 1) + 1
            _save(data)
            return
    data["best_practices"].append({"pattern": pattern, "solution": solution, "frequency": 1})
    _save(data)


def add_prompt_rule(rule: str):
    """Add a dynamic rule to inject into every future prompt."""
    data = _load()
    if rule not in data["prompt_rules"]:
        data["prompt_rules"].append(rule)
        _save(data)


def remove_prompt_rule(rule: str):
    data = _load()
    data["prompt_rules"] = [r for r in data["prompt_rules"] if r != rule]
    _save(data)


def log_run(task: str, score: float, iterations: int, success: bool, output_file: str = ""):
    """Append a run record for dataset creation."""
    data = _load()
    data["run_log"].append({
        "task":        task,
        "score":       score,
        "iterations":  iterations,
        "success":     success,
        "output_file": output_file,
        "timestamp":   datetime.now().isoformat(),
    })
    _save(data)


def top_mistakes(n: int = 5) -> List[Dict]:
    data = _load()
    return sorted(data["mistakes"], key=lambda x: x.get("frequency", 1), reverse=True)[:n]


def top_practices(n: int = 5) -> List[Dict]:
    data = _load()
    return sorted(data["best_practices"], key=lambda x: x.get("frequency", 1), reverse=True)[:n]


def prompt_rules() -> List[str]:
    return _load()["prompt_rules"]


def build_memory_block() -> str:
    """
    Returns a formatted string to inject into every agent prompt.
    This is the core of the self-improving loop.
    """
    data = _load()
    lines = []

    mistakes = sorted(data["mistakes"], key=lambda x: x.get("frequency", 1), reverse=True)[:5]
    if mistakes:
        lines.append("LEARNED MISTAKES TO AVOID:")
        for m in mistakes:
            lines.append(f"  ✗ {m['pattern']} → Fix: {m['fix']}")

    practices = sorted(data["best_practices"], key=lambda x: x.get("frequency", 1), reverse=True)[:5]
    if practices:
        lines.append("PROVEN BEST PRACTICES:")
        for bp in practices:
            lines.append(f"  ✓ {bp['pattern']} → {bp['solution']}")

    rules = data["prompt_rules"]
    if rules:
        lines.append("ALWAYS APPLY THESE RULES:")
        for r in rules:
            lines.append(f"  • {r}")

    return "\n".join(lines) if lines else ""
