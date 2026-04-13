"""
Layer 1 — MemoryManager
Clean persistent store with deduplication and ranking.
File: memory/maestro_memory.json
"""
import json
import os
from datetime import datetime
from typing import List, Dict

MEMORY_FILE = "memory/maestro_memory.json"


class MemoryManager:
    def __init__(self):
        os.makedirs("memory", exist_ok=True)
        if not os.path.exists(MEMORY_FILE):
            self._init_memory()
        self.memory = self._load()

    def _init_memory(self):
        with open(MEMORY_FILE, "w") as f:
            json.dump({
                "mistakes":       [],
                "best_practices": [],
                "prompt_rules":   [],
                "task_history":   [],
            }, f, indent=2)

    def _load(self) -> dict:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self):
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)

    # ── Task history (Layer 1 raw log) ────────────────────────────────────────

    def store_task(self, task: str, output: str, score: float, feedback: str):
        self.memory["task_history"].append({
            "task":      task,
            "output":    output[:1000],   # cap size
            "score":     score,
            "feedback":  feedback[:500],
            "timestamp": str(datetime.now()),
        })
        self.save()

    # ── Mistakes (Layer 2 extracted learning) ─────────────────────────────────

    def add_mistake(self, pattern: str, fix: str):
        """Deduplicate by pattern, increment count if exists."""
        pattern = pattern.strip()[:150]
        fix     = fix.strip()[:250]
        for m in self.memory["mistakes"]:
            if m["pattern"].lower() == pattern.lower():
                m["count"] = m.get("count", 1) + 1
                m["fix"]   = fix          # update fix with latest
                self.save()
                return
        self.memory["mistakes"].append({"pattern": pattern, "fix": fix, "count": 1})
        self.save()

    # ── Best practices ────────────────────────────────────────────────────────

    def add_best_practice(self, pattern: str, solution: str):
        """Deduplicate by pattern, increment count if exists."""
        pattern  = pattern.strip()[:150]
        solution = solution.strip()[:250]
        for bp in self.memory["best_practices"]:
            if bp["pattern"].lower() == pattern.lower():
                bp["count"] = bp.get("count", 1) + 1
                self.save()
                return
        self.memory["best_practices"].append({"pattern": pattern, "solution": solution, "count": 1})
        self.save()

    # ── Prompt rules (Layer 3 actionable) ────────────────────────────────────

    def add_prompt_rule(self, rule: str):
        """Add rule if not already present."""
        rule = rule.strip()
        if rule and rule not in self.memory["prompt_rules"]:
            self.memory["prompt_rules"].append(rule)
            self.save()

    def remove_prompt_rule(self, rule: str):
        self.memory["prompt_rules"] = [r for r in self.memory["prompt_rules"] if r != rule]
        self.save()

    def prune(self, max_mistakes: int = 20, max_practices: int = 20, max_rules: int = 10):
        """
        Keep only the highest-count entries.
        Prevents memory from becoming a garbage dump.
        """
        self.memory["mistakes"] = sorted(
            self.memory["mistakes"], key=lambda x: x.get("count", 1), reverse=True
        )[:max_mistakes]

        self.memory["best_practices"] = sorted(
            self.memory["best_practices"], key=lambda x: x.get("count", 1), reverse=True
        )[:max_practices]

        # Rules: keep most recent (most relevant)
        self.memory["prompt_rules"] = self.memory["prompt_rules"][-max_rules:]

        self.save()

    # ── Context builder (injected into every prompt) ──────────────────────────

    def get_learning_context(self) -> str:
        """
        Returns a compact block injected into Planner + Builder prompts.
        Only top-ranked entries — no garbage.
        """
        self.memory = self._load()   # always fresh

        top_rules    = self.memory["prompt_rules"][-10:]
        top_mistakes = sorted(
            self.memory["mistakes"], key=lambda x: x.get("count", 1), reverse=True
        )[:5]
        top_practices = sorted(
            self.memory["best_practices"], key=lambda x: x.get("count", 1), reverse=True
        )[:5]

        lines = []

        if top_rules:
            lines.append("ALWAYS APPLY THESE RULES:")
            for r in top_rules:
                lines.append(f"  • {r}")

        if top_mistakes:
            lines.append("COMMON MISTAKES — AVOID:")
            for m in top_mistakes:
                lines.append(f"  ✗ [{m.get('count',1)}x] {m['pattern']} → {m['fix']}")

        if top_practices:
            lines.append("PROVEN BEST PRACTICES:")
            for bp in top_practices:
                lines.append(f"  ✓ [{bp.get('count',1)}x] {bp['pattern']} → {bp['solution']}")

        return "\n".join(lines) if lines else ""

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        self.memory = self._load()
        return {
            "mistakes":       len(self.memory["mistakes"]),
            "best_practices": len(self.memory["best_practices"]),
            "prompt_rules":   len(self.memory["prompt_rules"]),
            "task_history":   len(self.memory["task_history"]),
        }
