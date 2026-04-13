"""
Learning Engine — runs after every critique cycle.
Extracts patterns, updates memory, evolves prompt rules.
"""
from maestro.schemas.models import CritiqueReport, Specification
from maestro.memory import store

# Issue categories → actionable rules
_CATEGORY_RULES = {
    "security":      "Always use secure libraries (bcrypt for passwords, parameterized queries for SQL)",
    "error_handling":"Always wrap critical operations in try/except with meaningful error messages",
    "documentation": "Always add docstrings, inline comments, and usage examples",
    "style":         "Always follow language style guides (PEP8 for Python, BEM for CSS)",
    "performance":   "Always consider time/space complexity and avoid unnecessary loops",
    "validation":    "Always validate all user inputs before processing",
    "accessibility": "Always add ARIA labels, keyboard navigation, and sufficient color contrast",
    "ui":            "Always include loading states, error states, and success feedback",
}


def learn_from_critique(
    critique_a: CritiqueReport,
    critique_b: CritiqueReport,
    spec: Specification,
    task: str,
    final_score: float,
):
    """
    Called after every iteration.
    Extracts mistakes and best practices, updates prompt rules.
    """
    all_issues = critique_a.issues + critique_b.issues

    # ── Extract mistakes from CRITICAL/HIGH issues ────────────────────────────
    for issue in all_issues:
        if issue.severity in ("CRITICAL", "HIGH"):
            store.add_mistake(
                pattern=issue.description[:120],
                fix=issue.recommendation[:200],
            )
            # Promote category-level rule
            rule = _CATEGORY_RULES.get(issue.category.lower())
            if rule:
                store.add_prompt_rule(rule)

    # ── Extract best practices from high-scoring runs ─────────────────────────
    if final_score >= 85:
        # What did the spec say that led to a good result?
        for req in spec.requirements:
            store.add_best_practice(
                pattern=req.description[:100],
                solution=req.acceptance_criteria[:200],
            )
        # Task-level pattern
        task_words = task.lower().split()
        for keyword in ["login", "ui", "api", "auth", "dashboard", "form", "animation"]:
            if keyword in task_words:
                store.add_best_practice(
                    pattern=f"{keyword} task",
                    solution=f"Score {final_score:.0f} achieved with: {spec.architecture[:150]}",
                )

    # ── Prune low-value prompt rules (keep top 10 by frequency) ──────────────
    _prune_rules()


def _prune_rules():
    """Keep only the most impactful rules (max 10)."""
    data = store._load()
    rules = data.get("prompt_rules", [])
    if len(rules) > 10:
        # Keep last 10 (most recently added = most relevant)
        data["prompt_rules"] = rules[-10:]
        store._save(data)


def learn_from_success(task: str, score: float, spec: Specification):
    """Called when both critics approve — reinforce what worked."""
    store.add_best_practice(
        pattern=f"successful: {task[:80]}",
        solution=f"architecture: {spec.architecture[:200]}",
    )
    # Reward: add a rule based on the task type
    task_l = task.lower()
    if any(w in task_l for w in ["login", "auth", "password"]):
        store.add_prompt_rule("For auth tasks: always use bcrypt, add rate limiting, validate inputs")
    if any(w in task_l for w in ["ui", "page", "html", "css", "design"]):
        store.add_prompt_rule("For UI tasks: always add animations, toast notifications, responsive design, accessibility")
    if any(w in task_l for w in ["api", "endpoint", "rest"]):
        store.add_prompt_rule("For API tasks: always add input validation, error responses, status codes, rate limiting")
