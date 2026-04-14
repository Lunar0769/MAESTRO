"""
Critic agent — runs as planner (gemma4) or executor (qwen3).
Prompt is kept SHORT to reduce token cost and latency.
"""
from maestro.schemas.models import Specification, BuildOutput, CritiqueReport

# Compact system prompt — fewer tokens = faster response
SYSTEM = """Strict code reviewer. Respond with ONLY a JSON object starting with {

JSON schema:
{"requirement_evaluations":[{"requirement_id":"REQ-001","status":"SATISFIED","evidence":"...","reasoning":"..."}],"issues":[{"id":"ISS-001","severity":"HIGH","category":"security","description":"...","location":"...","recommendation":"..."}],"overall_quality_score":75,"production_readiness_score":70,"severity_summary":{"CRITICAL":0,"HIGH":1,"MEDIUM":0,"LOW":0,"INFO":0},"fix_required":true,"blocking_issues":["ISS-001"],"recommendations":["..."],"intent_preserved":true,"language_drift_detected":false}

SCORING: start 0, +40 all SATISFIED, +15 no CRITICAL, +15 no HIGH, +10 error handling, +10 docs, +5 validation, +5 logging. -20/CRITICAL -10/HIGH -5/MEDIUM -15/FAILED -5/PARTIAL.
fix_required=true if CRITICAL/HIGH issue OR failed/partial req OR score<80."""


class Critic:
    def __init__(self, provider, role: str = "planner"):
        self.provider = provider
        self.role     = role

    def critique(self, spec: Specification, build: BuildOutput, task: str) -> CritiqueReport:
        prompt = self._prompt(spec, build, task)
        if self.role == "planner":
            data = self.provider.plan(SYSTEM, prompt)
        else:
            data = self.provider.execute(SYSTEM, prompt)
        return CritiqueReport(**data)

    def _prompt(self, spec: Specification, build: BuildOutput, task: str) -> str:
        # Send only first 2000 chars of code to keep tokens low
        code_preview = build.code[:2000] + ("…[truncated]" if len(build.code) > 2000 else "")
        reqs = "\n".join(f"  {r.id}: {r.description} | accept: {r.acceptance_criteria}"
                         for r in spec.requirements)
        return (
            f"Task: {task}\nLanguage: {spec.language}\n\n"
            f"Requirements:\n{reqs}\n\n"
            f"Code:\n```\n{code_preview}\n```\n\n"
            "Output critique JSON now."
        )
