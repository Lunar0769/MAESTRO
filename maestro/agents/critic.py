"""
Critic agent — can run as EITHER model (planner or executor).
Both must approve before the loop exits.
"""
from maestro.schemas.models import Specification, BuildOutput, CritiqueReport

SYSTEM = """You are a strict code reviewer. You MUST respond with ONLY a JSON object. No text before or after. No markdown. No explanation. No headers. No bullet points. ONLY the raw JSON object starting with { and ending with }.

Evaluate the code and return this exact JSON structure:
{
  "requirement_evaluations": [
    {"requirement_id": "REQ-001", "status": "SATISFIED", "evidence": "code shows...", "reasoning": "because..."}
  ],
  "issues": [
    {"id": "ISS-001", "severity": "HIGH", "category": "security", "description": "...", "location": "...", "recommendation": "..."}
  ],
  "overall_quality_score": 75,
  "production_readiness_score": 70,
  "severity_summary": {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 0, "LOW": 0, "INFO": 0},
  "fix_required": true,
  "blocking_issues": ["ISS-001"],
  "recommendations": ["fix the issue"],
  "intent_preserved": true,
  "language_drift_detected": false
}

SCORING RULES:
- Start at 0. Add: +40 all SATISFIED, +15 no CRITICAL, +15 no HIGH, +10 error handling, +10 docs, +5 validation, +5 logging
- Subtract: -20 per CRITICAL, -10 per HIGH, -5 per MEDIUM, -15 per FAILED req, -5 per PARTIAL req
- fix_required=true if ANY critical/high issue OR any failed/partial req OR score < 80

REMEMBER: Your entire response must be a single JSON object. Start your response with { immediately."""


class Critic:
    def __init__(self, provider, role: str = "planner"):
        """
        role: "planner" → uses gemma4:e4b
              "executor" → uses qwen3:8b
        """
        self.provider = provider
        self.role = role

    def critique(
        self,
        spec: Specification,
        build: BuildOutput,
        original_task: str
    ) -> CritiqueReport:
        prompt = self._build_prompt(spec, build, original_task)

        if self.role == "planner":
            data = self.provider.plan(SYSTEM, prompt)
        else:
            data = self.provider.execute(SYSTEM, prompt)

        return CritiqueReport(**data)

    def _build_prompt(
        self,
        spec: Specification,
        build: BuildOutput,
        original_task: str
    ) -> str:
        lines = [
            f"Original task: {original_task}",
            f"Specification language: {spec.language}",
            "",
            "Requirements to evaluate:",
        ]
        for r in spec.requirements:
            lines.append(f"  {r.id}: {r.description}")
            lines.append(f"    Acceptance: {r.acceptance_criteria}")

        lines += [
            "",
            "Code to review:",
            "```",
            build.code,
            "```",
            "",
            "Evaluate strictly. Output the critique JSON now.",
        ]
        return "\n".join(lines)
