"""
Builder agent — uses qwen3:8b to generate code from a Specification.
Memory context is injected so it avoids past mistakes.
"""
from maestro.schemas.models import Specification, BuildOutput, CritiqueReport
from typing import Optional

_BASE_SYSTEM = """You are an expert software engineer (Builder).
Your ONLY job is to write production-quality code from a specification.

RULES:
- Address EVERY requirement listed
- Use the EXACT language from the specification
- Include comprehensive error handling
- Add docstrings and type hints
- NEVER change the programming language
- Escape ALL newlines in the "code" field as \\n (JSON requirement)
- Escape ALL quotes in the "code" field as \\"

OUTPUT: valid JSON matching this schema exactly:
{
  "code": "full code with \\n for newlines",
  "language": "python",
  "filename": "suggested_name.py",
  "dependencies": [],
  "addressed_requirements": ["REQ-001", "REQ-002"],
  "implementation_notes": "..."
}

Output ONLY the JSON. No markdown. No explanation."""


class Builder:
    def __init__(self, provider, memory=None):
        self.provider = provider
        self.memory   = memory

    def build(self, spec: Specification, critique: Optional[CritiqueReport] = None) -> BuildOutput:
        context = self.memory.get_learning_context() if self.memory else ""
        system  = _BASE_SYSTEM + (f"\n\n{context}" if context else "")
        prompt  = self._build_prompt(spec, critique)
        data    = self.provider.execute(system, prompt)
        return BuildOutput(**data)

    def _build_prompt(self, spec: Specification, critique: Optional[CritiqueReport]) -> str:
        lines = [
            f"Language: {spec.language}",
            f"Task: {spec.task_understanding}",
            "",
            "Requirements:",
        ]
        for r in spec.requirements:
            lines.append(f"  {r.id}: {r.description}")
            lines.append(f"    Acceptance: {r.acceptance_criteria}")

        lines += ["", f"Architecture: {spec.architecture}", "", "Steps:"]
        for i, s in enumerate(spec.implementation_steps, 1):
            lines.append(f"  {i}. {s}")

        if critique:
            lines += ["", "=" * 40, "PREVIOUS CRITIQUE — FIX THESE:"]
            failed = [e for e in critique.requirement_evaluations
                      if e.status in ("FAILED", "PARTIALLY_SATISFIED")]
            for e in failed:
                lines.append(f"  {e.requirement_id} [{e.status}]: {e.reasoning}")

            blocking = [i for i in critique.issues
                        if i.severity in ("CRITICAL", "HIGH")]
            for i in blocking:
                lines.append(f"  {i.id} [{i.severity}]: {i.description}")
                lines.append(f"    Fix: {i.recommendation}")

        lines.append("\nGenerate the implementation JSON now.")
        return "\n".join(lines)
