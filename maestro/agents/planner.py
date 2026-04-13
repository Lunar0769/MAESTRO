"""
Planner agent — uses gemma4:e4b to analyse the task and produce a Specification.
Memory context is injected so it learns from past runs.
"""
from maestro.schemas.models import Specification
from maestro.memory import store

_BASE_SYSTEM = """You are a senior software architect (Planner).
Your ONLY job is to analyse a task and produce a structured specification.

RULES:
- Do NOT write any code
- Extract clear, testable requirements with IDs (REQ-001, REQ-002 …)
- Specify the programming language explicitly
- Be precise about acceptance criteria

OUTPUT: valid JSON matching this schema exactly:
{
  "task_understanding": "...",
  "language": "python",
  "requirements": [
    {"id": "REQ-001", "description": "...", "acceptance_criteria": "...", "priority": 8}
  ],
  "architecture": "...",
  "implementation_steps": ["step 1", "step 2"],
  "constraints": [],
  "risks": []
}

Output ONLY the JSON. No markdown. No explanation."""


class Planner:
    def __init__(self, provider):
        self.provider = provider

    def plan(self, task: str, language_hint: str = None) -> Specification:
        memory_block = store.build_memory_block()
        system = _BASE_SYSTEM
        if memory_block:
            system = _BASE_SYSTEM + f"\n\n{memory_block}"

        prompt = f'Task: "{task}"'
        if language_hint:
            prompt += f'\nLanguage hint: {language_hint}'
        prompt += "\n\nProduce the specification JSON now."

        data = self.provider.plan(system, prompt)
        return Specification(**data)
