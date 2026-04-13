"""
Ollama provider — calls local Ollama models, returns parsed JSON or plain text.
"""
import json
import re
import yaml
import ollama
from pathlib import Path


def _load_config() -> dict:
    cfg = Path("config.yaml")
    if cfg.exists():
        return yaml.safe_load(cfg.read_text())
    return {}


def _extract_json(text: str) -> dict:
    """Extract JSON from model response, handling markdown fences and prose."""
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for fence in ["```json", "```"]:
        if fence in stripped:
            try:
                inner = stripped.split(fence)[1].split("```")[0].strip()
                return json.loads(inner)
            except (IndexError, json.JSONDecodeError):
                pass

    start = stripped.find("{")
    end   = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start:end + 1]
        candidate = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from model response:\n{text[:400]}")


# Normalise values that models commonly get wrong
_STATUS_MAP = {
    "partial":              "PARTIALLY_SATISFIED",
    "partially":            "PARTIALLY_SATISFIED",
    "partially satisfied":  "PARTIALLY_SATISFIED",
    "partially_satisfied":  "PARTIALLY_SATISFIED",
    "satisfied":            "SATISFIED",
    "failed":               "FAILED",
    "not_applicable":       "NOT_APPLICABLE",
    "not applicable":       "NOT_APPLICABLE",
}
_SEVERITY_MAP = {
    "critical": "CRITICAL",
    "high":     "HIGH",
    "medium":   "MEDIUM",
    "med":      "MEDIUM",
    "low":      "LOW",
    "info":     "INFO",
    "information": "INFO",
}

def _normalise(data: dict) -> dict:
    """Fix enum values that models return in non-standard forms."""
    for ev in data.get("requirement_evaluations", []):
        raw = str(ev.get("status", "")).strip().lower()
        ev["status"] = _STATUS_MAP.get(raw, ev.get("status", "FAILED"))

    for issue in data.get("issues", []):
        raw = str(issue.get("severity", "")).strip().lower()
        issue["severity"] = _SEVERITY_MAP.get(raw, issue.get("severity", "MEDIUM"))

    return data


class OllamaProvider:
    """Thin wrapper around the Ollama Python client."""

    def __init__(self):
        cfg = _load_config().get("ollama", {})
        self.base_url    = cfg.get("base_url", "http://localhost:11434")
        self.planner     = cfg.get("planner_model", "gemma4:e4b")
        self.executor    = cfg.get("executor_model", "qwen3:8b")
        self._client     = ollama.Client(host=self.base_url)

    def _call(self, model: str, system: str, prompt: str, json_mode: bool = False) -> str:
        """Raw call — returns content string."""
        options = {"temperature": 0.2}
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            options=options,
        )
        if json_mode:
            kwargs["format"] = "json"
        resp = self._client.chat(**kwargs)
        return resp["message"]["content"]

    def plan(self, system: str, prompt: str) -> dict:
        """Call planner model, return parsed JSON."""
        raw = self._call(self.planner, system, prompt, json_mode=True)
        return _normalise(_extract_json(raw))

    def execute(self, system: str, prompt: str) -> dict:
        """Call executor model, return parsed JSON."""
        raw = self._call(self.executor, system, prompt, json_mode=True)
        return _normalise(_extract_json(raw))

    def plan_text(self, system: str, prompt: str) -> str:
        """Call planner model, return plain text."""
        return self._call(self.planner, system, prompt)

    def execute_text(self, system: str, prompt: str) -> str:
        """Call executor model, return plain text."""
        return self._call(self.executor, system, prompt)
