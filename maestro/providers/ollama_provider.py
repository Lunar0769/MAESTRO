"""
Ollama provider — fast, parallel-ready calls to local models.
"""
import json
import re
import yaml
import ollama
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def _load_config() -> dict:
    cfg = Path("config.yaml")
    if cfg.exists():
        return yaml.safe_load(cfg.read_text())
    return {}


def _extract_json(text: str) -> dict:
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
        candidate = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "",
                           stripped[start:end + 1])
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON:\n{text[:300]}")


_STATUS_MAP = {
    "partial": "PARTIALLY_SATISFIED", "partially": "PARTIALLY_SATISFIED",
    "partially satisfied": "PARTIALLY_SATISFIED",
    "partially_satisfied": "PARTIALLY_SATISFIED",
    "satisfied": "SATISFIED", "failed": "FAILED",
    "not_applicable": "NOT_APPLICABLE", "not applicable": "NOT_APPLICABLE",
}
_SEVERITY_MAP = {
    "critical": "CRITICAL", "high": "HIGH",
    "medium": "MEDIUM", "med": "MEDIUM",
    "low": "LOW", "info": "INFO", "information": "INFO",
}


def _normalise(data: dict) -> dict:
    for ev in data.get("requirement_evaluations", []):
        raw = str(ev.get("status", "")).strip().lower()
        ev["status"] = _STATUS_MAP.get(raw, ev.get("status", "FAILED"))
    for issue in data.get("issues", []):
        raw = str(issue.get("severity", "")).strip().lower()
        issue["severity"] = _SEVERITY_MAP.get(raw, issue.get("severity", "MEDIUM"))
    return data


class OllamaProvider:
    def __init__(self):
        cfg = _load_config().get("ollama", {})
        self.base_url = cfg.get("base_url", "http://localhost:11434")
        self.planner  = cfg.get("planner_model",  "gemma4:e4b")
        self.executor = cfg.get("executor_model", "qwen3:8b")
        self._client  = ollama.Client(host=self.base_url)
        # Token limits — keep responses tight and fast
        self._options = {
            "temperature":  0.1,   # lower = faster, more deterministic
            "num_predict":  1024,  # max tokens to generate
            "num_ctx":      4096,  # context window (smaller = faster)
        }

    def _call(self, model: str, system: str, prompt: str, json_mode: bool = False) -> str:
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            options=self._options,
        )
        if json_mode:
            kwargs["format"] = "json"
        resp = self._client.chat(**kwargs)
        return resp["message"]["content"]

    def plan(self, system: str, prompt: str) -> dict:
        raw = self._call(self.planner, system, prompt, json_mode=True)
        return _normalise(_extract_json(raw))

    def execute(self, system: str, prompt: str) -> dict:
        raw = self._call(self.executor, system, prompt, json_mode=True)
        return _normalise(_extract_json(raw))

    def plan_text(self, system: str, prompt: str) -> str:
        return self._call(self.planner, system, prompt)

    def execute_text(self, system: str, prompt: str) -> str:
        return self._call(self.executor, system, prompt)

    def dual_critique(
        self,
        system_a: str, prompt_a: str,
        system_b: str, prompt_b: str,
    ) -> tuple:
        """
        Run both critics IN PARALLEL using threads.
        Returns (result_a, result_b) — both are dicts.
        Cuts critique time roughly in half.
        """
        results = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(self._call, self.planner,  system_a, prompt_a, True)
            fb = pool.submit(self._call, self.executor, system_b, prompt_b, True)
            results["a"] = _normalise(_extract_json(fa.result()))
            results["b"] = _normalise(_extract_json(fb.result()))
        return results["a"], results["b"]
