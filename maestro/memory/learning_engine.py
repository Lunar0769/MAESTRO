"""Layer 2 - LearningEngine"""
import json
from typing import List, Dict, Tuple
from maestro.schemas.models import CritiqueReport, Specification

_KW = [
    ("no validation","missing input validation","validate all inputs"),
    ("missing validation","missing input validation","validate all inputs"),
    ("no feedback","no user feedback","add toast notifications"),
    ("missing feedback","no user feedback","add toast notifications"),
    ("accessibility","poor accessibility","add ARIA labels and keyboard nav"),
    ("no error handling","missing error handling","wrap in try/except"),
    ("sha-256","weak password hashing","use bcrypt or argon2"),
    ("sha256","weak password hashing","use bcrypt or argon2"),
    ("sql injection","SQL injection risk","use parameterized queries"),
    ("no docstring","missing documentation","add docstrings"),
    ("hardcoded","hardcoded secrets","use environment variables"),
    ("no animation","static UI","add CSS transitions"),
    ("no responsive","not responsive","add responsive breakpoints"),
    ("no toast","no user feedback","add toast notifications"),
]

def _keyword_extract(text):
    t,results,seen=text.lower(),[],set()
    for trigger,pattern,fix in _KW:
        if trigger in t and pattern not in seen:
            results.append({"pattern":pattern,"fix":fix})
            seen.add(pattern)
    return results

def _llm_extract(provider,feedback_text):
    prompt=f"Extract learning patterns from this feedback as JSON array [{{'pattern':'...','fix':'...'}}]:\n{feedback_text[:1000]}"
    system="Return only a valid JSON array of learning patterns."
    try:
        raw=provider._call(provider.executor,system,prompt,json_mode=True)
        if isinstance(raw,str):
            s,e=raw.find("["),raw.rfind("]")
            if s!=-1 and e!=-1: raw=raw[s:e+1]
            data=json.loads(raw)
        else: data=raw
        if isinstance(data,list): return [d for d in data if "pattern" in d and "fix" in d]
    except: pass
    return []

class LearningEngine:
    def __init__(self,memory_manager,provider=None):
        self.memory=memory_manager
        self.provider=provider

    def process_critique(self,critique_a,critique_b,spec,task,score):
        parts=[]
        for issue in critique_a.issues+critique_b.issues:
            parts.append(f"{issue.severity}: {issue.description}. Fix: {issue.recommendation}")
        for ev in critique_a.requirement_evaluations+critique_b.requirement_evaluations:
            if ev.status in ("FAILED","PARTIALLY_SATISFIED"):
                parts.append(f"Req {ev.requirement_id} {ev.status}: {ev.reasoning}")
        feedback="\n".join(parts)
        patterns=_llm_extract(self.provider,feedback) if self.provider else []
        existing={p["pattern"].lower() for p in patterns}
        for kp in _keyword_extract(feedback):
            if kp["pattern"].lower() not in existing: patterns.append(kp)
        for p in patterns:
            self.memory.add_mistake(p["pattern"],p["fix"])
            self.memory.add_prompt_rule(f"Always: {p['fix']}")
        for issue in critique_a.issues+critique_b.issues:
            if issue.severity in ("CRITICAL","HIGH"):
                self.memory.add_mistake(issue.description[:120],issue.recommendation[:200])
        self.memory.store_task(task,feedback,score,f"Issues: {len(critique_a.issues+critique_b.issues)}")
        self.memory.prune()

    def process_success(self,task,score,spec):
        self.memory.add_best_practice(f"task: {task[:80]}",f"arch: {spec.architecture[:200]}")
        t=task.lower()
        if any(w in t for w in ["login","auth","password"]):
            self.memory.add_prompt_rule("Auth: use bcrypt, rate limiting, validate inputs, clear errors")
        if any(w in t for w in ["ui","page","html","css","design","form"]):
            self.memory.add_prompt_rule("UI: animations, toast notifications, responsive, ARIA accessibility")
        if any(w in t for w in ["api","endpoint","rest"]):
            self.memory.add_prompt_rule("API: validate inputs, proper status codes, rate limiting, log errors")
        if any(w in t for w in ["database","sql","query"]):
            self.memory.add_prompt_rule("DB: parameterized queries, transactions, handle connection errors")
        self.memory.store_task(task,spec.architecture,score,"APPROVED by both models")
        self.memory.prune()
