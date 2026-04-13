"""
Pydantic models for all agent I/O.
"""
from __future__ import annotations
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from enum import Enum


# ── Enums ────────────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    GENERATION = "GENERATION"
    REVIEW     = "REVIEW"
    REFACTOR   = "REFACTOR"
    DEBUG      = "DEBUG"
    ANALYSIS   = "ANALYSIS"

class ReqStatus(str, Enum):
    SATISFIED         = "SATISFIED"
    PARTIALLY         = "PARTIALLY_SATISFIED"
    FAILED            = "FAILED"

    @classmethod
    def _missing_(cls, value):
        # Accept any variant the model might return
        mapping = {
            "partial": cls.PARTIALLY,
            "partially": cls.PARTIALLY,
            "partially satisfied": cls.PARTIALLY,
        }
        return mapping.get(str(value).strip().lower())

class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"
    INFO     = "INFO"


# ── Planner output ────────────────────────────────────────────────────────────

class Requirement(BaseModel):
    id:                 str
    description:        str
    acceptance_criteria: str
    priority:           int = Field(ge=1, le=10)

class Specification(BaseModel):
    task_understanding:   str
    language:             str
    requirements:         List[Requirement]
    architecture:         str
    implementation_steps: List[str]
    constraints:          List[str] = []
    risks:                List[str] = []


# ── Executor output ───────────────────────────────────────────────────────────

class BuildOutput(BaseModel):
    code:                    str
    language:                str
    filename:                str
    dependencies:            List[str] = []
    addressed_requirements:  List[str]
    implementation_notes:    str = ""


# ── Critic output ─────────────────────────────────────────────────────────────

class ReqEval(BaseModel):
    requirement_id: str
    status:         ReqStatus
    evidence:       str
    reasoning:      str

class Issue(BaseModel):
    id:             str
    severity:       Severity
    category:       str
    description:    str
    location:       Optional[str] = None
    recommendation: str

class CritiqueReport(BaseModel):
    requirement_evaluations:  List[ReqEval]
    issues:                   List[Issue]
    overall_quality_score:    int = Field(ge=0, le=100)
    production_readiness_score: int = Field(ge=0, le=100)
    severity_summary:         Dict[str, int]
    fix_required:             bool
    blocking_issues:          List[str]
    recommendations:          List[str]
    intent_preserved:         bool
    language_drift_detected:  bool


# ── Final result ──────────────────────────────────────────────────────────────

class FinalResult(BaseModel):
    task_type:           str
    success:             bool
    specification:       Optional[Specification]  = None
    build_output:        Optional[BuildOutput]    = None
    planner_critique:    Optional[CritiqueReport] = None
    executor_critique:   Optional[CritiqueReport] = None
    iterations:          int
    total_ms:            float
    output_file:         Optional[str]            = None
    warnings:            List[str]                = []
    errors:              List[str]                = []
