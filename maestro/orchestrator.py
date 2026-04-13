"""
Maestro Orchestrator
────────────────────
Pipeline:
  gemma4:e4b  → Planner   (specification)
  qwen3:8b    → Builder   (code)
  gemma4:e4b  → Critic A  (planner perspective)
  qwen3:8b    → Critic B  (executor perspective)

Loop continues until BOTH critics approve OR max_iterations reached.
"""
import time
import yaml
from pathlib import Path
from typing import Optional

from maestro.providers.ollama_provider import OllamaProvider
from maestro.agents.planner  import Planner
from maestro.agents.builder  import Builder
from maestro.agents.critic   import Critic
from maestro.schemas.models  import FinalResult, CritiqueReport, BuildOutput, Specification
from maestro.utils.logger    import log_run
from maestro.utils.output_handler import OutputHandler
from maestro.memory          import store as mem
from maestro.memory.learning_engine import learn_from_critique, learn_from_success


def _load_cfg() -> dict:
    p = Path("config.yaml")
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def _both_approve(ca: CritiqueReport, cb: CritiqueReport) -> bool:
    return (not ca.fix_required) and (not cb.fix_required)


def _merge_critique(ca: CritiqueReport, cb: CritiqueReport) -> CritiqueReport:
    """Return the stricter of the two critiques for display."""
    return ca if ca.production_readiness_score <= cb.production_readiness_score else cb


class Orchestrator:
    def __init__(self):
        self.cfg      = _load_cfg()
        self.provider = OllamaProvider()
        self.planner  = Planner(self.provider)
        self.builder  = Builder(self.provider)
        self.critic_a = Critic(self.provider, role="planner")   # gemma4
        self.critic_b = Critic(self.provider, role="executor")  # qwen3
        self.output   = OutputHandler()
        self.max_iter = self.cfg.get("orchestration", {}).get("max_iterations", 15)

    # ── public entry point ────────────────────────────────────────────────────

    def run(self, task: str) -> FinalResult:
        t0 = time.time()
        warnings, errors = [], []

        # ── 1. Plan ──────────────────────────────────────────────────────────
        print("🧠 [gemma4] Planning …")
        try:
            spec = self.planner.plan(task)
            log_run("planner", spec.model_dump())
            print(f"   → {len(spec.requirements)} requirements | lang: {spec.language}")
        except Exception as e:
            errors.append(f"Planner failed: {e}")
            return self._error_result(task, errors, time.time() - t0)

        # ── 2. Build + dual-critic loop ───────────────────────────────────────
        build:    Optional[BuildOutput]    = None
        critique_a: Optional[CritiqueReport] = None
        critique_b: Optional[CritiqueReport] = None

        for iteration in range(1, self.max_iter + 1):
            label = f"(iter {iteration}/{self.max_iter})"

            # Build
            print(f"\n⚙️  [qwen3] Building {label} …")
            try:
                prev_critique = _merge_critique(critique_a, critique_b) \
                    if (critique_a and critique_b) else (critique_a or critique_b)
                build = self.builder.build(spec, prev_critique)
                log_run(f"builder_{iteration}", build.model_dump())
            except Exception as e:
                errors.append(f"Builder failed {label}: {e}")
                break

            # Critic A — gemma4 (planner perspective)
            print(f"🔍 [gemma4] Critiquing {label} …")
            try:
                critique_a = self.critic_a.critique(spec, build, task)
                log_run(f"critic_planner_{iteration}", critique_a.model_dump())
                _print_critique("gemma4", critique_a)
            except Exception as e:
                errors.append(f"Critic-A failed {label}: {e}")
                break

            # Critic B — qwen3 (executor perspective)
            print(f"🔍 [qwen3]  Critiquing {label} …")
            try:
                critique_b = self.critic_b.critique(spec, build, task)
                log_run(f"critic_executor_{iteration}", critique_b.model_dump())
                _print_critique("qwen3", critique_b)
            except Exception as e:
                errors.append(f"Critic-B failed {label}: {e}")
                break

            # Check approval
            if _both_approve(critique_a, critique_b):
                print(f"\n✅ Both models approved after {iteration} iteration(s)!")
                learn_from_success(task, critique_a.production_readiness_score, spec)
                break
            else:
                # ── Learn from this iteration's failures ──────────────────
                avg_score = (
                    critique_a.production_readiness_score +
                    critique_b.production_readiness_score
                ) / 2
                learn_from_critique(critique_a, critique_b, spec, task, avg_score)

                scores = (
                    f"gemma4={critique_a.production_readiness_score}/100  "
                    f"qwen3={critique_b.production_readiness_score}/100"
                )
                print(f"   ↻ Fixes needed — {scores}  [memory updated]")
        else:
            warnings.append(f"Max iterations ({self.max_iter}) reached without full approval")

        # ── 3. Save output ────────────────────────────────────────────────────
        output_file = None
        if build:
            try:
                fp = self.output.save_code(build.code, task)
                output_file = str(fp)
                print(f"\n💾 Saved → {fp}")
            except Exception as e:
                warnings.append(f"Could not save file: {e}")

        # ── 4. Log run to memory dataset ──────────────────────────────────────
        final_score = 0.0
        if critique_a and critique_b:
            final_score = (
                critique_a.production_readiness_score +
                critique_b.production_readiness_score
            ) / 2
        mem.log_run(
            task=task,
            score=final_score,
            iterations=iteration if build else 0,
            success=approved if 'approved' in dir() else False,
            output_file=output_file or "",
        )

        # ── 4. Display summary ────────────────────────────────────────────────
        if build and critique_a and critique_b:
            _print_summary(spec, build, critique_a, critique_b)

        approved = bool(
            critique_a and critique_b and _both_approve(critique_a, critique_b)
        )

        # ── 5. Display memory stats ───────────────────────────────────────────
        _print_memory_stats()

        return FinalResult(
            task_type="GENERATION",
            success=approved,
            specification=spec,
            build_output=build,
            planner_critique=critique_a,
            executor_critique=critique_b,
            iterations=iteration if build else 0,
            total_ms=(time.time() - t0) * 1000,
            output_file=output_file,
            warnings=warnings,
            errors=errors,
        )

    def _error_result(self, task, errors, elapsed) -> FinalResult:
        return FinalResult(
            task_type="ERROR",
            success=False,
            iterations=0,
            total_ms=elapsed * 1000,
            errors=errors,
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _print_memory_stats():
    from maestro.memory import store as m
    data = m._load()
    mistakes  = len(data.get("mistakes", []))
    practices = len(data.get("best_practices", []))
    rules     = len(data.get("prompt_rules", []))
    runs      = len(data.get("run_log", []))
    print(f"\n🧠 Memory: {mistakes} mistakes · {practices} best practices · "
          f"{rules} rules · {runs} total runs")


def _print_critique(model: str, c: CritiqueReport):
    status = "✅ APPROVED" if not c.fix_required else "⚠️  NEEDS FIXES"
    print(f"   [{model}] {status} | score={c.production_readiness_score}/100 "
          f"| CRIT={c.severity_summary.get('CRITICAL',0)} "
          f"HIGH={c.severity_summary.get('HIGH',0)}")


def _print_summary(spec, build, ca, cb):
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"\nRequirements ({len(spec.requirements)}):")
    for r in spec.requirements:
        ea = next((e for e in ca.requirement_evaluations if e.requirement_id == r.id), None)
        eb = next((e for e in cb.requirement_evaluations if e.requirement_id == r.id), None)
        sa = ea.status if ea else "?"
        sb = eb.status if eb else "?"
        icon = "✅" if sa == "SATISFIED" and sb == "SATISFIED" else "⚠️ "
        print(f"  {icon} {r.id}: {r.description[:55]}…")
        print(f"       gemma4={sa}  qwen3={sb}")

    print(f"\nScores:  gemma4={ca.production_readiness_score}/100  "
          f"qwen3={cb.production_readiness_score}/100")

    all_issues = {i.id: i for i in ca.issues + cb.issues}
    crit = [i for i in all_issues.values() if i.severity == "CRITICAL"]
    high = [i for i in all_issues.values() if i.severity == "HIGH"]
    if crit:
        print(f"\n🔴 CRITICAL ({len(crit)}):")
        for i in crit:
            print(f"  {i.id}: {i.description[:70]}")
    if high:
        print(f"\n🟠 HIGH ({len(high)}):")
        for i in high:
            print(f"  {i.id}: {i.description[:70]}")

    print("\n" + "=" * 60)
    print("💻 GENERATED CODE")
    print("=" * 60)
    print(build.code)
    print("=" * 60)


# ── entry point ───────────────────────────────────────────────────────────────

def run(task: str) -> FinalResult:
    return Orchestrator().run(task)
