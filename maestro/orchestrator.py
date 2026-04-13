"""
Maestro Orchestrator
────────────────────
Pipeline:
  gemma4:e4b  → Planner   (specification)
  qwen3:8b    → Builder   (code)
  gemma4:e4b  → Critic A  (planner perspective)
  qwen3:8b    → Critic B  (executor perspective)
  LearningEngine → extract patterns → update memory
  MemoryManager  → inject context into next iteration

Loop continues until BOTH critics approve OR max_iterations reached.
"""
import time
import yaml
from pathlib import Path
from typing import Optional

from maestro.providers.ollama_provider import OllamaProvider
from maestro.agents.planner            import Planner
from maestro.agents.builder            import Builder
from maestro.agents.critic             import Critic
from maestro.schemas.models            import FinalResult, CritiqueReport, BuildOutput, Specification
from maestro.utils.logger              import log_run
from maestro.utils.output_handler      import OutputHandler
from maestro.memory.manager            import MemoryManager
from maestro.memory.learning_engine    import LearningEngine


def _load_cfg() -> dict:
    p = Path("config.yaml")
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def _both_approve(ca: CritiqueReport, cb: CritiqueReport) -> bool:
    return (not ca.fix_required) and (not cb.fix_required)


def _stricter(ca: CritiqueReport, cb: CritiqueReport) -> CritiqueReport:
    return ca if ca.production_readiness_score <= cb.production_readiness_score else cb


class Orchestrator:
    def __init__(self):
        self.cfg      = _load_cfg()
        self.provider = OllamaProvider()
        self.memory   = MemoryManager()
        self.learner  = LearningEngine(self.memory, self.provider)
        self.planner  = Planner(self.provider, self.memory)
        self.builder  = Builder(self.provider, self.memory)
        self.critic_a = Critic(self.provider, role="planner")
        self.critic_b = Critic(self.provider, role="executor")
        self.output   = OutputHandler()
        self.max_iter = self.cfg.get("orchestration", {}).get("max_iterations", 15)

    def run(self, task: str) -> FinalResult:
        t0 = time.time()
        warnings, errors = [], []
        iteration = 0

        # ── 1. Plan ───────────────────────────────────────────────────────────
        print("🧠 [gemma4] Planning …")
        try:
            spec = self.planner.plan(task)
            log_run("planner", spec.model_dump())
            print(f"   → {len(spec.requirements)} requirements | lang: {spec.language}")
        except Exception as e:
            errors.append(f"Planner failed: {e}")
            return self._error(task, errors, time.time() - t0)

        # ── 2. Build + dual-critic loop ───────────────────────────────────────
        build:      Optional[BuildOutput]    = None
        critique_a: Optional[CritiqueReport] = None
        critique_b: Optional[CritiqueReport] = None
        approved   = False

        for iteration in range(1, self.max_iter + 1):
            label = f"(iter {iteration}/{self.max_iter})"

            # Build
            print(f"\n⚙️  [qwen3] Building {label} …")
            try:
                prev = _stricter(critique_a, critique_b) \
                    if (critique_a and critique_b) else (critique_a or critique_b)
                build = self.builder.build(spec, prev)
                log_run(f"builder_{iteration}", build.model_dump())
            except Exception as e:
                errors.append(f"Builder failed {label}: {e}")
                break

            # Critic A — gemma4
            print(f"🔍 [gemma4] Critiquing {label} …")
            try:
                critique_a = self.critic_a.critique(spec, build, task)
                log_run(f"critic_planner_{iteration}", critique_a.model_dump())
                _show_critique("gemma4", critique_a)
            except Exception as e:
                errors.append(f"Critic-A failed {label}: {e}")
                break

            # Critic B — qwen3
            print(f"🔍 [qwen3]  Critiquing {label} …")
            try:
                critique_b = self.critic_b.critique(spec, build, task)
                log_run(f"critic_executor_{iteration}", critique_b.model_dump())
                _show_critique("qwen3", critique_b)
            except Exception as e:
                errors.append(f"Critic-B failed {label}: {e}")
                break

            avg_score = (
                critique_a.production_readiness_score +
                critique_b.production_readiness_score
            ) / 2

            if _both_approve(critique_a, critique_b):
                approved = True
                print(f"\n✅ Both models approved after {iteration} iteration(s)!")
                # ── Reinforce success ─────────────────────────────────────
                self.learner.process_success(task, avg_score, spec)
                break
            else:
                # ── Extract learnings from failure ────────────────────────
                self.learner.process_critique(critique_a, critique_b, spec, task, avg_score)
                scores = (f"gemma4={critique_a.production_readiness_score}  "
                          f"qwen3={critique_b.production_readiness_score}")
                print(f"   ↻ {scores}/100  [memory updated]")
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

        # ── 4. Summary + memory stats ─────────────────────────────────────────
        if build and critique_a and critique_b:
            _print_summary(spec, build, critique_a, critique_b)

        s = self.memory.stats()
        print(f"\n🧠 Memory: {s['mistakes']} mistakes · "
              f"{s['best_practices']} practices · "
              f"{s['prompt_rules']} rules · "
              f"{s['task_history']} runs")

        return FinalResult(
            task_type="GENERATION",
            success=approved,
            specification=spec,
            build_output=build,
            planner_critique=critique_a,
            executor_critique=critique_b,
            iterations=iteration,
            total_ms=(time.time() - t0) * 1000,
            output_file=output_file,
            warnings=warnings,
            errors=errors,
        )

    def _error(self, task, errors, elapsed) -> FinalResult:
        return FinalResult(
            task_type="ERROR", success=False,
            iterations=0, total_ms=elapsed * 1000, errors=errors,
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _show_critique(model: str, c: CritiqueReport):
    status = "✅ APPROVED" if not c.fix_required else "⚠️  NEEDS FIXES"
    print(f"   [{model}] {status} | score={c.production_readiness_score}/100 "
          f"| CRIT={c.severity_summary.get('CRITICAL', 0)} "
          f"HIGH={c.severity_summary.get('HIGH', 0)}")


def _print_summary(spec, build, ca, cb):
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    for r in spec.requirements:
        ea = next((e for e in ca.requirement_evaluations if e.requirement_id == r.id), None)
        eb = next((e for e in cb.requirement_evaluations if e.requirement_id == r.id), None)
        sa = ea.status if ea else "?"
        sb = eb.status if eb else "?"
        icon = "✅" if sa == "SATISFIED" and sb == "SATISFIED" else "⚠️ "
        print(f"  {icon} {r.id}: {r.description[:55]}")
        print(f"       gemma4={sa}  qwen3={sb}")

    print(f"\nScores: gemma4={ca.production_readiness_score}/100  "
          f"qwen3={cb.production_readiness_score}/100")

    all_issues = {i.id: i for i in ca.issues + cb.issues}
    for sev, icon in [("CRITICAL", "🔴"), ("HIGH", "🟠")]:
        found = [i for i in all_issues.values() if i.severity == sev]
        if found:
            print(f"\n{icon} {sev} ({len(found)}):")
            for i in found:
                print(f"  {i.id}: {i.description[:70]}")

    print("\n" + "=" * 60)
    print("💻 GENERATED CODE")
    print("=" * 60)
    print(build.code)
    print("=" * 60)


def run(task: str) -> FinalResult:
    return Orchestrator().run(task)
