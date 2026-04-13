"""
Maestro CLI
Usage:
  python -m maestro run "your task"
  python -m maestro memory
  python -m maestro memory clear
"""
import sys
from maestro.orchestrator import run


def _cmd_memory(args):
    from maestro.memory import store as m
    import json

    if args and args[0] == "clear":
        from pathlib import Path
        p = Path("memory/knowledge.json")
        if p.exists():
            p.unlink()
        print("🗑  Memory cleared.")
        return

    data = m._load()
    print("\n🧠 MAESTRO MEMORY\n" + "=" * 50)

    mistakes = sorted(data.get("mistakes", []), key=lambda x: x.get("frequency", 1), reverse=True)
    print(f"\n📌 Top Mistakes ({len(mistakes)}):")
    for i, mk in enumerate(mistakes[:5], 1):
        print(f"  {i}. [{mk.get('frequency',1)}x] {mk['pattern'][:70]}")
        print(f"     Fix: {mk['fix'][:80]}")

    practices = sorted(data.get("best_practices", []), key=lambda x: x.get("frequency", 1), reverse=True)
    print(f"\n✅ Best Practices ({len(practices)}):")
    for i, bp in enumerate(practices[:5], 1):
        print(f"  {i}. [{bp.get('frequency',1)}x] {bp['pattern'][:70]}")

    rules = data.get("prompt_rules", [])
    print(f"\n⚡ Active Prompt Rules ({len(rules)}):")
    for i, r in enumerate(rules, 1):
        print(f"  {i}. {r}")

    runs = data.get("run_log", [])
    print(f"\n📊 Run History ({len(runs)} total):")
    for r in runs[-5:]:
        status = "✅" if r.get("success") else "❌"
        print(f"  {status} score={r.get('score',0):.0f} iter={r.get('iterations',0)} | {r.get('task','')[:60]}")

    print()


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m maestro run "task"')
        print('       python -m maestro memory')
        print('       python -m maestro memory clear')
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "memory":
        _cmd_memory(sys.argv[2:])
        return

    if cmd != "run" or len(sys.argv) < 3:
        print('Usage: python -m maestro run "task description"')
        sys.exit(1)

    task = sys.argv[2]
    print("=" * 60)
    print("🎼 MAESTRO — Self-Improving Dual-Model Orchestration")
    print(f"   Planner : gemma4:e4b")
    print(f"   Executor: qwen3:8b")
    print("=" * 60)
    print(f"\n📝 Task: {task}\n")

    try:
        result = run(task)

        if result.warnings:
            print("\n⚠️  Warnings:")
            for w in result.warnings:
                print(f"  - {w}")
        if result.errors:
            print("\n❌ Errors:")
            for e in result.errors:
                print(f"  - {e}")

        print(f"\n⏱  Total time : {result.total_ms/1000:.1f}s")
        print(f"🔄 Iterations : {result.iterations}")
        print(f"{'✅ SUCCESS' if result.success else '❌ FAILED'}")
        sys.exit(0 if result.success else 1)

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
