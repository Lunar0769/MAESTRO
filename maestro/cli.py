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
    from maestro.memory.manager import MemoryManager, MEMORY_FILE
    import os

    if args and args[0] == "clear":
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
        print("🗑  Memory cleared.")
        return

    mm   = MemoryManager()
    data = mm.memory
    print("\n🧠 MAESTRO MEMORY\n" + "=" * 50)

    mistakes = sorted(data.get("mistakes", []), key=lambda x: x.get("count", 1), reverse=True)
    print(f"\n📌 Mistakes ({len(mistakes)}):")
    for i, m in enumerate(mistakes[:8], 1):
        print(f"  {i}. [{m.get('count',1)}x] {m['pattern'][:70]}")
        print(f"     Fix: {m['fix'][:80]}")

    practices = sorted(data.get("best_practices", []), key=lambda x: x.get("count", 1), reverse=True)
    print(f"\n✅ Best Practices ({len(practices)}):")
    for i, bp in enumerate(practices[:5], 1):
        print(f"  {i}. [{bp.get('count',1)}x] {bp['pattern'][:70]}")

    rules = data.get("prompt_rules", [])
    print(f"\n⚡ Active Rules ({len(rules)}):")
    for i, r in enumerate(rules, 1):
        print(f"  {i}. {r}")

    history = data.get("task_history", [])
    print(f"\n📊 Task History ({len(history)} total):")
    for r in history[-5:]:
        print(f"  score={r.get('score', 0):.0f} | {r.get('task', '')[:60]}")
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
