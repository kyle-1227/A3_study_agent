"""
Demo and integration example for the AI-native User Profile System.

Run::

    cd d:/ruanjian/A3_study_agent
    python scripts/demo_profile.py

This script:
1. Creates a ProfileManager
2. Simulates several conversation turns
3. Shows how the profile evolves over time
4. Demonstrates prompt injection
5. Shows how to integrate with the existing app.py / LangGraph
"""

import asyncio
import os
import sys

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

from src.profile import (
    ProfileManager,
    SQLiteProfileStore,
    UserProfile,
    get_profile_manager,
    profile_to_summary,
    score_snapshot,
    top_skills,
    weakest_skills,
)


# ── Demo: simulate a user learning Python ──────────────────────────────────


async def demo_basic_flow():
    """Demonstrate the core profile lifecycle."""

    print("=" * 60)
    print("AI Study Agent — User Profile System Demo")
    print("=" * 60)

    # 1. Create the manager
    store = SQLiteProfileStore("data/demo_profile.db")
    manager = ProfileManager(store=store)
    print("\n✓ ProfileManager initialized (SQLite at data/demo_profile.db)")

    user_id = "demo_user_001"

    # 2. Simulate conversation turns
    conversations = [
        {
            "user": "你好，我刚开始学Python，能教我写一个Hello World吗？",
            "assistant": "当然！Python 的 Hello World 非常简单：\n\n```python\nprint('Hello, World!')\n```\n\n你只需要一行代码。Python 的 print() 函数会在屏幕上输出括号里的内容。",
        },
        {
            "user": "list和tuple有什么区别？能不能给我举个具体的例子？",
            "assistant": "好的！主要区别是 list 可变，tuple 不可变：\n\n```python\n# list 可以修改\nfruits = ['apple', 'banana']\nfruits.append('orange')  # ✓ 可以\n\n# tuple 不能修改\ncolors = ('red', 'green')\n# colors.append('blue')  # ✗ 报错！\n```\n\n用 list 存需要修改的数据，用 tuple 存储不变的数据如坐标、配置等。",
        },
        {
            "user": "我理解了。那闭包是什么？能不能一步一步解释？",
            "assistant": "没问题，我们一步一步来。\n\n**第1步：理解嵌套函数**\n```python\ndef outer():\n    def inner():\n        print('hello')\n    return inner\n```\n\n**第2步：内部函数记住外部变量**\n```python\ndef outer(x):\n    def inner():\n        print(x)  # inner 记住了 x\n    return inner\n\nf = outer(10)\nf()  # 输出 10\n```\n\n**第3步：这就是闭包** — 内部函数记住了创建时的环境。常用于工厂函数和装饰器。",
        },
        {
            "user": "我对递归还不太熟，能再讲一遍吗？上次讲到一半没听懂。不要只给理论，给具体代码例子。",
            "assistant": "完全理解！我们直接用代码说话：\n\n```python\ndef factorial(n):\n    # 第1步：终止条件\n    if n <= 1:\n        return 1\n    # 第2步：递归调用\n    return n * factorial(n - 1)\n\nprint(factorial(5))  # 120\n```\n\n执行过程：factorial(5) → 5 * factorial(4) → ... → 5 * 4 * 3 * 2 * 1 = 120\n\n关键点：递归 = 终止条件 + 自我调用。",
        },
    ]

    for i, turn in enumerate(conversations, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {turn['user'][:60]}...")

        result = await manager.process_conversation(
            user_id=user_id,
            user_message=turn["user"],
            assistant_response=turn["assistant"],
        )

        if result.changes:
            print(f"  Profile changes ({len(result.changes)}):")
            for change in result.changes:
                print(f"    • {change}")
        else:
            print("  (no new profile signals)")

    # 3. Show the final profile
    print("\n" + "=" * 60)
    print("Final Profile")
    print("=" * 60)

    profile = await manager.get_profile(user_id)
    if profile:
        print(f"\nUser ID: {profile.user_id}")
        print(f"Created: {profile.created_at}")
        print(f"Updated: {profile.updated_at}")
        print(f"\nSkills ({len(profile.skills)}):")
        for name, entry in profile.skills.items():
            print(f"  {name}: level={entry.level:.2f}, confidence={entry.confidence:.2f}, "
                  f"evidence_count={entry.evidence_count}")

        print(f"\nLearning Style:")
        s = profile.learning_style
        for dim in ["prefer_examples", "prefer_step_by_step", "prefer_concise",
                     "prefer_theory", "prefer_practice", "prefer_visual", "prefer_analogy"]:
            val = getattr(s, dim)
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            print(f"  {dim:25s}: {bar} {val:.2f}")

        print(f"\nGoals ({len(profile.goals)}):")
        for g in profile.goals:
            print(f"  • {g.goal} (importance: {g.importance:.2f})")

        print(f"\nObservations ({len(profile.agent_observations)}):")
        for obs in profile.agent_observations[-5:]:
            print(f"  • [{obs.category}] {obs.content}")

        # 4. Show prompt injection
        print("\n" + "=" * 60)
        print("Prompt Injection (what the AI sees)")
        print("=" * 60)
        ctx = profile_to_summary(profile)
        print(ctx)

        # 5. Score snapshot
        print("\n" + "=" * 60)
        print("Score Snapshot")
        print("=" * 60)
        scores = score_snapshot(profile)
        for name, data in scores.items():
            print(f"  {name}: {data['level']:.2f} (conf: {data['confidence']:.2f})")

        top = top_skills(profile, min_confidence=0.1)
        weak = weakest_skills(profile, min_confidence=0.1)
        print(f"\n  Top skills: {top}")
        print(f"  Weakest skills: {weak}")

    print("\n✓ Demo complete!")


# ── Integration with app.py / LangGraph ────────────────────────────────────


def integration_guide():
    """Print integration instructions for the existing app."""

    print("""
╔══════════════════════════════════════════════════════════════╗
║  Integration Guide: User Profile in app.py / LangGraph      ║
╚══════════════════════════════════════════════════════════════╝

1. In app.py, add profile injection to /stream:

```python
from src.profile import get_profile_manager

@app.post("/stream")
async def stream_chat(request: ChatRequest):
    manager = get_profile_manager()

    # Inject profile into the query
    profile_ctx = await manager.build_profile_context(user_id)
    enriched_query = (profile_ctx + "\\n\\n" + request.query) if profile_ctx else request.query

    # Pass enriched query to the graph...
```

2. After the assistant responds, process the turn:

```python
    # After streaming is complete
    await manager.process_conversation(
        user_id=user_id,
        user_message=request.query,
        assistant_response=final_response_text,
    )
```

3. In LearningState, add a profile_context field:

```python
# In src/graph/state.py
class LearningState(TypedDict):
    ...
    profile_context: str  # Injected profile summary
    user_id: str          # For profile lookups
```

4. In the supervisor or generate_answer node:

```python
# Use profile_context to personalize the response
if state.get("profile_context"):
    system_prompt = base_prompt + "\\n\\n" + state["profile_context"]
```

5. For the volunteer (志愿填报) page:
   - The frontend already sends user queries with regional context
   - Add user_id to the query so the backend can attach profiles
   - The backend merges profile context + regional context

6. Future extensions:
   - Add profile-aware difficulty adjustment in academic.py
   - Use profile.goals to prioritize relevant search results
   - Use profile.learning_style to choose explanation format
   - Add quiz accuracy tracking via behavior.quiz_accuracy
""")


# ── Main ───────────────────────────────────────────────────────────────────


async def main():
    """Run the demo."""
    # Check for API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("⚠  DEEPSEEK_API_KEY not set in .env — extraction will fail")
        print("   Set it to run the full demo with LLM-based extraction.\n")

    await demo_basic_flow()
    integration_guide()


if __name__ == "__main__":
    asyncio.run(main())
