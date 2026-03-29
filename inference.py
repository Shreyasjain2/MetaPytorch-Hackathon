"""
Baseline Inference Script for SQL/Data Cleaning Sandbox  OpenAI Edition.

Uses OpenAI (gpt-4o) to solve all three tasks and prints reproducible
scores via the OpenEnv WebSocket client.

Usage:
    set HF_TOKEN=sk-...          # Windows
    export HF_TOKEN=sk-...       # Linux/macOS
    python inference.py                    # local server
    python inference.py --url https://...  # remote server
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from client import SqlSandboxEnv
from models import SqlSandboxAction


# ---------------------------------------------------------------------------
# System prompt shared across all tasks
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a data engineering assistant working inside a SQLite sandbox.

You can execute two types of actions:
1. {"tool": "sql",    "command": "<SQL query>"}
2. {"tool": "python", "command": "<Python code>"}

Rules:
- Respond with EXACTLY ONE JSON object per turn  no markdown, no explanation.
- In Python code, the variables `conn` (sqlite3.Connection) and `cursor`
  (sqlite3.Cursor) are already available. Do NOT call sqlite3.connect().
- SQLite STRFTIME months are zero-padded: use '01' not '1', or use LIKE '2024-01-%'.
- When you believe the task is fully complete, send:
  {"tool": "sql", "command": "SELECT 'DONE'"}
"""


# ---------------------------------------------------------------------------
# Core agent loop  one task, one WebSocket session
# ---------------------------------------------------------------------------
def _run_task_agent(base_url: str, task_id: str, max_turns: int = 15) -> float:
    """
    Open a fresh WebSocket session, reset the environment to the given task,
    then run an LLM agent loop until done or max_turns is reached.
    Returns the final reward (0.0  1.0).
    """
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
    api_base_url = os.environ.get("API_BASE_URL")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o")

    client_llm = OpenAI(
        api_key=api_key,
        base_url=api_base_url,
    )
    final_reward = 0.0

    # Each task gets its own WebSocket session to avoid state leakage
    with SqlSandboxEnv(base_url=base_url).sync() as env:
        # reset() with task_id seeds the correct DB table for this task
        reset_resp = env.reset(task_id=task_id)
        task_desc = reset_resp.observation.task_description

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Task: {task_desc}\n\nBegin."},
        ]

        print(f"\n  --- Session: {task_id} ---")

        for turn in range(max_turns):
            # 1. Ask the LLM
            response = client_llm.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            assistant_msg = response.choices[0].message.content.strip()

            # 2. Parse action JSON (handle optional markdown fences)
            try:
                raw = assistant_msg
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                action_data = json.loads(raw)
                tool    = action_data["tool"]
                command = action_data["command"]
            except (json.JSONDecodeError, KeyError):
                # Feed parse error back to LLM, do NOT count as a step
                messages.append({"role": "assistant", "content": assistant_msg})
                messages.append({
                    "role": "user",
                    "content": (
                        'Invalid JSON. Reply with exactly one JSON object:\n'
                        '{"tool": "sql" | "python", "command": "..."}'
                    ),
                })
                continue

            # 3. Execute the action via OpenEnv step()
            step_resp = env.step(SqlSandboxAction(tool=tool, command=command))

            reward = step_resp.reward or 0.0
            done   = step_resp.done
            output = step_resp.observation.output or ""
            error  = step_resp.observation.error  or ""

            final_reward = reward
            print(f"  [Turn {turn+1:02d}] tool={tool:<6} | reward={reward:.4f} | done={done}")

            if done:
                break

            # 4. Feed result back to LLM for the next turn
            messages.append({"role": "assistant", "content": assistant_msg})
            feedback = f"Output:\n{output[:1500]}"
            if error:
                feedback += f"\nError:\n{error[:500]}"
            feedback += f"\nReward so far: {reward:.4f}"
            messages.append({"role": "user", "content": feedback})

    return final_reward


# ---------------------------------------------------------------------------
# Per-difficulty entry points (called by main, importable for custom use)
# ---------------------------------------------------------------------------
def easy_run(base_url: str, max_turns: int = 15) -> float:
    print(f"\n{'='*50}\nRunning task: easy\n{'='*50}")
    score = _run_task_agent(base_url, "easy", max_turns)
    print(f"  Final score: {score:.4f}")
    return score


def med_run(base_url: str, max_turns: int = 15) -> float:
    print(f"\n{'='*50}\nRunning task: medium\n{'='*50}")
    score = _run_task_agent(base_url, "medium", max_turns)
    print(f"  Final score: {score:.4f}")
    return score


def hard_run(base_url: str, max_turns: int = 15) -> float:
    print(f"\n{'='*50}\nRunning task: hard\n{'='*50}")
    score = _run_task_agent(base_url, "hard", max_turns)
    print(f"  Final score: {score:.4f}")
    return score


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="OpenAI baseline inference for the SQL/Data Cleaning Sandbox"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the running environment server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=15,
        help="Maximum agent turns per task (default: 15)",
    )
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN") and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: HF_TOKEN (or OPENAI_API_KEY) environment variable is not set per checklist.")
        sys.exit(1)

    results: dict[str, float] = {}
    results["easy"]   = easy_run(args.url, args.max_turns)
    results["medium"] = med_run(args.url,  args.max_turns)
    results["hard"]   = hard_run(args.url, args.max_turns)

    avg = sum(results.values()) / len(results)
    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print(f"{'='*50}")
    for task_id, score in results.items():
        print(f"  {task_id:<10}: {score:.4f}")
    print(f"  {'average':<10}: {avg:.4f}")


if __name__ == "__main__":
    main()
