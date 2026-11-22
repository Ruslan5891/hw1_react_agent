"""Завдання 3. Логування траєкторії виконання агента у JSON.

TrajectoryLogger створюється окремо для кожного запуску. Вузли графа
викликають log_step(), після завершення траєкторія зберігається у файл.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

# Довгі поля обрізаються, щоб файл траєкторії не розростався
MAX_FIELD_LENGTH = 500


def _truncate(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) > MAX_FIELD_LENGTH:
        return text[:MAX_FIELD_LENGTH] + "…"
    return text


class TrajectoryLogger:
    """Журнал кроків одного запуску агента."""

    def __init__(self, run_id: str = "run", query: str = ""):
        self.run_id = run_id
        self.query = query
        self.steps: list[dict] = []
        self.start_time = time.monotonic()
        self.started_at = datetime.now(tz=timezone.utc).isoformat()

    def log_step(
        self,
        node: str,
        input_data: Any,
        output_data: Any,
        tool_calls: Optional[list] = None,
        duration_ms: int = 0,
        **extra: Any,
    ) -> None:
        """Записати один крок.

        node        — назва вузла: agent, tools, guard, formatter;
        tool_calls  — список викликів інструментів у форматі {name, args};
        duration_ms — тривалість саме цього кроку;
        extra       — додаткові поля (iteration, status, tokens тощо).
        """
        entry = {
            "step_number": len(self.steps) + 1,
            "node_name": node,
            "input": _truncate(input_data),
            "output": _truncate(output_data),
            "tool_calls": tool_calls or [],
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "duration_ms": int(duration_ms),
            "elapsed_ms": int((time.monotonic() - self.start_time) * 1000),
        }
        entry.update(extra)
        self.steps.append(entry)

    def to_dict(self) -> dict:
        """Траєкторія як словник, готовий до запису в JSON."""
        return {
            "run_id": self.run_id,
            "query": self.query,
            "started_at": self.started_at,
            "total_steps": len(self.steps),
            "total_time_ms": int((time.monotonic() - self.start_time) * 1000),
            "trajectory": self.steps,
        }

    def save(self, filepath: str) -> None:
        """Зберегти траєкторію одного запуску."""
        save_json(filepath, self.to_dict())
        print(f"Trajectory saved: {filepath} ({len(self.steps)} steps)")


def save_json(filepath: str, data: Any) -> None:
    """Записати JSON з кирилицею та відступами."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_trajectories(filepath: str, runs: list[dict]) -> None:
    """Зберегти траєкторії кількох запусків в один файл.

    Запуски з тим самим run_id замінюються, решта у файлі зберігається:
    тому test_runner.py і safety.py пишуть в один trajectory.json.
    """
    merged: dict[str, dict] = {}
    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as f:
            for run in json.load(f).get("runs", []):
                merged[run["run_id"]] = run
    for run in runs:
        merged[run["run_id"]] = run

    save_json(filepath, {"total_runs": len(merged), "runs": list(merged.values())})
    print(f"Trajectories saved: {filepath} ({len(runs)} runs, {len(merged)} у файлі)")
