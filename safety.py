"""Завдання 3. Захисні механізми агента: max_steps, timeout, детекція зациклення.

SafetyGuard об'єднує три перевірки в один об'єкт, який створюється окремо
для кожного запуску агента і викликається з вузла agent на кожній ітерації.

Запуск демонстрації трьох механізмів:  python safety.py
(траєкторії демо-запусків додаються у trajectory.json)
"""

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from langchain_core.messages import AIMessage

# Причини зупинки агента
STOP_MAX_STEPS = "max_steps"
STOP_TIMEOUT = "timeout"
STOP_LOOP = "loop_detected"
STOP_LLM_ERROR = "llm_error"


@dataclass(frozen=True)
class SafetyConfig:
    """Налаштування захисних механізмів."""

    max_steps: int = 10
    timeout_seconds: float = 60.0
    max_repeats: int = 3


@dataclass(frozen=True)
class StopDecision:
    """Рішення про зупинку: машинна причина + повідомлення для користувача."""

    reason: str
    message: str


class LoopDetector:
    """Детекція зациклення — однакових tool calls, що йдуть поспіль."""

    def __init__(self, max_repeats: int = 3):
        self.max_repeats = max_repeats
        self.recent_calls: list[str] = []

    @staticmethod
    def _call_hash(tool_name: str, args: dict) -> str:
        # sort_keys: порядок аргументів не впливає на хеш
        payload = json.dumps(
            {"tool": tool_name, "args": args},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def check(self, tool_name: str, args: dict) -> bool:
        """Зареєструвати виклик. Повертає True, якщо виявлено зациклення."""
        self.recent_calls.append(self._call_hash(tool_name, args))
        # Зациклення: останні max_repeats викликів однакові
        last_n = self.recent_calls[-self.max_repeats:]
        return len(last_n) == self.max_repeats and len(set(last_n)) == 1


class SafetyGuard:
    """Контролер зупинки для одного запуску агента.

    Параметр clock дозволяє підмінити годинник (за замовчуванням time.monotonic).
    """

    def __init__(
        self,
        config: Optional[SafetyConfig] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config or SafetyConfig()
        self._clock = clock
        self.start_time = clock()
        self.loop_detector = LoopDetector(self.config.max_repeats)

    def elapsed_seconds(self) -> float:
        return self._clock() - self.start_time

    def _timeout_decision(self) -> Optional[StopDecision]:
        elapsed = self.elapsed_seconds()
        if elapsed > self.config.timeout_seconds:
            return StopDecision(
                STOP_TIMEOUT,
                f"Перевищено тайм-аут: минуло {elapsed:.1f} с "
                f"за ліміту {self.config.timeout_seconds:g} с.",
            )
        return None

    def check_before_llm(self, step_count: int) -> Optional[StopDecision]:
        """Перевірка перед черговим викликом LLM: ліміт кроків і тайм-аут."""
        if step_count >= self.config.max_steps:
            return StopDecision(
                STOP_MAX_STEPS,
                f"Досягнуто ліміт кроків ({self.config.max_steps}).",
            )
        return self._timeout_decision()

    def check_tool_calls(self, tool_calls: list[dict]) -> Optional[StopDecision]:
        """Перевірка після відповіді LLM, до виконання інструментів."""
        timeout = self._timeout_decision()
        if timeout:
            return timeout

        # Кожен tool call з відповіді LLM реєструється в детекторі зациклення
        for call in tool_calls:
            if self.loop_detector.check(call["name"], call.get("args", {})):
                return StopDecision(
                    STOP_LOOP,
                    f"Виявлено зациклення: інструмент '{call['name']}' викликано "
                    f"{self.config.max_repeats} рази поспіль з однаковими аргументами.",
                )
        return None


# ── Демонстрація захисних механізмів ──────────────────────────────────────────

TRAJECTORY_FILE = "trajectory.json"
SLOW_TOOL_DELAY_SECONDS = 3


class _ScriptedStructuredOutput:
    """Заміна llm.with_structured_output() для сценарної моделі."""

    def __init__(self, schema: type):
        self._schema = schema

    def invoke(self, messages: list, **kwargs: Any):
        return self._schema(answer="Сценарна відповідь.", confidence=1.0, sources=[])


class ScriptedChatModel:
    """Сценарна «модель»: щоразу повертає той самий tool call без звернення до API.

    Справжня LLM не зациклюється на замовлення, тому зациклення відтворюємо
    моделлю з тим самим інтерфейсом, який використовує граф.
    """

    def __init__(self, tool_name: str, args: dict):
        self._tool_name = tool_name
        self._args = args
        self.calls = 0

    def bind_tools(self, tools: list, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def with_structured_output(self, schema: type, **kwargs: Any) -> _ScriptedStructuredOutput:
        return _ScriptedStructuredOutput(schema)

    def invoke(self, messages: list, **kwargs: Any) -> AIMessage:
        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": self._tool_name,
                    "args": self._args,
                    "id": f"scripted_{self.calls}",
                    "type": "tool_call",
                }
            ],
        )


def demo_max_steps() -> dict:
    """Складний запит потребує кількох ітерацій, а дозволена лише одна."""
    from agent import run_agent

    return run_agent(
        "Я пробіг 10 км за 50:00. Спрогнозуй мій час на півмарафоні і порахуй, "
        "скільки калорій я спалю за цей півмарафон при вазі 75 кг.",
        safety=SafetyConfig(max_steps=1),
        run_id="DEMO-max-steps",
    )


def demo_timeout() -> dict:
    """Інструмент працює 3 с при загальному ліміті 2 с."""
    from time import sleep

    from langchain_core.tools import tool

    from agent import run_agent
    from tools import TOOLS, PaceInput, calculate_pace

    # Повільна копія calculate_pace: та сама назва і схема, але із затримкою
    @tool("calculate_pace", args_schema=PaceInput, description=calculate_pace.description)
    def slow_calculate_pace(distance_km: float, time: str) -> str:
        sleep(SLOW_TOOL_DELAY_SECONDS)
        return calculate_pace.invoke({"distance_km": distance_km, "time": time})

    tools = [slow_calculate_pace] + [t for t in TOOLS if t.name != "calculate_pace"]
    return run_agent(
        "Я пробіг 10 км за 50:00. Який у мене був темп?",
        tools=tools,
        safety=SafetyConfig(timeout_seconds=2),
        run_id="DEMO-timeout",
    )


def demo_loop() -> dict:
    """Модель щоразу просить той самий виклик з тими самими аргументами."""
    from agent import run_agent

    looping_model = ScriptedChatModel("calculate_pace", {"distance_km": 10, "time": "50:00"})
    return run_agent(
        "Я пробіг 10 км за 50:00. Який у мене був темп?",
        llm=looping_model,
        safety=SafetyConfig(max_repeats=3),
        run_id="DEMO-loop",
    )


DEMOS = [
    (STOP_MAX_STEPS, "max_steps=1, справжня LLM", demo_max_steps),
    (STOP_TIMEOUT, "timeout=2 с, інструмент відповідає 3 с", demo_timeout),
    (STOP_LOOP, "max_repeats=3, сценарна модель повторює виклик", demo_loop),
]


def main() -> None:
    from agent import print_result
    from logger import save_trajectories

    trajectories = []
    for expected_reason, description, demo in DEMOS:
        print(f"\n{'=' * 70}\nДемо: {description}\n{'=' * 70}")
        result = demo()
        print_result(result)
        triggered = result["stop_reason"] == expected_reason
        print(f"Механізм спрацював: {'так' if triggered else 'НІ'}")
        trajectories.append(result["trajectory"])
    print()
    save_trajectories(TRAJECTORY_FILE, trajectories)


if __name__ == "__main__":
    main()
