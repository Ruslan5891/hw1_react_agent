"""Завдання 4. Прогін тест-кейсів та збір результатів.

Запускає агента на п'яти тест-кейсах різної складності та зберігає:
  * test_results.json — результати (query, expected, actual, steps, tools, time);
  * trajectory.json   — повні траєкторії цих запусків (демо-запуски з safety.py
                        у файлі зберігаються).

Запуск:  python test_runner.py
"""

from collections import Counter

from agent import MODEL_NAME, run_agent
from logger import save_json, save_trajectories
from safety import STOP_LOOP, STOP_MAX_STEPS, STOP_TIMEOUT, SafetyConfig

RESULTS_FILE = "test_results.json"
TRAJECTORY_FILE = "trajectory.json"

# Однакові ліміти для всіх тест-кейсів
SAFETY = SafetyConfig(max_steps=10, timeout_seconds=60, max_repeats=3)

# expected_tools  — інструменти, які агент має викликати;
# expected_facts  — факти у відповіді (для кожного достатньо одного з варіантів запису);
# forbidden_facts — рядки, яких у відповіді бути не повинно.
TEST_CASES = [
    {
        "id": "TC-001",
        "complexity": "simple",
        "query": "Я пробіг 10 км за 50:00. Який у мене був темп?",
        "expected": "Темп 5:00 хв/км (швидкість 12 км/год). Один виклик calculate_pace.",
        "expected_tools": ["calculate_pace"],
        "expected_facts": [["5:00"]],
    },
    {
        "id": "TC-002",
        "complexity": "medium",
        "query": (
            "Я пробіг 5 км за 25:00, а мій друг — 10 км за 55:00. "
            "У кого з нас вищий темп і на скільки секунд на кілометр?"
        ),
        "expected": (
            "Мій темп 5:00 хв/км, у друга 5:30 хв/км; я швидший на 30 с/км. "
            "Два виклики calculate_pace."
        ),
        "expected_tools": ["calculate_pace", "calculate_pace"],
        "expected_facts": [["5:00"], ["5:30"], ["30"]],
    },
    {
        "id": "TC-003",
        "complexity": "complex",
        "query": (
            "Мені 30 років, вага 75 кг, пульс у спокої 60. Нещодавно я пробіг 10 км "
            "за 50:00. Хочу пробігти півмарафон: спрогнозуй мій час, скажи потрібний "
            "темп, підкажи пульсові зони і порахуй, скільки калорій я спалю за цей забіг."
        ),
        "expected": (
            "Прогноз 1:50:19, темп 5:14 хв/км, макс. пульс 190 і п'ять зон "
            "(125–190 уд/хв), близько 1350 ккал. Потрібні predict_race_time, "
            "calculate_heart_rate_zones та estimate_calories; калорії залежать "
            "від прогнозованого часу, тому потрібно щонайменше дві ітерації."
        ),
        "expected_tools": [
            "predict_race_time",
            "calculate_heart_rate_zones",
            "estimate_calories",
        ],
        "expected_facts": [
            ["1:50:19", "1 година 50 хвилин", "1 год 50 хв"],
            ["5:14"],
            ["190"],
            ["135", "134"],
        ],
    },
    {
        "id": "TC-004",
        "complexity": "medium",
        "query": "Мені 95 років, пульс у спокої 130. Порахуй пульсові зони.",
        "expected": (
            "Некоректні дані користувача: інструмент відхиляє пульс 130 "
            "(ValidationError: від 30 до 120). Агент пояснює помилку і не вигадує зони."
        ),
        "expected_tools": ["calculate_heart_rate_zones"],
        "expected_facts": [["120"]],
    },
    {
        "id": "TC-005",
        "complexity": "edge",
        "query": "Я пробіг півмарафон за 1:75:00. Який у мене був темп?",
        "expected": (
            "Час 1:75:00 некоректний (75 хвилин у форматі H:MM:SS). Агент має передати "
            "його в calculate_pace дослівно, отримати ValidationError і повідомити про "
            "помилку — а не підставляти інший час."
        ),
        "expected_tools": ["calculate_pace"],
        "expected_facts": [],
        "forbidden_facts": ["3:33"],
    },
]


def check_tools(expected: list[str], actual: list[str]) -> bool:
    """Чи викликав агент очікувані інструменти потрібну кількість разів."""
    if not expected:
        return not actual
    actual_counts = Counter(actual)
    return all(actual_counts[name] >= count for name, count in Counter(expected).items())


def check_facts(expected_facts: list[list[str]], answer: str) -> list[str]:
    """Повернути перелік фактів, яких бракує у відповіді."""
    return [
        " | ".join(variants)
        for variants in expected_facts
        if not any(variant in answer for variant in variants)
    ]


def run_test_case(tc: dict) -> tuple[dict, dict | None]:
    """Виконати один тест-кейс. Повертає (запис результату, траєкторія)."""
    try:
        result = run_agent(tc["query"], safety=SAFETY, run_id=tc["id"])
    except Exception as e:
        return (
            {
                "test_id": tc["id"],
                "complexity": tc["complexity"],
                "query": tc["query"],
                "expected": tc["expected"],
                "actual": f"Error: {type(e).__name__}: {e}"[:500],
                "status": "error",
                "steps": -1,
                "tool_calls": [],
                "elapsed_ms": -1,
            },
            None,
        )

    final = result["final_answer"]
    tool_names = [call["name"] for call in result["tool_calls"]]
    tools_ok = check_tools(tc["expected_tools"], tool_names)
    missing_facts = check_facts(tc["expected_facts"], final["answer"])
    forbidden_found = [
        fact for fact in tc.get("forbidden_facts", []) if fact in final["answer"]
    ]

    # success — агент завершив роботу сам і пройшов усі перевірки, інакше partial
    if result["stop_reason"] is None and tools_ok and not missing_facts and not forbidden_found:
        status = "success"
    else:
        status = "partial"

    record = {
        "test_id": tc["id"],
        "complexity": tc["complexity"],
        "query": tc["query"],
        "expected": tc["expected"],
        "actual": final["answer"][:500],
        "confidence": final["confidence"],
        "sources": final["sources"],
        "status": status,
        "steps": result["steps"],
        "tool_calls": tool_names,
        "tool_calls_detailed": result["tool_calls"],
        "elapsed_ms": result["elapsed_ms"],
        "stop_reason": result["stop_reason"],
        "max_steps_reached": result["stop_reason"] == STOP_MAX_STEPS,
        "timeout_reached": result["stop_reason"] == STOP_TIMEOUT,
        "loop_detected": result["stop_reason"] == STOP_LOOP,
        "checks": {
            "expected_tools": tc["expected_tools"],
            "tools_ok": tools_ok,
            "missing_facts": missing_facts,
            "forbidden_facts_found": forbidden_found,
        },
    }
    return record, result["trajectory"]


def main() -> None:
    print(f"Модель: {MODEL_NAME}")
    print(f"Ліміти: max_steps={SAFETY.max_steps}, timeout={SAFETY.timeout_seconds:g} с, "
          f"max_repeats={SAFETY.max_repeats}\n")

    results, trajectories = [], []
    for tc in TEST_CASES:
        print(f"Running {tc['id']} [{tc['complexity']}]: {tc['query'][:60]}...")
        record, trajectory = run_test_case(tc)
        results.append(record)
        if trajectory:
            trajectories.append(trajectory)

    save_json(RESULTS_FILE, results)
    save_trajectories(TRAJECTORY_FILE, trajectories)

    print(f"\nResults saved: {RESULTS_FILE} ({len(results)} test cases)")
    for r in results:
        print(
            f"  {r['test_id']} [{r['complexity']:<7}] {r['status']:<7} "
            f"{r['steps']} steps, {r['elapsed_ms']} ms, tools: {r['tool_calls']}"
        )
    print(
        "\nСтатус partial означає, що агент відпрацював без збою, але не пройшов "
        "автоматичну перевірку (деталі — у полі checks)."
        )


if __name__ == "__main__":
    main()
