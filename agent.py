"""Завдання 2. ReAct-агент у LangGraph: цикл LLM – tools – LLM.

Граф:

    START → agent ──(є tool_calls)──────→ tools ──→ agent   (цикл)
              │
              ├──(фінальна відповідь)───→ formatter ──────→ END
              └──(спрацював захист)─────→ guard_finalize ─→ END

  * agent          — LLM з прив'язаними інструментами вирішує: викликати tool чи відповісти;
  * tools          — ToolNode виконує tool calls і повертає результати як ToolMessage;
  * formatter      — перетворює відповідь на структуровану модель FinalAnswer;
  * guard_finalize — формує часткову відповідь, якщо спрацював max_steps / timeout / loop.

Запуск демонстрації:  python agent.py
"""

import os
import time
from typing import Annotated, Any, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, ValidationError

from logger import TrajectoryLogger
from safety import STOP_LLM_ERROR, SafetyConfig, SafetyGuard
from tools import TOOLS

load_dotenv()


# ── Налаштування LLM ──────────────────────────────────────────────────────────

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
LLM_REQUEST_TIMEOUT = 30.0

SYSTEM_PROMPT = """Ти — спортивний асистент для бігунів і людей, які тренуються.

Правила:
1. Для будь-яких розрахунків (темп, швидкість, прогноз часу, пульсові зони, калорії) \
ЗАВЖДИ викликай інструменти. Не рахуй самостійно і не вигадуй числа.
2. Якщо для наступного розрахунку потрібен результат попереднього — спершу отримай його, \
а потім роби наступний виклик.
3. Значення користувача (час, дистанцію, вік, вагу, пульс) передавай в інструменти ДОСЛІВНО, \
навіть якщо вони здаються помилковими чи нереалістичними: перевірку виконує інструмент. \
Ніколи не «виправляй» дані користувача самостійно і не відмовляй без виклику інструмента.
4. Якщо інструмент повернув помилку: виправ аргументи, якщо ти сам помилився у форматі. Якщо \
некоректні самі дані користувача — поясни, що саме не так, і не підставляй інші значення.
5. Якщо питання не потребує розрахунків (теорія, поради) — відповідай без інструментів.
6. Відповідай українською, стисло, з конкретними числами з результатів інструментів."""

FORMATTER_PROMPT = """Оформи фінальну відповідь спортивного асистента у структурованому вигляді.

Поля:
- answer: текст відповіді користувачу українською. Візьми за основу чернетку відповіді, \
не змінюй числа і не додавай нових фактів.
- confidence: число від 0 до 1. 0.9–1.0 — усі числа взято з успішних результатів інструментів; \
0.6–0.8 — відповідь без інструментів (загальні знання) або були помилки, які вдалося виправити; \
нижче 0.5 — потрібних даних отримати не вдалося.
- sources: назви інструментів, результати яких використано у відповіді. Порожній список, \
якщо інструменти не використовувалися. (Це поле після генерації звіряється з фактичними викликами.)"""


def create_llm() -> ChatOpenAI:
    """Створити клієнт LLM: модель OpenAI через OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Не знайдено OPENROUTER_API_KEY. Створіть файл .env за зразком .env.example"
        )
    return ChatOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        model=MODEL_NAME,
        temperature=0,
        timeout=LLM_REQUEST_TIMEOUT,
        max_retries=2,
    )


# ── Структурована відповідь і стан графа ──────────────────────────────────────


class FinalAnswer(BaseModel):
    """Структурована фінальна відповідь агента."""

    answer: str = Field(description="Текст відповіді користувачу")
    confidence: float = Field(ge=0.0, le=1.0, description="Впевненість у відповіді, 0–1")
    sources: list[str] = Field(
        default_factory=list,
        description="Назви інструментів, результати яких використано",
    )


class AgentState(TypedDict):
    """Стан, що передається між вузлами графа."""

    # add_messages дописує нові повідомлення до історії, а не замінює її
    messages: Annotated[list[AnyMessage], add_messages]
    step_count: int
    stop_reason: Optional[str]
    stop_message: Optional[str]
    final_answer: Optional[dict]


# ── Допоміжні функції ─────────────────────────────────────────────────────────


def _describe_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Залишити від tool call тільки назву та аргументи (для логів і звітів)."""
    return [{"name": call["name"], "args": call.get("args", {})} for call in tool_calls]


def _collect_tool_results(messages: list[AnyMessage]) -> list[dict]:
    """Зібрати з історії всі результати інструментів разом з аргументами виклику."""
    args_by_id = {}
    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls:
                args_by_id[call["id"]] = call.get("args", {})
    return [
        {
            "tool": msg.name,
            "args": args_by_id.get(msg.tool_call_id, {}),
            "status": msg.status,
            "content": str(msg.content),
        }
        for msg in messages
        if isinstance(msg, ToolMessage)
    ]


def _successful_tool_names(results: list[dict]) -> list[str]:
    """Назви інструментів, що відпрацювали без помилок (без повторів, у порядку викликів)."""
    return list(dict.fromkeys(r["tool"] for r in results if r["status"] != "error"))


def format_tool_error(e: Exception) -> str:
    """Перетворити помилку інструмента на зрозуміле для моделі повідомлення.

    Стандартний обробник ToolNode губить текст помилок model_validator
    (у них порожнє поле loc), тому повідомлення збираємо самі.
    """
    source = getattr(e, "source", e)
    if isinstance(source, ValidationError):
        lines = [
            f"- {'.'.join(str(part) for part in err['loc']) or 'аргументи загалом'}: {err['msg']}"
            for err in source.errors()
        ]
        tool_name = getattr(e, "tool_name", "?")
        return (
            f"Помилка валідації аргументів інструмента '{tool_name}':\n"
            + "\n".join(lines)
            + "\nЯкщо помилка у форматі — виправ аргументи і повтори виклик. "
            "Якщо некоректні дані користувача — поясни йому, що саме не так."
        )
    return f"Помилка виконання інструмента: {type(e).__name__}: {e}"


def _user_query(messages: list[AnyMessage]) -> str:
    for msg in messages:
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


# ── Граф ──────────────────────────────────────────────────────────────────────


def build_graph(llm: Any, tools: list, guard: SafetyGuard, logger: TrajectoryLogger):
    """Зібрати та скомпілювати граф ReAct-агента.

    guard і logger створюються окремо для кожного запуску, тому вузли
    оголошено як замикання, що мають до них доступ.
    """
    llm_with_tools = llm.bind_tools(tools)
    structured_llm = llm.with_structured_output(FinalAnswer, method="function_calling")

    def agent_node(state: AgentState) -> dict:
        """LLM приймає рішення: викликати інструмент чи дати відповідь."""
        step = state.get("step_count", 0)

        # Захист до виклику LLM: ліміт кроків і тайм-аут
        decision = guard.check_before_llm(step)
        if decision:
            logger.log_step(
                "guard", f"перевірка перед кроком {step + 1}", decision.message,
                iteration=step, stop_reason=decision.reason,
            )
            return {"stop_reason": decision.reason, "stop_message": decision.message}

        last_message = state["messages"][-1]
        started = time.monotonic()
        try:
            response = llm_with_tools.invoke(
                [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
            )
        except Exception as e:
            message = f"Помилка виклику LLM: {type(e).__name__}: {e}"
            logger.log_step(
                "agent", last_message.content, message,
                duration_ms=(time.monotonic() - started) * 1000,
                iteration=step + 1, stop_reason=STOP_LLM_ERROR,
            )
            return {
                "step_count": step + 1,
                "stop_reason": STOP_LLM_ERROR,
                "stop_message": message,
            }
        duration_ms = (time.monotonic() - started) * 1000

        calls = _describe_tool_calls(response.tool_calls)
        usage = getattr(response, "usage_metadata", None) or {}
        logger.log_step(
            "agent",
            last_message.content,
            response.content or f"tool calls: {[c['name'] for c in calls]}",
            tool_calls=calls,
            duration_ms=duration_ms,
            iteration=step + 1,
            total_tokens=usage.get("total_tokens"),
        )

        # Захист до виконання інструментів: тайм-аут і зациклення
        if response.tool_calls:
            decision = guard.check_tool_calls(response.tool_calls)
            if decision:
                logger.log_step(
                    "guard", f"tool calls: {calls}", decision.message,
                    tool_calls=calls, iteration=step + 1, stop_reason=decision.reason,
                )
                return {
                    "step_count": step + 1,
                    "stop_reason": decision.reason,
                    "stop_message": decision.message,
                }

        return {"messages": [response], "step_count": step + 1}

    def log_tool_call(request, execute):
        """Обгортка навколо кожного виклику інструмента: вимірює час і логує."""
        call = request.tool_call
        started = time.monotonic()
        result = execute(request)
        logger.log_step(
            "tools",
            f"{call['name']}({call.get('args', {})})",
            getattr(result, "content", str(result)),
            tool_calls=_describe_tool_calls([call]),
            duration_ms=(time.monotonic() - started) * 1000,
            status=getattr(result, "status", "success"),
        )
        return result

    tool_node = ToolNode(
        tools, handle_tool_errors=format_tool_error, wrap_tool_call=log_tool_call
    )

    def route_after_agent(state: AgentState) -> Literal["tools", "formatter", "guard_finalize"]:
        """Conditional edge — серце ReAct-циклу."""
        if state.get("stop_reason"):
            return "guard_finalize"
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "formatter"

    def formatter_node(state: AgentState) -> dict:
        """Перетворити вільну відповідь агента на структурований FinalAnswer."""
        draft = str(state["messages"][-1].content)
        results = _collect_tool_results(state["messages"])
        results_text = "\n".join(
            f"- {r['tool']}({r['args']}) [{r['status']}]: {r['content']}" for r in results
        ) or "(інструменти не використовувалися)"
        prompt = (
            f"Питання користувача:\n{_user_query(state['messages'])}\n\n"
            f"Результати інструментів:\n{results_text}\n\n"
            f"Чернетка відповіді:\n{draft}"
        )

        started = time.monotonic()
        try:
            final = structured_llm.invoke(
                [SystemMessage(content=FORMATTER_PROMPT), HumanMessage(content=prompt)]
            )
        except Exception as e:
            final = FinalAnswer(
                answer=draft, confidence=0.5, sources=_successful_tool_names(results)
            )
            logger.log_step(
                "formatter", draft, f"fallback через помилку: {type(e).__name__}: {e}",
                duration_ms=(time.monotonic() - started) * 1000,
            )
        else:
            # sources беремо з фактичних викликів, а не з відповіді моделі
            final = final.model_copy(update={"sources": _successful_tool_names(results)})
            logger.log_step(
                "formatter", draft, final.model_dump(),
                duration_ms=(time.monotonic() - started) * 1000,
            )
        return {"final_answer": final.model_dump()}

    def guard_finalize_node(state: AgentState) -> dict:
        """Часткова відповідь після спрацювання захисного механізму.

        LLM тут свідомо НЕ викликається: ліміт уже вичерпано, тому відповідь
        збирається детерміновано з результатів, які агент встиг отримати.
        """
        results = [
            r for r in _collect_tool_results(state["messages"]) if r["status"] != "error"
        ]
        if results:
            collected = "\n".join(
                f"- {r['tool']}({r['args']}): {r['content'][:300]}" for r in results
            )
            answer = (
                f"{state['stop_message']} Часткова відповідь — дані, які вдалося "
                f"зібрати до зупинки:\n{collected}"
            )
            confidence = 0.3
        else:
            answer = f"{state['stop_message']} Зібрати дані до зупинки не вдалося."
            confidence = 0.0

        final = FinalAnswer(
            answer=answer, confidence=confidence, sources=_successful_tool_names(results)
        )
        logger.log_step(
            "guard_finalize", state["stop_reason"], final.model_dump(),
            stop_reason=state["stop_reason"],
        )
        return {"final_answer": final.model_dump()}

    # Вузли
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("formatter", formatter_node)
    graph.add_node("guard_finalize", guard_finalize_node)

    # Ребра: agent → (router) → tools / formatter / guard_finalize, tools → agent
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent)
    graph.add_edge("tools", "agent")
    graph.add_edge("formatter", END)
    graph.add_edge("guard_finalize", END)

    return graph.compile()


# ── Запуск агента ─────────────────────────────────────────────────────────────


def run_agent(
    query: str,
    *,
    llm: Any = None,
    tools: Optional[list] = None,
    safety: Optional[SafetyConfig] = None,
    run_id: str = "run",
) -> dict:
    """Виконати один запит і повернути результат разом з траєкторією.

    llm    — можна передати власну модель (наприклад, ScriptedChatModel у демо зациклення);
    tools  — можна підмінити набір інструментів (наприклад, повільним для демо тайм-ауту);
    safety — налаштування max_steps / timeout / max_repeats.
    """
    safety = safety or SafetyConfig()
    llm = llm or create_llm()
    logger = TrajectoryLogger(run_id=run_id, query=query)
    guard = SafetyGuard(safety)
    app = build_graph(llm, tools or TOOLS, guard, logger)

    started = time.monotonic()
    state = app.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "step_count": 0,
            "stop_reason": None,
            "stop_message": None,
            "final_answer": None,
        },
        # Запас понад max_steps, щоб першим спрацьовував наш захист, а не ліміт LangGraph
        config={"recursion_limit": safety.max_steps * 2 + 10},
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)

    executed_calls = [
        call
        for msg in state["messages"]
        if isinstance(msg, AIMessage)
        for call in _describe_tool_calls(msg.tool_calls)
    ]
    trajectory = logger.to_dict()
    trajectory["stop_reason"] = state.get("stop_reason")

    return {
        "run_id": run_id,
        "query": query,
        "status": "stopped" if state.get("stop_reason") else "success",
        "stop_reason": state.get("stop_reason"),
        "stop_message": state.get("stop_message"),
        "final_answer": state["final_answer"],
        "steps": state["step_count"],
        "tool_calls": executed_calls,
        "elapsed_ms": elapsed_ms,
        "trajectory": trajectory,
    }


def print_result(result: dict) -> None:
    """Надрукувати результат запуску у зручному для читання вигляді."""
    final = result["final_answer"]
    print(f"Запит:       {result['query']}")
    print(f"Статус:      {result['status']}"
          + (f" ({result['stop_reason']})" if result["stop_reason"] else ""))
    print(f"Кроків:      {result['steps']}, час: {result['elapsed_ms']} мс")
    print(f"Tool calls:  {[c['name'] for c in result['tool_calls']]}")
    print(f"Відповідь:   {final['answer']}")
    print(f"Confidence:  {final['confidence']}")
    print(f"Sources:     {final['sources']}")


if __name__ == "__main__":
    demo_queries = [
        "Я пробіг 10 км за 50:00. Який у мене був темп?",
        "Мені 30 років, пульс у спокої 60. Які в мене пульсові зони?",
        "Я пробіг 10 км за 50:00. За скільки я пробіжу марафон і скільки калорій "
        "спалю за цей марафон при вазі 75 кг?",
    ]
    for i, demo_query in enumerate(demo_queries, start=1):
        print(f"\n{'=' * 70}\nПриклад {i}\n{'=' * 70}")
        print_result(run_agent(demo_query, run_id=f"demo-{i}"))
