"""Завдання 1. Інструменти спортивного асистента з Pydantic v2 схемами.

Домен: тренування з бігу та загальна фізична активність.
Кожен інструмент має:
  * BaseModel зі схемою параметрів (Field з description);
  * field_validator для перевірки та нормалізації вхідних даних;
  * docstring, який LLM читає під час вибору інструмента.

Усі інструменти детерміновані (формули, а не зовнішні API), тому їх результати
легко перевірити вручну.

Запуск демонстрації:  python tools.py
"""

import json
import re
from typing import Optional

from langchain_core.tools import tool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


# ── Допоміжні функції ─────────────────────────────────────────────────────────

_TIME_RE = re.compile(r"^(\d{1,3}):(\d{2})(?::(\d{2}))?$")

MAX_DISTANCE_KM = 1000.0
MAX_TIME_SECONDS = 48 * 3600


def parse_time_to_seconds(value: str) -> int:
    """Перетворити рядок часу ("MM:SS" або "H:MM:SS") на секунди."""
    match = _TIME_RE.match(value.strip())
    if not match:
        raise ValueError(
            'Час має бути у форматі "MM:SS" або "H:MM:SS", '
            'наприклад "50:00" або "1:45:30"'
        )
    first, second, third = match.groups()
    if third is None:
        hours, minutes, seconds = 0, int(first), int(second)
    else:
        hours, minutes, seconds = int(first), int(second), int(third)
        if minutes >= 60:
            raise ValueError("Хвилини у форматі H:MM:SS мають бути від 00 до 59")
    if seconds >= 60:
        raise ValueError("Секунди мають бути від 00 до 59")

    total = hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise ValueError("Час має бути більшим за нуль")
    if total > MAX_TIME_SECONDS:
        raise ValueError("Час не може перевищувати 48 годин")
    return total


def format_seconds(total_seconds: float) -> str:
    """Секунди → рядок "H:MM:SS"."""
    total = int(round(total_seconds))
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def format_pace(seconds_per_km: float) -> str:
    """Секунди на кілометр → рядок темпу "M:SS"."""
    total = int(round(seconds_per_km))
    minutes, seconds = divmod(total, 60)
    return f"{minutes}:{seconds:02d}"


def _check_distance(value: float) -> float:
    """Спільна перевірка дистанції для кількох схем."""
    if value <= 0:
        raise ValueError("Дистанція має бути більшою за 0 км")
    if value > MAX_DISTANCE_KM:
        raise ValueError(f"Дистанція не може перевищувати {MAX_DISTANCE_KM:.0f} км")
    return value


def _normalize_time(value: str) -> str:
    """Спільна перевірка часу: повертає нормалізований рядок "H:MM:SS"."""
    return format_seconds(parse_time_to_seconds(value))


def _to_json(data: dict) -> str:
    """Результат інструмента — JSON із чіткими полями, а не суцільний текст."""
    return json.dumps(data, ensure_ascii=False)


# ── Tool 1: темп і швидкість ──────────────────────────────────────────────────


class PaceInput(BaseModel):
    """Параметри для розрахунку темпу."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    distance_km: float = Field(
        description="Подолана дистанція у кілометрах, наприклад 10 або 21.0975"
    )
    time: str = Field(
        description=(
            'Час подолання дистанції у форматі "MM:SS" або "H:MM:SS". '
            'Приклади: "50:00" — 50 хвилин; "1:45:30" — 1 год 45 хв 30 с'
        )
    )

    @field_validator("distance_km")
    @classmethod
    def distance_in_range(cls, v: float) -> float:
        return _check_distance(v)

    @field_validator("time")
    @classmethod
    def time_is_valid(cls, v: str) -> str:
        return _normalize_time(v)


@tool(args_schema=PaceInput)
def calculate_pace(distance_km: float, time: str) -> str:
    """Розрахувати темп (хв/км) і середню швидкість (км/год) за дистанцією та часом.

    Використовуй цей інструмент, коли відомі дистанція і час та потрібно
    дізнатися темп або швидкість, а також щоб порівняти темп кількох забігів
    (тоді виклич інструмент окремо для кожного забігу).
    Приклад: "пробіг 10 км за 50:00" → distance_km=10, time="50:00".

    Повертає JSON: pace_per_km ("M:SS"), pace_seconds_per_km, speed_kmh.
    """
    seconds = parse_time_to_seconds(time)
    pace_seconds = seconds / distance_km
    return _to_json(
        {
            "distance_km": distance_km,
            "time": format_seconds(seconds),
            "pace_per_km": format_pace(pace_seconds),
            "pace_seconds_per_km": round(pace_seconds, 1),
            "speed_kmh": round(distance_km / (seconds / 3600), 2),
        }
    )


# ── Tool 2: прогноз часу на іншій дистанції ───────────────────────────────────

# Показник втоми у формулі Рігеля: T2 = T1 * (D2 / D1) ** 1.06
RIEGEL_EXPONENT = 1.06


class RacePredictionInput(BaseModel):
    """Параметри для прогнозу результату на іншій дистанції."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    known_distance_km: float = Field(
        description="Дистанція вже відомого результату у кілометрах, наприклад 10"
    )
    known_time: str = Field(
        description=(
            'Час на відомій дистанції у форматі "MM:SS" або "H:MM:SS", '
            'наприклад "50:00"'
        )
    )
    target_distance_km: float = Field(
        description=(
            "Дистанція, для якої потрібен прогноз, у кілометрах. "
            "Півмарафон — 21.0975, марафон — 42.195"
        )
    )

    @field_validator("known_distance_km", "target_distance_km")
    @classmethod
    def distance_in_range(cls, v: float) -> float:
        return _check_distance(v)

    @field_validator("known_time")
    @classmethod
    def time_is_valid(cls, v: str) -> str:
        return _normalize_time(v)

    @model_validator(mode="after")
    def distances_differ(self):
        # Правило між двома полями, тому model_validator, а не field_validator
        if self.known_distance_km == self.target_distance_km:
            raise ValueError("Цільова дистанція має відрізнятися від відомої")
        return self


@tool(args_schema=RacePredictionInput)
def predict_race_time(
    known_distance_km: float, known_time: str, target_distance_km: float
) -> str:
    """Спрогнозувати час на цільовій дистанції за відомим результатом (формула Рігеля).

    Використовуй цей інструмент, коли користувач має результат на одній
    дистанції і хоче знати очікуваний час на іншій (наприклад, з 10 км
    на півмарафон чи марафон). Не використовуй для розрахунку темпу вже
    завершеного забігу — для цього є calculate_pace.
    Приклад: 10 км за 50:00, прогноз на півмарафон →
    known_distance_km=10, known_time="50:00", target_distance_km=21.0975.

    Повертає JSON: predicted_time ("H:MM:SS"), predicted_time_minutes,
    predicted_pace_per_km ("M:SS").
    """
    known_seconds = parse_time_to_seconds(known_time)
    predicted_seconds = known_seconds * (
        (target_distance_km / known_distance_km) ** RIEGEL_EXPONENT
    )
    return _to_json(
        {
            "known_distance_km": known_distance_km,
            "known_time": format_seconds(known_seconds),
            "target_distance_km": target_distance_km,
            "predicted_time": format_seconds(predicted_seconds),
            "predicted_time_minutes": round(predicted_seconds / 60, 1),
            "predicted_pace_per_km": format_pace(predicted_seconds / target_distance_km),
            "method": "формула Рігеля, показник 1.06",
        }
    )


# ── Tool 3: пульсові зони ─────────────────────────────────────────────────────

# (номер зони, назва, нижня і верхня межа як частка інтенсивності)
HR_ZONES = [
    (1, "відновлення", 0.50, 0.60),
    (2, "аеробна база", 0.60, 0.70),
    (3, "темпова", 0.70, 0.80),
    (4, "порогова", 0.80, 0.90),
    (5, "максимальна", 0.90, 1.00),
]


class HeartRateZonesInput(BaseModel):
    """Параметри для розрахунку пульсових зон."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(description="Вік людини у повних роках (від 10 до 100)")
    resting_hr: Optional[int] = Field(
        default=None,
        description=(
            "Пульс у спокої, ударів за хвилину (від 30 до 120). Необов'язковий: "
            "якщо вказано — зони рахуються точніше, за методом Карвонена"
        ),
    )

    @field_validator("age")
    @classmethod
    def age_in_range(cls, v: int) -> int:
        if not 10 <= v <= 100:
            raise ValueError("Вік має бути від 10 до 100 років")
        return v

    @field_validator("resting_hr")
    @classmethod
    def resting_hr_in_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not 30 <= v <= 120:
            raise ValueError("Пульс у спокої має бути від 30 до 120 уд/хв")
        return v

    @model_validator(mode="after")
    def resting_below_max(self):
        if self.resting_hr is not None and self.resting_hr >= 220 - self.age:
            raise ValueError(
                "Пульс у спокої має бути меншим за максимальний пульс (220 − вік)"
            )
        return self


@tool(args_schema=HeartRateZonesInput)
def calculate_heart_rate_zones(age: int, resting_hr: Optional[int] = None) -> str:
    """Розрахувати максимальний пульс і п'ять тренувальних пульсових зон.

    Використовуй цей інструмент, коли користувач питає про пульсові зони,
    максимальний пульс або на якому пульсі тренуватися. Потрібен вік;
    пульс у спокої передавай лише тоді, коли користувач його назвав.
    Приклад: "мені 30 років, пульс спокою 60" → age=30, resting_hr=60.

    Повертає JSON: max_hr, method і список zones (zone, name, min_bpm, max_bpm).
    """
    max_hr = 220 - age
    zones = []
    for number, name, low, high in HR_ZONES:
        if resting_hr is None:
            min_bpm, max_bpm = max_hr * low, max_hr * high
        else:
            # Метод Карвонена: відсоток береться від резерву пульсу
            reserve = max_hr - resting_hr
            min_bpm, max_bpm = resting_hr + reserve * low, resting_hr + reserve * high
        zones.append(
            {
                "zone": number,
                "name": name,
                "min_bpm": round(min_bpm),
                "max_bpm": round(max_bpm),
            }
        )
    return _to_json(
        {
            "age": age,
            "resting_hr": resting_hr,
            "max_hr": max_hr,
            "method": "Карвонен" if resting_hr is not None else "відсоток від макс. пульсу",
            "zones": zones,
        }
    )


# ── Tool 4: витрата калорій ───────────────────────────────────────────────────

# MET — у скільки разів активність енергозатратніша за стан спокою
MET_VALUES = {
    "running": 9.8,
    "jogging": 7.0,
    "walking": 3.5,
    "cycling": 7.5,
    "swimming": 6.0,
    "football": 7.0,
    "basketball": 6.5,
    "tennis": 7.3,
    "strength_training": 3.5,
    "yoga": 2.5,
}

ACTIVITY_ALIASES = {
    "біг": "running",
    "біг підтюпцем": "jogging",
    "ходьба": "walking",
    "велосипед": "cycling",
    "велоспорт": "cycling",
    "плавання": "swimming",
    "футбол": "football",
    "баскетбол": "basketball",
    "теніс": "tennis",
    "силове тренування": "strength_training",
    "йога": "yoga",
}


class CaloriesInput(BaseModel):
    """Параметри для оцінки витрати калорій."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    activity: str = Field(
        description=(
            "Вид активності англійською. Допустимі значення: "
            + ", ".join(MET_VALUES)
        )
    )
    weight_kg: float = Field(description="Вага людини у кілограмах (від 30 до 250)")
    duration_minutes: float = Field(
        description="Тривалість активності у хвилинах (від 1 до 1440)"
    )

    @field_validator("activity")
    @classmethod
    def activity_is_supported(cls, v: str) -> str:
        normalized = v.lower().replace("-", "_")
        normalized = ACTIVITY_ALIASES.get(normalized, normalized).replace(" ", "_")
        if normalized not in MET_VALUES:
            raise ValueError(
                f"Невідомий вид активності '{v}'. Допустимі: {', '.join(MET_VALUES)}"
            )
        return normalized

    @field_validator("weight_kg")
    @classmethod
    def weight_in_range(cls, v: float) -> float:
        if not 30 <= v <= 250:
            raise ValueError("Вага має бути від 30 до 250 кг")
        return v

    @field_validator("duration_minutes")
    @classmethod
    def duration_in_range(cls, v: float) -> float:
        if not 1 <= v <= 1440:
            raise ValueError("Тривалість має бути від 1 до 1440 хвилин")
        return v


@tool(args_schema=CaloriesInput)
def estimate_calories(activity: str, weight_kg: float, duration_minutes: float) -> str:
    """Оцінити витрату калорій за видом активності, вагою та тривалістю (метод MET).

    Використовуй цей інструмент, коли користувач питає, скільки калорій він
    спалив або спалить під час тренування чи забігу. Тривалість передавай
    у хвилинах: якщо відомий час "1:50:19", спершу переведи його у хвилини
    (або візьми поле predicted_time_minutes з predict_race_time).
    Приклад: біг 60 хвилин при вазі 75 кг →
    activity="running", weight_kg=75, duration_minutes=60.

    Повертає JSON: calories_kcal, met.
    """
    met = MET_VALUES[activity]
    # ккал = MET × вага (кг) × тривалість (год)
    calories = met * weight_kg * (duration_minutes / 60)
    return _to_json(
        {
            "activity": activity,
            "weight_kg": weight_kg,
            "duration_minutes": duration_minutes,
            "met": met,
            "calories_kcal": round(calories),
            "note": "приблизна оцінка за усередненим значенням MET",
        }
    )


# Список інструментів для bind_tools та ToolNode
TOOLS = [
    calculate_pace,
    predict_race_time,
    calculate_heart_rate_zones,
    estimate_calories,
]


def _demo() -> None:
    """Показати успішні та неуспішні виклики кожного інструмента."""
    cases = [
        (calculate_pace, {"distance_km": 10, "time": "50:00"}),
        (calculate_pace, {"distance_km": 10, "time": "50:75"}),
        (calculate_pace, {"distance_km": -5, "time": "25:00"}),
        (
            predict_race_time,
            {"known_distance_km": 10, "known_time": "50:00", "target_distance_km": 21.0975},
        ),
        (
            predict_race_time,
            {"known_distance_km": 10, "known_time": "50:00", "target_distance_km": 10},
        ),
        (calculate_heart_rate_zones, {"age": 30, "resting_hr": 60}),
        (calculate_heart_rate_zones, {"age": 150}),
        (estimate_calories, {"activity": "біг", "weight_kg": 75, "duration_minutes": 60}),
        (estimate_calories, {"activity": "шахи", "weight_kg": 75, "duration_minutes": 60}),
    ]
    for tool_obj, args in cases:
        print(f"\n{tool_obj.name}({args})")
        try:
            print("  OK:", tool_obj.invoke(args))
        except ValidationError as e:
            for err in e.errors():
                field = err["loc"][0] if err["loc"] else "модель"
                print(f"  ВІДХИЛЕНО [{field}]: {err['msg']}")


if __name__ == "__main__":
    _demo()
