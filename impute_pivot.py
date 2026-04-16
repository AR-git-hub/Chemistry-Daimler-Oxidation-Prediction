"""
Заполнение пропусков в файле pivot_components_pivot.csv

Структура файла:
  - Компонент + Наименование партии → идентификаторы строки
  - 29 числовых столбцов → свойства компонентов (вязкость, плотность, …)

Стратегия заполнения для каждой пустой ячейки:

  1. SKIP (не заполнять):
     Свойство ни разу не измерялось ни у одного компонента данного типа
     (напр., «Деэм.вода» у антиоксидантов → физически не актуально).

  2. MEAN_BY_COMPONENT (среднее/медиана по партиям того же компонента):
     Есть ≥1 другая партия этого же компонента с измеренным значением.
     Если CV > 50% → берём медиану вместо среднего.

  3. LLM (запрос к языковой модели):
     Свойство актуально для типа компонента, но у данного конкретного
     компонента нет ни одной партии с этим значением.
     LLM получает: тип компонента, свойство, значения у других компонентов
     того же типа — и оценивает вероятное значение.
     Если LLM не может оценить → ячейка остаётся пустой.

LLM подключается через OpenAI-совместимый API (Ollama).

Использование:
    pip install openai pandas
    # При необходимости настройте переменные окружения:
    #   OLLAMA_BASE_URL  (по умолчанию http://localhost:11434/v1)
    #   OLLAMA_MODEL     (по умолчанию qwen2.5:14b)
    python impute_pivot.py
"""

import pandas as pd
import numpy as np
import json
import logging
import os
import sys
from typing import Optional
from openai import OpenAI

# ════════════════════════ НАСТРОЙКИ ════════════════════════════

INPUT_FILE = os.environ.get("INPUT_FILE", "pivot_components_pivot.csv")
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "pivot_components_filled.csv")
REPORT_FILE = os.environ.get("REPORT_FILE", "imputation_report.csv")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:1234/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen/qwen3.5-35b-a3b")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "not-needed")

# Минимальная доля заполненности свойства в типе компонента,
# чтобы считать его «актуальным» и пытаться заполнить через LLM
MIN_TYPE_FILL_RATE = 0.25

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ════════════════════ РАБОТА С LLM ════════════════════════════

class LLMClient:
    """Обёртка для вызова LLM через OpenAI-совместимый API (Ollama)."""

    def __init__(self, base_url: str, model: str, api_key: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self._available: Optional[bool] = None

    # ── проверка связи ──────────────────────────────────────────
    def check_connection(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Ответь OK /no_think"}],
                max_tokens=1024,
                temperature=0.1,
                extra_body={
                    # или True
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            )
            self._available = bool(r.choices[0].message.content)
            return self._available
        except Exception as e:
            log.warning(f"LLM недоступна: {e}")
            self._available = False
            return False

    # ── универсальный запрос ────────────────────────────────────
    def _ask(self, system: str, user: str, temperature: float = 0.1) -> str:
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=4096,
                extra_body={
                    # или True
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            log.error(f"Ошибка LLM: {e}")
            return ""

    # ── оценка значения свойства ────────────────────────────────
    def impute_value(
        self,
        component: str,
        component_type: str,
        property_name: str,
        same_type_values: dict[str, float],
    ) -> Optional[float]:
        """
        Попросить LLM оценить пропущенное значение.

        Параметры:
            component:        полное имя компонента  (напр. «Базовое_масло_3»)
            component_type:   тип без номера         (напр. «Базовое_масло»)
            property_name:    название свойства       (столбец)
            same_type_values: словарь «Компонент/Партия → значение»
                              для ДРУГИХ компонентов того же типа

        Возвращает float или None (если оценить невозможно).
        """
        system = (
            "Ты — эксперт-химик в области моторных масел и смазочных материалов.\n"
            "Тебе нужно оценить пропущенное числовое значение свойства компонента.\n"
            "У тебя есть значения этого же свойства у других компонентов такого же типа.\n\n"
            "Правила ответа:\n"
            "- Если можешь обоснованно оценить значение — ответь ОДНИМ числом "
            "(десятичная точка, без единиц измерения, без пояснений).\n"
            "- Если информации недостаточно или оценка будет слишком неточной — "
            "ответь словом SKIP.\n"
            "- Никакого дополнительного текста. /no_think"
        )

        # Формируем таблицу «Компонент / Партия → значение»
        if same_type_values:
            lines = [f"  {k}: {v}" for k, v in same_type_values.items()]
            ref_block = "\n".join(lines)
        else:
            ref_block = "  (нет данных по другим компонентам этого типа)"

        user = (
            f"Компонент: {component}  (тип: {component_type})\n"
            f"Свойство:  {property_name}\n\n"
            f"Значения этого свойства у других компонентов типа «{component_type}»:\n"
            f"{ref_block}\n\n"
            f"Оцени наиболее вероятное значение для «{component}». /no_think"
        )

        answer = self._ask(system, user)
        return self._parse_number(answer)

    # ── решение: заполнять или нет ──────────────────────────────
    def decide_should_fill(
        self,
        component_type: str,
        property_name: str,
        fill_rate: float,
    ) -> tuple[bool, str]:
        """
        Спросить LLM, актуально ли данное свойство для типа компонента.
        Возвращает (should_fill, reason).
        """
        system = (
            "Ты — эксперт по моторным маслам. "
            "Определи, измеряется ли данное физико-химическое свойство "
            "для данного типа компонента масляной смеси.\n\n"
            "Ответь СТРОГО в формате JSON (без markdown-обёрток):\n"
            '{"fill": true, "reason": "краткое пояснение"}\n'
            "или\n"
            '{"fill": false, "reason": "краткое пояснение"}'
        )

        user = (
            f"Тип компонента: {component_type}\n"
            f"Свойство: {property_name}\n"
            f"Доля компонентов этого типа, у которых свойство измерено: "
            f"{fill_rate:.0%}\n\n"
            f"Нужно ли заполнять пропуск данного свойства у этого типа?"
        )

        answer = self._ask(system, user)
        try:
            start = answer.index("{")
            end = answer.rindex("}") + 1
            data = json.loads(answer[start:end])
            return bool(data.get("fill", False)), str(data.get("reason", ""))
        except (ValueError, json.JSONDecodeError):
            # Если LLM не смогла дать JSON — эвристика
            return fill_rate >= MIN_TYPE_FILL_RATE, \
                f"LLM не ответила корректно; эвристика (fill_rate={fill_rate:.0%})"

    # ── вспомогательные ─────────────────────────────────────────
    @staticmethod
    def _parse_number(text: str) -> Optional[float]:
        if not text or "SKIP" in text.upper():
            return None
        # Извлекаем первое число из строки
        buf = ""
        started = False
        for ch in text:
            if ch in "0123456789.-":
                buf += ch
                started = True
            elif started:
                break
        try:
            return float(buf) if buf else None
        except ValueError:
            return None


# ════════════════════ ВСПОМОГАТЕЛЬНЫЕ ═════════════════════════

def component_type(name: str) -> str:
    """Извлечь тип компонента: 'Базовое_масло_3' → 'Базовое_масло'."""
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return name


# ════════════════════ ОСНОВНАЯ ЛОГИКА ═════════════════════════

def impute(df: pd.DataFrame, llm: LLMClient) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Анализирует и заполняет пропуски.

    Возвращает:
        filled_df  — датафрейм с заполненными значениями
        report_df  — отчёт по каждой ячейке
    """
    df = df.copy()
    id_cols = ["Компонент", "Наименование партии"]
    prop_cols = [c for c in df.columns if c not in id_cols]

    # Добавим тип компонента
    df["_type"] = df["Компонент"].apply(component_type)

    # Предрассчитаем: для каждого (тип, свойство) → доля заполненности
    type_fill_rate: dict[tuple[str, str], float] = {}
    for ctype in df["_type"].unique():
        mask = df["_type"] == ctype
        n = mask.sum()
        for col in prop_cols:
            filled = df.loc[mask, col].notna().sum()
            type_fill_rate[(ctype, col)] = filled / n if n > 0 else 0.0

    # Предрассчитаем: для каждого (компонент, свойство) → значения других партий
    comp_vals: dict[tuple[str, str], list[float]] = {}
    for comp in df["Компонент"].unique():
        mask = df["Компонент"] == comp
        for col in prop_cols:
            vals = df.loc[mask, col].dropna().tolist()
            comp_vals[(comp, col)] = vals

    # ── Обход каждой пустой ячейки ──────────────────────────────
    report_rows = []
    total_filled = 0
    total_skipped = 0
    total_llm_calls = 0

    # Кэш решений LLM «актуально ли свойство для типа»
    relevance_cache: dict[tuple[str, str], tuple[bool, str]] = {}

    for idx in df.index:
        comp = df.at[idx, "Компонент"]
        batch = df.at[idx, "Наименование партии"]
        ctype = df.at[idx, "_type"]

        for col in prop_cols:
            if pd.notna(df.at[idx, col]):
                continue  # уже заполнено

            fill_rate = type_fill_rate.get((ctype, col), 0.0)

            # ─── SKIP: свойство не актуально для типа ───────────
            if fill_rate == 0.0:
                report_rows.append({
                    "Компонент": comp,
                    "Партия": batch,
                    "Свойство": col,
                    "Метод": "skip",
                    "Значение": "",
                    "Причина": "Свойство не измеряется ни у одного "
                               f"компонента типа «{ctype}»",
                })
                total_skipped += 1
                continue

            # ─── MEAN/MEDIAN: есть данные по другим партиям ─────
            other_vals = [
                v for i, v in enumerate(comp_vals[(comp, col)])
                if not (df.at[idx, col] == v)  # noqa — просто берём все
            ]
            # На самом деле comp_vals содержит ВСЕ непустые значения
            # по компоненту, включая текущую строку. Но текущая пуста,
            # так что other_vals = comp_vals[(comp, col)].
            other_vals = comp_vals[(comp, col)]

            if len(other_vals) >= 1:
                arr = np.array(other_vals)
                mean = float(arr.mean())
                if len(arr) >= 2 and mean != 0:
                    cv = float(arr.std() / abs(mean))
                else:
                    cv = 0.0

                if cv > 0.5 and len(arr) >= 2:
                    value = float(np.median(arr))
                    method = "median_by_component"
                else:
                    value = mean
                    method = "mean_by_component"

                df.at[idx, col] = value
                total_filled += 1
                report_rows.append({
                    "Компонент": comp,
                    "Партия": batch,
                    "Свойство": col,
                    "Метод": method,
                    "Значение": value,
                    "Причина": f"На основе {len(other_vals)} партий "
                    f"компонента «{comp}» (CV={cv:.1%})",
                })
                continue

            # ─── LLM: нет данных по тому же компоненту ──────────
            # Сначала проверяем, актуально ли свойство для типа
            cache_key = (ctype, col)
            if cache_key not in relevance_cache:
                if fill_rate < MIN_TYPE_FILL_RATE and llm.check_connection():
                    should, reason = llm.decide_should_fill(
                        ctype, col, fill_rate)
                    relevance_cache[cache_key] = (should, reason)
                else:
                    relevance_cache[cache_key] = (
                        fill_rate >= MIN_TYPE_FILL_RATE,
                        f"Эвристика: fill_rate={fill_rate:.0%}",
                    )

            should_fill, relevance_reason = relevance_cache[cache_key]

            if not should_fill:
                report_rows.append({
                    "Компонент": comp,
                    "Партия": batch,
                    "Свойство": col,
                    "Метод": "skip",
                    "Значение": "",
                    "Причина": f"Свойство не актуально: {relevance_reason}",
                })
                total_skipped += 1
                continue

            # Собираем значения у ДРУГИХ компонентов того же типа
            same_type_ref: dict[str, float] = {}
            for j in df.index:
                if df.at[j, "_type"] == ctype and pd.notna(df.at[j, col]):
                    key = f"{df.at[j, 'Компонент']} / {df.at[j, 'Наименование партии']}"
                    same_type_ref[key] = float(df.at[j, col])

            if not llm.check_connection():
                # LLM недоступна — пробуем среднее по типу
                if same_type_ref:
                    arr = np.array(list(same_type_ref.values()))
                    value = float(np.median(arr))
                    df.at[idx, col] = value
                    total_filled += 1
                    report_rows.append({
                        "Компонент": comp,
                        "Партия": batch,
                        "Свойство": col,
                        "Метод": "median_by_type",
                        "Значение": value,
                        "Причина": f"LLM недоступна; медиана по {len(same_type_ref)} "
                        f"партиям типа «{ctype}»",
                    })
                else:
                    report_rows.append({
                        "Компонент": comp,
                        "Партия": batch,
                        "Свойство": col,
                        "Метод": "skip",
                        "Значение": "",
                        "Причина": "LLM недоступна и нет данных по типу",
                    })
                    total_skipped += 1
                continue

            # Вызов LLM
            total_llm_calls += 1
            value = llm.impute_value(
                component=comp,
                component_type=ctype,
                property_name=col,
                same_type_values=same_type_ref,
            )

            if value is not None:
                df.at[idx, col] = value
                total_filled += 1
                report_rows.append({
                    "Компонент": comp,
                    "Партия": batch,
                    "Свойство": col,
                    "Метод": "llm",
                    "Значение": value,
                    "Причина": f"Оценка LLM на основе {len(same_type_ref)} "
                    f"компонентов типа «{ctype}»",
                })
            else:
                report_rows.append({
                    "Компонент": comp,
                    "Партия": batch,
                    "Свойство": col,
                    "Метод": "skip (llm отказ)",
                    "Значение": "",
                    "Причина": "LLM не смогла оценить значение",
                })
                total_skipped += 1

    df.drop(columns=["_type"], inplace=True)

    report_df = pd.DataFrame(report_rows)

    # ── Сводка ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  СВОДКА ЗАПОЛНЕНИЯ ПРОПУСКОВ")
    print("=" * 70)
    print(f"  Всего ячеек:                {len(df) * len(prop_cols)}")
    print(
        f"  Было заполнено:             {(len(df) * len(prop_cols)) - len(report_rows)}")
    print(f"  Было пропусков:             {len(report_rows)}")
    print(f"  ├─ Заполнено:               {total_filled}")
    print(f"  │   ├─ mean/median по комп.: "
          f"{len(report_df[report_df['Метод'].str.contains('by_component', na=False)])}")
    print(f"  │   ├─ median по типу:      "
          f"{len(report_df[report_df['Метод'] == 'median_by_type'])}")
    print(f"  │   └─ LLM:                 "
          f"{len(report_df[report_df['Метод'] == 'llm'])}")
    print(f"  └─ Пропущено (skip):        {total_skipped}")
    print(f"  Вызовов LLM:                {total_llm_calls}")
    print("=" * 70)
    print()

    return df, report_df


# ════════════════════════ MAIN ════════════════════════════════

def main():
    # ── Загрузка ────────────────────────────────────────────────
    if not os.path.exists(INPUT_FILE):
        log.error(f"Файл не найден: {INPUT_FILE}")
        log.info("Укажите путь через переменную окружения INPUT_FILE")
        sys.exit(1)

    log.info(f"Загрузка {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    log.info(f"Размер: {df.shape[0]} строк × {df.shape[1]} столбцов")

    total_cells = df.shape[0] * (df.shape[1] - 2)  # без Компонент и Партия
    missing_cells = df.iloc[:, 2:].isna().sum().sum()
    log.info(f"Пропусков: {missing_cells} из {total_cells} "
             f"({missing_cells/total_cells:.0%})")

    # ── LLM ─────────────────────────────────────────────────────
    log.info(f"Подключение к LLM: {OLLAMA_BASE_URL}  модель: {OLLAMA_MODEL}")
    llm = LLMClient(OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_API_KEY)

    if llm.check_connection():
        log.info("LLM доступна ✓")
    else:
        log.warning(
            "LLM недоступна — будут использованы только "
            "статистические методы (mean/median)"
        )

    # ── Заполнение ──────────────────────────────────────────────
    filled_df, report_df = impute(df, llm)

    # ── Сохранение ──────────────────────────────────────────────
    filled_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    log.info(f"Результат: {OUTPUT_FILE}")

    report_df.to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")
    log.info(f"Отчёт:     {REPORT_FILE}")

    # Проверка: сколько пропусков осталось
    remaining = filled_df.iloc[:, 2:].isna().sum().sum()
    log.info(f"Осталось пропусков: {remaining} из {total_cells}")


if __name__ == "__main__":
    main()
