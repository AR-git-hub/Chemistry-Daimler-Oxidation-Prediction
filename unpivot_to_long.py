"""
Перевод сводной таблицы (компонент × свойство) из CSV
в длинный формат аналогичный daimler_component_properties_cutted.csv.

Структура входного CSV:
    Компонент | Наименование партии | Свойство_1 | Свойство_2 | ...

Структура выходного CSV:
    Компонент | Наименование партии | Наименование показателя | Единица измерения_по_партиям | Значение показателя
"""

import pandas as pd

# ── Пути к файлам ──────────────────────────────────────────────────────────────
INPUT_CSV   = "pivot_components_filled.csv"          # входная сводная таблица
OUTPUT_CSV  = "daimler_component_properties_filled.csv" # выходной длинный формат

# Названия колонок-индексов в сводной таблице (не являются свойствами)
INDEX_COLS  = ["Компонент", "Наименование партии"]

# Опционально: словарь единиц измерения {название показателя: единица}.
# Если словарь пуст — колонка «Единица измерения_по_партиям» будет пустой.
# Можно заполнить вручную или указать путь к эталонному файлу ниже.
UNITS_SOURCE_CSV = ""   # путь к daimler_component_properties_cutted.csv
                        # для автоматического извлечения единиц; оставьте ""
                        # чтобы не использовать
# ──────────────────────────────────────────────────────────────────────────────


def load_units(source_path: str) -> dict:
    """Извлекает словарь {показатель: единица} из эталонного длинного CSV."""
    if not source_path:
        return {}
    df = pd.read_csv(source_path, encoding="utf-8-sig")
    return (
        df.dropna(subset=["Наименование показателя"])
        .drop_duplicates("Наименование показателя")
        .set_index("Наименование показателя")["Единица измерения_по_партиям"]
        .to_dict()
    )


def forward_fill_component(df: pd.DataFrame, col: str = "Компонент") -> pd.DataFrame:
    """Заполняет пропуски в колонке компонента (когда название не повторяется)."""
    df = df.copy()
    df[col] = df[col].ffill()
    return df


def unpivot(input_csv: str, output_csv: str, index_cols: list, units: dict):
    df = pd.read_csv(input_csv, encoding="utf-8-sig")

    # Восстанавливаем пропущенные названия компонентов
    df = forward_fill_component(df, index_cols[0])

    # Колонки со свойствами — всё кроме индексных
    value_cols = [c for c in df.columns if c not in index_cols]

    # melt: широкий → длинный
    long = df.melt(
        id_vars=index_cols,
        value_vars=value_cols,
        var_name="Наименование показателя",
        value_name="Значение показателя",
    )

    # Убираем строки без значения
    long = long.dropna(subset=["Значение показателя"])

    # Добавляем единицы измерения
    long["Единица измерения_по_партиям"] = (
        long["Наименование показателя"].map(units).fillna("")
    )

    # Финальный порядок колонок
    long = long[
        index_cols
        + ["Наименование показателя", "Единица измерения_по_партиям", "Значение показателя"]
    ]

    # Сортировка: по компоненту, партии, показателю
    long = long.sort_values(index_cols + ["Наименование показателя"]).reset_index(drop=True)

    long.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✓ Готово: {output_csv}  ({len(long)} строк)")


if __name__ == "__main__":
    units = load_units(UNITS_SOURCE_CSV)
    unpivot(INPUT_CSV, OUTPUT_CSV, INDEX_COLS, units)
