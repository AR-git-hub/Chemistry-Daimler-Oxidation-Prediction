"""
Конвертация сводной таблицы компонент x свойство из CSV в XLSX с красивым оформлением.
Использование: python convert_pivot_to_xlsx.py <input.csv> <output.xlsx>
"""

import sys
import pandas as pd
import openpyxl
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side, GradientFill
)
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows

# ── цветовая схема ─────────────────────────────────────────────────────────────
HEADER_BG      = "1F3864"   # тёмно-синий — шапка столбцов
HEADER_FG      = "FFFFFF"   # белый текст
INDEX_BG       = "2E74B5"   # синий — строки индекса
INDEX_FG       = "FFFFFF"
SUBINDEX_BG    = "D6E4F0"   # светло-голубой — партия
SUBINDEX_FG    = "1F3864"
ALT_ROW_BG     = "EBF3FB"   # чередование строк
WHITE          = "FFFFFF"
BORDER_COLOR   = "B8CCE4"

def thin_border(color=BORDER_COLOR):
    s = Side(style="thin", color=color)
    return Border(left=s, right=s, top=s, bottom=s)

def header_border():
    thick = Side(style="medium", color="FFFFFF")
    thin  = Side(style="thin",   color=BORDER_COLOR)
    return Border(left=thin, right=thin, top=thick, bottom=thick)

def convert(input_csv: str, output_xlsx: str):
    # 1. Читаем и строим пивот
    df_raw = pd.read_csv(input_csv, encoding="utf-8-sig")

    required = {"Компонент", "Наименование партии",
                "Наименование показателя", "Значение показателя"}
    if not required.issubset(df_raw.columns):
        # Если файл уже является пивот-таблицей — используем как есть
        pivot = df_raw.reset_index(drop=True)
        index_cols = list(pivot.columns[:2])
        data_cols  = list(pivot.columns[2:])
    else:
        pivot = df_raw.pivot_table(
            index=["Компонент", "Наименование партии"],
            columns="Наименование показателя",
            values="Значение показателя",
            aggfunc="first"
        )
        pivot = pivot.reset_index()
        index_cols = ["Компонент", "Наименование партии"]
        data_cols  = [c for c in pivot.columns if c not in index_cols]

    # Единицы измерения (если есть в исходнике)
    units = {}
    if "Единица измерения_по_партиям" in df_raw.columns:
        units = (
            df_raw.dropna(subset=["Наименование показателя"])
            .drop_duplicates("Наименование показателя")
            .set_index("Наименование показателя")["Единица измерения_по_партиям"]
            .to_dict()
        )

    # 2. Создаём книгу
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Компоненты × Свойства"

    all_cols = index_cols + data_cols
    n_index  = len(index_cols)
    n_data   = len(data_cols)
    n_cols   = len(all_cols)
    n_rows   = len(pivot)

    HEADER_ROW   = 1   # заголовок показателя
    UNITS_ROW    = 2   # строка единиц
    DATA_START   = 3   # первая строка данных

    # 3. Шапка — индексные колонки
    for ci, col_name in enumerate(index_cols, start=1):
        c = ws.cell(row=HEADER_ROW, column=ci, value=col_name)
        c.font      = Font(name="Arial", bold=True, color=HEADER_FG, size=10)
        c.fill      = PatternFill("solid", fgColor=HEADER_BG)
        c.alignment = Alignment(horizontal="center", vertical="center",
                                wrap_text=True)
        c.border    = header_border()
        # Единицы
        u = ws.cell(row=UNITS_ROW, column=ci, value="")
        u.fill      = PatternFill("solid", fgColor=HEADER_BG)
        u.border    = header_border()

    # 4. Шапка — колонки свойств
    for ci, col_name in enumerate(data_cols, start=n_index + 1):
        c = ws.cell(row=HEADER_ROW, column=ci, value=col_name)
        c.font      = Font(name="Arial", bold=True, color=HEADER_FG, size=10)
        c.fill      = PatternFill("solid", fgColor=HEADER_BG)
        c.alignment = Alignment(horizontal="center", vertical="center",
                                wrap_text=True)
        c.border    = header_border()

        unit_val = units.get(col_name, "")
        u = ws.cell(row=UNITS_ROW, column=ci, value=unit_val)
        u.font      = Font(name="Arial", italic=True, color="D9E1F2", size=9)
        u.fill      = PatternFill("solid", fgColor="2B5397")
        u.alignment = Alignment(horizontal="center", vertical="center")
        u.border    = header_border()

    # 5. Данные
    prev_component = None
    for ri, row in pivot.iterrows():
        excel_row = DATA_START + ri
        component = row[index_cols[0]]
        is_alt    = (ri % 2 == 0)
        row_bg    = ALT_ROW_BG if is_alt else WHITE

        # Индексные ячейки
        for ci, col_name in enumerate(index_cols, start=1):
            val = row[col_name]
            c = ws.cell(row=excel_row, column=ci, value=val)
            c.border = thin_border()
            c.alignment = Alignment(horizontal="left", vertical="center",
                                    wrap_text=False)
            if ci == 1:  # Компонент
                if val != prev_component:
                    c.font = Font(name="Arial", bold=True,
                                  color=INDEX_FG, size=10)
                    c.fill = PatternFill("solid", fgColor=INDEX_BG)
                else:
                    c.font = Font(name="Arial", bold=True,
                                  color=INDEX_FG, size=10)
                    c.fill = PatternFill("solid", fgColor=INDEX_BG)
                    c.value = ""          # не повторяем название компонента
            else:         # Партия
                c.font = Font(name="Arial", bold=False,
                              color=SUBINDEX_FG, size=10)
                c.fill = PatternFill("solid", fgColor=SUBINDEX_BG)

        prev_component = component if str(row[index_cols[0]]) != "" else prev_component

        # Ячейки данных
        for ci, col_name in enumerate(data_cols, start=n_index + 1):
            raw_val = row[col_name]
            # Попытка привести к числу
            val = raw_val
            if pd.notna(raw_val):
                try:
                    val = float(str(raw_val).replace(",", "."))
                    if val == int(val):
                        val = int(val)
                except (ValueError, TypeError):
                    pass
            else:
                val = None

            c = ws.cell(row=excel_row, column=ci, value=val)
            c.font      = Font(name="Arial", size=10,
                               color="1F3864" if val is not None else "AAAAAA")
            c.fill      = PatternFill("solid", fgColor=row_bg)
            c.alignment = Alignment(horizontal="center", vertical="center")
            c.border    = thin_border()
            if val is None:
                c.value = "—"

    # 6. Ширина столбцов
    ws.column_dimensions[get_column_letter(1)].width = 26   # Компонент
    ws.column_dimensions[get_column_letter(2)].width = 14   # Партия
    for ci in range(n_index + 1, n_cols + 1):
        # авто-ширина по длине заголовка
        col_name = data_cols[ci - n_index - 1]
        length   = min(max(len(col_name) * 0.55, 12), 28)
        ws.column_dimensions[get_column_letter(ci)].width = length

    # 7. Высота строк шапки
    ws.row_dimensions[HEADER_ROW].height = 55
    ws.row_dimensions[UNITS_ROW].height  = 18
    for ri in range(n_rows):
        ws.row_dimensions[DATA_START + ri].height = 16

    # 8. Заморозка первых двух строк и двух колонок
    ws.freeze_panes = ws.cell(row=DATA_START, column=n_index + 1)

    # 9. Авто-фильтр на шапке
    ws.auto_filter.ref = (
        f"A{HEADER_ROW}:{get_column_letter(n_cols)}{DATA_START + n_rows - 1}"
    )

    # 10. Имя листа
    ws.sheet_view.showGridLines = False

    wb.save(output_xlsx)
    print(f"✓ Сохранено: {output_xlsx}  ({n_rows} строк × {n_cols} колонок)")


if __name__ == "__main__":
    INPUT_CSV   = "pivot_components_filled.csv"   # путь до входного CSV
    OUTPUT_XLSX = "pivot_components_filled.xlsx"  # путь до выходного XLSX

    convert(INPUT_CSV, OUTPUT_XLSX)
