# Chemistry-Daimler-Oxidation-Prediction

Пайплайн для подготовки и анализа физико-химических свойств компонентов масляных смесей. 

---

## Структура репозитория

```
├── data/
│   ├── input/
│   │   ├── daimler_component_properties.csv   # Исходные свойства компонентов в длинном формате
│   │   ├── daimler_mixtures_train.csv          # Обучающая выборка: смеси + результаты испытаний DOT
│   │   └── daimler_mixtures_test.csv           # Тестовая выборка: смеси без целевых переменных
│   │
│   ├── interim/
│   │   ├── pivot_components_pivot.csv          # Сводная таблица (компонент × свойство), с пропусками
│   │   ├── pivot_components_filled.csv         # Сводная таблица после заполнения пропусков
│   │   ├── daimler_component_properties_filled.csv  # Заполненные свойства в длинном формате
│   │   └── imputation_report.csv               # Лог заполнения: метод и причина для каждой ячейки
│   │
│   └── output/
│       ├── pivot_components.xlsx               # Сводная таблица с пропусками (Excel, форматированная)
│       └── pivot_components_filled.xlsx        # Сводная таблица без пропусков (Excel, форматированная)
│
├── pivot_builder.py         # Шаг 1. Строит сводную таблицу из длинного CSV, анализирует пропуски
├── impute_pivot.py          # Шаг 2. Заполняет пропуски: среднее/медиана по компоненту -> LLM (Ollama)
├── convert_pivot_to_xlsx.py # Шаг 3. Конвертирует заполненную сводную таблицу в форматированный XLSX
└── unpivot_to_long.py       # Вспомогательный. Переводит сводную таблицу обратно в длинный формат
```

---

## Скрипты

### `pivot_builder.py`
Читает `daimler_component_properties.csv` и строит сводную таблицу **компонент × показатель**.  
Сохраняет три файла: саму сводную таблицу, анализ пропусков по строкам (компонентам) и по столбцам (показателям).  
Также создаёт форматированный Excel с тремя листами.

### `impute_pivot.py`
Заполняет пропуски в `pivot_components_pivot.csv` по трёхуровневой стратегии:
1. **SKIP** - свойство физически нерелевантно для данного типа компонента.
2. **mean / median по компоненту** - есть другие партии того же компонента.
3. **LLM** (через OpenAI-совместимый API, по умолчанию Ollama) - нет партий с данным значением, но свойство актуально для типа.

Сохраняет заполненную таблицу и подробный лог (`imputation_report.csv`).

### `convert_pivot_to_xlsx.py`
Конвертирует любой CSV в стиле сводной таблицы в красиво оформленный XLSX:  
двухстрочная шапка с единицами измерения, чередование строк, заморозка панелей, автофильтр.

### `unpivot_to_long.py`
Обратная операция к `pivot_builder.py`: переводит сводную таблицу (широкий формат) обратно  
в длинный формат, совместимый с исходным `daimler_component_properties.csv`.

---

## Быстрый старт

```bash
pip install pandas openpyxl openai

# 1. Построить сводную таблицу
python pivot_builder.py --input data/input/daimler_component_properties.csv \
                        --output data/output/pivot_components.xlsx

# 2. Заполнить пропуски (LLM опционально - нужен запущенный Ollama/LM Studio)
INPUT_FILE=data/interim/pivot_components_pivot.csv \
OUTPUT_FILE=data/interim/pivot_components_filled.csv \
python impute_pivot.py

# 3. Экспортировать в Excel
python convert_pivot_to_xlsx.py  # пути задаются внутри скрипта
```
