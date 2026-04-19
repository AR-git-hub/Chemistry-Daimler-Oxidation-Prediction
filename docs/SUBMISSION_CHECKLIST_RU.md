# Чеклист сдачи по ТЗ

## Что сдаем
1. Код обучения модели: `scripts/train.py` (использует `src/dot/train.py`).
2. Код получения предсказаний: `scripts/predict.py` (использует `src/dot/infer.py`).
3. `notebooks/inference.ipynb` — демонстрация формирования финального `predictions.csv`.
4. Параметры обученной модели (артефакты): папки в `artifacts/`.
5. Презентация: материал в `docs/presentation_ru.md`.

## Как использовать inference.ipynb
- Открыть `notebooks/inference.ipynb`.
- В ячейке с `MODE` выбрать:
  - `historical_best` — взять исторический лучший файл.
  - `final_candidate` — взять финальный кандидат.
- Выполнить ячейку.
- На выходе будет `predictions.csv` в корне проекта.

## Рекомендованный финальный файл
- `predictions_submit_blend_965wave2_03sumpool_005v18ens4.csv`

## Исторически лучший подтвержденный LB-файл
- `predictions_submit_blend_97wave2_03sumpoolstack.csv`
