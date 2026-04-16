# Chemistry-Daimler-Oxidation-Prediction
## ветка `features/eda`
- файл `eda.py`: здесь сосредоточены функции слияния таблиц, очистки данных, разведочного анализа данных. Все результаты сохраняются в data/output
- `data/output/train` - таблицы с тренировочной выборкой новой структуры. Таблицы `train_full.csv` содержит все необходимые данные, остальные - лишь часть
- `data/output/test` - аналогично `data/output/train`
- `data/output/eda_graphs` - результаты EDA. Главный файл здесь - это результаты корреляции `train/corr_res.csv` или `test/corr_res.csv`.