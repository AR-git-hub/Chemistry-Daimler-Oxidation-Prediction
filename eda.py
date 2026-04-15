from pathlib import Path

import pandas as pd

def merge():
    SOURCE = Path(__file__).resolve().parent / 'data/source'
    OUTPUT = Path(__file__).resolve().parent / 'data/merged'
    props = pd.read_csv(SOURCE / 'daimler_component_properties.csv')
    train = pd.read_csv(SOURCE / 'daimler_mixtures_train.csv')
    test = pd.read_csv(SOURCE / 'daimler_mixtures_test.csv')

    def helper(table: pd.DataFrame):
        nonlocal props
        props["value"] = (
                props["Значение показателя"].astype(str) +
                pd.Series([' '] * props.shape[0]) +
                props["Единица измерения_по_партиям"].astype(str)
        )
        pivot = props.pivot_table(
            index=['Компонент', 'Наименование партии'],
            columns='Наименование показателя',
            values='value',
            aggfunc='first'
        )
        pivot.to_csv('2.csv')
        table_res = table.merge(
            pivot,
            left_on=['Компонент', 'Наименование партии'],
            right_index=True,
            how='left'
        )
        table_global = table_res[[
            "scenario_id",
            "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C",
            "Время испытания | - Daimler Oxidation Test (DOT), ч",
            "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс",
            "Дозировка катализатора, категория"
        ]].drop_duplicates()
        table_details = table_res[[
            "scenario_id",
            "Компонент",
            "Наименование партии",
            "Массовая доля, %",
            *table_res.columns[10:]
        ]]

        for column in table_res.columns[10:]:
            col = table_res[column].astype("string")
            parts = col.str.split()
            values = parts.str[0]
            units = parts.str[1]
            # проверка единиц (векторно)
            unique_units = units.dropna().unique()

            if len(unique_units) > 1:
                raise Exception(
                    f"Единицы измерения не совпадают в столбце {column}: {unique_units}"
                )
            table_res[column] = values

        return {
            "full": table_res,
            "global": table_global,
            "detailed": table_details
        }


    train = helper(train)
    test = helper(test)
    for key in train.keys():
        train[key].to_csv(OUTPUT / f'train_{key}.csv', index=False)
    for key in test.keys():
        test[key].to_csv(OUTPUT / f'test_{key}.csv', index=False)


if __name__ == "__main__":
    merge()