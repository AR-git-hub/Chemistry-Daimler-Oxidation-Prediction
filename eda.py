from pathlib import Path
import io

import pandas as pd

SOURCE = Path(__file__).resolve().parent / 'data/source'
OUTPUT = Path(__file__).resolve().parent / 'data/output'

def merge():
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
        table_res = table.merge(
            pivot,
            left_on=['Компонент', 'Наименование партии'],
            right_index=True,
            how='left'
        )

        for column in table_res.columns[8:]:
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

        return table_res


    train = helper(train)
    test = helper(test)
    return train, test


def clean(table: pd.DataFrame):
    cols = table.columns[3:]
    table[cols] = (
        table[cols]
        .astype("object")
        .fillna(0.0)
        .apply(lambda col: pd.to_numeric(
            col.map(lambda x: x.replace(",", ".") if isinstance(x, str) else x),
            errors="coerce"
        ))
    )


def correlation_table_to_csv(table: pd.DataFrame, output_path):
    """
    table:
        либо исходные данные (наблюдения),
        либо уже числовой DataFrame признаков

    output:
        CSV с колонками:
        - feature_1
        - feature_2
        - pearson
        - spearman
    """

    # если table — это исходные данные, считаем корреляции
    pearson_corr = table.corr(method="pearson", numeric_only=True)
    spearman_corr = table.corr(method="spearman", numeric_only=True)

    features = pearson_corr.columns

    rows = []

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1 = features[i]
            f2 = features[j]

            rows.append({
                "feature_1": f1,
                "feature_2": f2,
                "Пирсон": pearson_corr.loc[f1, f2],
                "Спирман": spearman_corr.loc[f1, f2]
            })

    result = pd.DataFrame(rows)

    # сортировка по модулю корреляции (самые сильные связи сверху)
    result["abs_pearson"] = result["Пирсон"].abs()
    result = result.sort_values("abs_pearson", ascending=False).drop(columns=["abs_pearson"])

    # сохранение
    result.to_csv(output_path / "corr_res.csv", index=False, decimal=".", encoding="utf-8")

    return result


def eda(table: pd.DataFrame, name: str):

    report_dir = OUTPUT / "eda" / name
    report_dir.mkdir(parents=True, exist_ok=True)

    def write_txt(filename: str, content: str):
        with open(report_dir / f"{filename}.txt", mode="w", encoding="UTF-8") as f:
            f.write(content)

    def write_csv(filename: str, content: pd.DataFrame | pd.Series):
        content.to_csv(report_dir / f"{filename}.csv", encoding="UTF-8")

    # info → .txt
    buffer = io.StringIO()
    table.info(buf=buffer)
    write_txt("info", buffer.getvalue())

    # describe, null_count, null_mean → .csv
    write_csv("describe", table.describe())
    write_csv("null_count", table.isna().sum().rename("null_count"))
    write_csv("null_mean", (table.isna().mean() * 100).sort_values(ascending=False).rename("null_mean_%"))

    # списки признаков → .txt
    write_txt("numeric_features",
              f"Числовые признаки: {list(table.select_dtypes(include='number').columns)}")
    write_txt("non_numeric_features",
              f"Не числовые признаки: {list(table.select_dtypes(exclude='number').columns)}")
    write_txt("const_features",
              f"Столбцы с постоянным значением: {[c for c in table.columns if table[c].nunique() <= 1]}")

    # корреляции → .csv
    write_csv("pearson_corr", table.corr(numeric_only=True))
    write_csv("spearman_corr", table.corr(method='spearman', numeric_only=True))

    correlation_table_to_csv(table, report_dir)


def save(table: pd.DataFrame, name: str):
    table_global = table[[
        "scenario_id",
        "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C",
        "Время испытания | - Daimler Oxidation Test (DOT), ч",
        "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс",
        "Дозировка катализатора, категория"
    ]].drop_duplicates()

    table_details = table[[
        "scenario_id",
        "Компонент",
        "Наименование партии",
        "Массовая доля, %",
        *table.columns[10:]
    ]]
    res = {
        "full": table,
        "global": table_global,
        "detailed": table_details
    }
    (OUTPUT / name).mkdir(parents=True, exist_ok=True)
    for key in res.keys():
        res[key].to_csv(OUTPUT / f'{name}/{name}_{key}.csv', index=False, decimal=".")


if __name__ == "__main__":
    train, test = merge()
    pairs = [
        {"table": train, "name": "train"},
        {"table": test, "name": "test"}
    ]
    for pair in pairs:
        table = pair["table"]
        OUTPUT = OUTPUT / "merged"
        save(table, pair["name"])
        OUTPUT = OUTPUT.parent
        clean(table)
        eda(table, pair["name"])
        save(table, pair["name"])
    train_temp = train.drop(columns=["Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %", "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"])
    eda(pd.concat([train_temp, test], ignore_index=True), "mutual")