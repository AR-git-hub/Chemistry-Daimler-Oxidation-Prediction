"""Build validated blend CSVs for the last submission slots (conservative: mostly wave2).

Эталон: predictions_stacked_k5_v7_ridge_tuned_rmse_wave2.csv (~0.0889 LB).
Слабые эксперименты (одиночный sumpool) не используем — только СТЕК sumpool-slot + малый вес.

Запуск из корня репозитория:
  PYTHONPATH=src python scripts/build_final_submit_pack.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    wave2 = ROOT / "predictions_stacked_k5_v7_ridge_tuned_rmse_wave2.csv"
    sumpool_stack = ROOT / "predictions_stacked_k5_v14_v14_sumpool_s311_m12_rich_dualalpha.csv"
    poly2 = ROOT / "predictions_stacked_k5_v7_poly2t1_tuned_wave2.csv"
    ridge_a6 = ROOT / "predictions_stacked_oof_k5_v3v7_ridge_a6.csv"
    test_csv = ROOT / "data" / "merged" / "test_full.csv"

    if not wave2.is_file():
        raise SystemExit(f"Missing anchor: {wave2}")
    if not sumpool_stack.is_file():
        raise SystemExit(f"Missing sumpool stack (run lb_v14_predict_stack for sumpool): {sumpool_stack}")

    jobs: list[tuple[str, Path, float, Path, float]] = []
    # (out_name, file_a, w_a, file_b, w_b)  weights before normalize in blend_predictions

    # 1–3: чуть подмешать sumpool-стек (лучший CV MSE слота), не ломая эталон
    jobs.append(("submit_blend_97wave2_03sumpoolstack", wave2, 0.97, sumpool_stack, 0.03))
    jobs.append(("submit_blend_94wave2_06sumpoolstack", wave2, 0.94, sumpool_stack, 0.06))
    jobs.append(("submit_blend_91wave2_09sumpoolstack", wave2, 0.91, sumpool_stack, 0.09))

    if poly2.is_file():
        jobs.append(("submit_blend_90wave2_10poly2wave2", wave2, 0.90, poly2, 0.10))
    elif ridge_a6.is_file():
        jobs.append(("submit_blend_90wave2_10ridgea6", wave2, 0.90, ridge_a6, 0.10))
    else:
        jobs.append(("submit_blend_90wave2_10sumpoolstack", wave2, 0.90, sumpool_stack, 0.10))

    if ridge_a6.is_file():
        jobs.append(("submit_blend_88wave2_12ridgea6", wave2, 0.88, ridge_a6, 0.12))
    else:
        jobs.append(("submit_blend_86wave2_14sumpoolstack", wave2, 0.86, sumpool_stack, 0.14))

    out_paths: list[Path] = []
    for name, fa, wa, fb, wb in jobs:
        outp = ROOT / f"predictions_{name}.csv"
        run(
            [
                PY,
                str(ROOT / "scripts" / "blend_predictions.py"),
                "--inputs",
                str(fa),
                str(fb),
                "--weights",
                str(wa),
                str(wb),
                "--output-path",
                str(outp),
            ]
        )
        run(
            [
                PY,
                str(ROOT / "scripts" / "validate_submission.py"),
                "--predictions-path",
                str(outp),
                "--test-path",
                str(test_csv),
            ]
        )
        out_paths.append(outp)

    plan = ROOT / "SUBMIT_PACK_ORDER.txt"
    lines = [
        "Порядок сабмитов (5 слотов; подставь свои LB после каждого):",
        "",
        "Якорь — не создавали заново:",
        f"  A0: {wave2.name}   (уже ~0.088865 LB)",
        "",
        "Новые бленды (консервативно, в основном wave2):",
    ]
    for i, p in enumerate(out_paths, start=1):
        lines.append(f"  A{i}: {p.name}")
    lines.extend(
        [
            "",
            "Реалистично: 0.083–0.084 за 5 сабмитов с такого уровня (~0.089) — редкость; цель пакета —",
            "не просесть и слегка выиграть за счёт sumpool-stack в малых долях + poly2/a6 варианты.",
            "",
            "НЕ слать одиночный sumpool TTA — уже дал ~0.103 LB.",
        ]
    )
    plan.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nWrote", plan, flush=True)
    for p in out_paths:
        print(" ", p.name, flush=True)


if __name__ == "__main__":
    main()
