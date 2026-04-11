#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_ground_truth_dataset.py
─────────────────────────────────
Генерує синтетичний CSV датасет де дрифти відбуваються в ТОЧНО ВІДОМИХ місцях.
Це дозволяє перевірити: чи твоя система знаходить дрифт там де він є насправді.

ЗАПУСК:
    python scripts/generate_ground_truth_dataset.py

ВИХІД:
    ground_truth_500.csv          ← подавай у main.py --custom-file
    ground_truth_500_labels.json  ← еталонні відповіді для порівняння

ФОРМАТ CSV (останній стовпець = мітка):
    F1,F2,F3,F4,F5,Label

ВІДОМІ ДРИФТИ (ground truth):
    - Інстанції 0–149:   Концепт A (стабільний)
    - Інстанція  150:    РАПТОВИЙ ДРИФТ (sudden)
    - Інстанції 150–299: Концепт B (стабільний)
    - Інстанції 300–399: ПОСТУПОВИЙ ДРИФТ (gradual), A→C
    - Інстанції 400–499: Концепт C (стабільний)

ВИКОРИСТАННЯ для верифікації:
    1. python scripts/generate_ground_truth_dataset.py
    2. python main.py --custom-file ground_truth_500.csv --window-size 50 --theta 0.35
    3. python verify_results.py experiments/results/.../report_debug.json ground_truth_500_labels.json
"""

import json
import csv
import random
import math
import sys
import io
from pathlib import Path

# Fix UTF-8 encoding on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SEED = 42
random.seed(SEED)

N = 500
N_FEATURES = 5
SUDDEN_AT = 150       # точна інстанція раптового дрифту
GRADUAL_START = 300   # початок поступового дрифту
GRADUAL_END = 400     # кінець поступового дрифту

# ── Концепти (центри розподілів) ──────────────────────────────────────────────

def concept_A(n: int) -> list:
    """Концепт A: значення близькі до 0.2–0.4"""
    rows = []
    for _ in range(n):
        row = [round(random.gauss(0.3, 0.05), 3) for _ in range(N_FEATURES)]
        label = 0
        rows.append(row + [label])
    return rows

def concept_B(n: int) -> list:
    """Концепт B: значення близькі до 0.7–0.9 — РІЗКО відрізняється від A"""
    rows = []
    for _ in range(n):
        row = [round(random.gauss(0.8, 0.05), 3) for _ in range(N_FEATURES)]
        label = 1
        rows.append(row + [label])
    return rows

def concept_C_gradual(n_total: int) -> list:
    """
    Поступовий перехід: B → C протягом n_total інстанцій.
    Кожна наступна інстанція більш схожа на C ніж B.
    """
    rows = []
    for i in range(n_total):
        t = i / n_total  # 0.0 → 1.0
        # лінійна інтерполяція між B (0.8) і C (0.5)
        mean = 0.8 * (1 - t) + 0.5 * t
        row = [round(random.gauss(mean, 0.05), 3) for _ in range(N_FEATURES)]
        label = 2
        rows.append(row + [label])
    return rows

def concept_C_stable(n: int) -> list:
    """Концепт C після завершення переходу"""
    rows = []
    for _ in range(n):
        row = [round(random.gauss(0.5, 0.05), 3) for _ in range(N_FEATURES)]
        label = 2
        rows.append(row + [label])
    return rows

# ── Генерація ─────────────────────────────────────────────────────────────────

print("[GENERATE] Синтезую датасет з відомими дрифтами...")
print(f"  - Раптовий дрифт на інстанції {SUDDEN_AT}")
print(f"  - Поступовий дрифт з інстанції {GRADUAL_START} до {GRADUAL_END}")

all_rows = []
all_rows += concept_A(SUDDEN_AT)                              # 0–149
all_rows += concept_B(GRADUAL_START - SUDDEN_AT)              # 150–299
all_rows += concept_C_gradual(GRADUAL_END - GRADUAL_START)    # 300–399
all_rows += concept_C_stable(N - GRADUAL_END)                 # 400–499

assert len(all_rows) == N, f"Expected {N} rows, got {len(all_rows)}"

# ── Запис CSV ─────────────────────────────────────────────────────────────────

csv_path = Path("ground_truth_500.csv")
header = [f"F{i+1}" for i in range(N_FEATURES)] + ["Label"]

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(all_rows)

print(f"[OK] CSV saved: {csv_path}  ({N} instances, {N_FEATURES} features)")

# ── Запис еталонних відповідей (ground truth) ─────────────────────────────────

ground_truth = {
    "description": "Synthetic dataset with known drift points",
    "total_instances": N,
    "window_size_recommendation": 50,
    "concepts": {
        "A": {"range": [0, SUDDEN_AT - 1], "label": 0, "mean": 0.3},
        "B": {"range": [SUDDEN_AT, GRADUAL_START - 1], "label": 1, "mean": 0.8},
        "C_gradual": {"range": [GRADUAL_START, GRADUAL_END - 1], "label": 2, "note": "gradual transition"},
        "C_stable": {"range": [GRADUAL_END, N - 1], "label": 2, "mean": 0.5},
    },
    "expected_drifts": [
        {
            "type": "sudden",
            "true_instance": SUDDEN_AT,
            "detection_window": [SUDDEN_AT, SUDDEN_AT + 60],
            "note": "Система з вікном W=50 може виявити на інстанції 150–210"
        },
        {
            "type": "gradual",
            "true_start": GRADUAL_START,
            "true_end": GRADUAL_END,
            "detection_window": [GRADUAL_START, GRADUAL_END + 50],
            "note": "Поступовий дрифт — очікуємо кілька детекцій або один тривалий епізод"
        },
    ],
    "how_to_verify": (
        "1. Запусти: python main.py --custom-file ground_truth_500.csv "
        "--window-size 50 --theta 0.35 --alpha 2.0\n"
        "2. Відкрий report_debug.json в experiments/results/\n"
        "3. Порівняй episodes[*].range з expected_drifts[*].detection_window\n"
        "4. Раптовий дрифт: хоча б 1 episode в діапазоні [150, 210]\n"
        "5. Поступовий дрифт: хоча б 1 episode в діапазоні [300, 450]"
    ),
}

labels_path = Path("ground_truth_500_labels.json")
with open(labels_path, "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, indent=2, ensure_ascii=False)

print(f"[OK] Ground truth saved: {labels_path}")

# ── Запис скрипту для верифікації ───────────────────────────────────────────

VERIFY_CODE = '''#!/usr/bin/env python3
"""
verify_results.py — перевіряє чи система знайшла дрифти там де вони є насправді.

ЗАПУСК:
    python verify_results.py experiments/results/custom_w50_t0.35_a2.0/report_debug.json ground_truth_500_labels.json
"""

import json
import sys

if len(sys.argv) < 3:
    print("ЗАПУСК: python verify_results.py <report_debug.json> <ground_truth_labels.json>")
    sys.exit(1)

debug_path = sys.argv[1]
gt_path = sys.argv[2]

try:
    with open(debug_path) as f:
        debug = json.load(f)
except FileNotFoundError:
    print(f"✗ Файл не знайдено: {debug_path}")
    sys.exit(1)

try:
    with open(gt_path) as f:
        gt = json.load(f)
except FileNotFoundError:
    print(f"✗ Файл не знайдено: {gt_path}")
    sys.exit(1)

detected_starts = []
if "episodes" in debug:
    for ep in debug["episodes"]:
        if isinstance(ep, dict):
            if "range" in ep:
                detected_starts.append(ep["range"][0])  # start of range
            elif "start" in ep:
                detected_starts.append(ep["start"])

print("\\n" + "="*60)
print("  ВЕРИФІКАЦІЯ РЕЗУЛЬТАТІВ ДЕТЕКЦІЇ ДРИФТУ")
print("="*60)
print(f"\\n📊 Знайдено епізодів: {len(detected_starts)}")
print(f"   Початки епізодів: {sorted(detected_starts)}")

all_ok = True
print(f"\\n🔍 Перевірка відомих дрифтів:")

for i, expected in enumerate(gt.get("expected_drifts", []), 1):
    window = expected["detection_window"]
    drift_type = expected.get("type", "unknown")

    found = any(window[0] <= s <= window[1] for s in detected_starts)
    status = "✓ ЗНАЙДЕНО" if found else "✗ ПРОПУЩЕНО"

    if not found:
        all_ok = False

    true_inst = expected.get("true_instance", expected.get("true_start"))
    print(f"\\n  [{i}] {drift_type.upper()} дрифт")
    print(f"      Очікуємо в діапазоні: [{window[0]}, {window[1]}]")
    print(f"      Результат: {status}")
    if found:
        in_window = [s for s in detected_starts if window[0] <= s <= window[1]]
        print(f"      Виявлено на інстанції(ях): {in_window}")

print("\\n" + "="*60)
if all_ok:
    print("✓ РЕЗУЛЬТАТ: ВСІ ДРИФТИ ЗНАЙДЕНО")
    print("  Система коректно детектує дрифти в еталонному датасеті.")
else:
    print("✗ РЕЗУЛЬТАТ: ДЕЯКІ ДРИФТИ ПРОПУЩЕНО")
    print("  Потрібна коригування параметрів θ, α, window_size")
print("="*60 + "\\n")

sys.exit(0 if all_ok else 1)
'''

verify_path = Path("verify_results.py")
with open(verify_path, "w", encoding="utf-8") as f:
    f.write(VERIFY_CODE)

print(f"[OK] Verifier saved: {verify_path}")

print()
print("="*70)
print("  КРОК ЗА КРОКОМ — ЯК ПЕРЕВІРИТИ СИСТЕМУ")
print("="*70)
print(f"\n1️⃣  Запусти детекцію на еталонному датасеті:")
print("    python main.py --custom-file ground_truth_500.csv \\")
print("                    --window-size 50 --theta 0.35 --alpha 2.0")
print(f"\n2️⃣  Знайди результати в: experiments/results/custom_w50_t0.35_a2.0/report_debug.json")
print(f"\n3️⃣  Верифікуй результати:")
print("    python verify_results.py \\")
print("        experiments/results/custom_w50_t0.35_a2.0/report_debug.json \\")
print("        ground_truth_500_labels.json")
print()
print("="*70)
print("  ОЧІКУВАНІ ДРИФТИ В ДАТАСЕТІ")
print("="*70)
print(f"\n🔴 РАПТОВИЙ ДРИФТ (sudden)")
print(f"   На інстанції: {SUDDEN_AT}")
print(f"   Система має знайти в діапазоні: [{SUDDEN_AT}, {SUDDEN_AT+60}]")
print(f"   Тип зміни: Концепт A (μ≈0.3) → Концепт B (μ≈0.8)")
print(f"\n🟠 ПОСТУПОВИЙ ДРИФТ (gradual)")
print(f"   На інстанціях: {GRADUAL_START}–{GRADUAL_END}")
print(f"   Система має знайти в діапазоні: [{GRADUAL_START}, {GRADUAL_END+50}]")
print(f"   Тип зміни: Плавний перехід Концепт B (μ≈0.8) → Концепт C (μ≈0.5)")
print()
