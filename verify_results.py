#!/usr/bin/env python3
"""
verify_results.py — перевіряє чи система знайшла дрифти там де вони є насправді.

ЗАПУСК:
    python verify_results.py experiments/results/custom_w300_t0.35_a2.0/report_debug.json ground_truth_5000_labels.json
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

print("\n" + "="*60)
print("  ВЕРИФІКАЦІЯ РЕЗУЛЬТАТІВ ДЕТЕКЦІЇ ДРИФТУ")
print("="*60)
print(f"\n📊 Знайдено епізодів: {len(detected_starts)}")
print(f"   Початки епізодів: {sorted(detected_starts)}")

all_ok = True
print(f"\n🔍 Перевірка відомих дрифтів:")

for i, expected in enumerate(gt.get("expected_drifts", []), 1):
    window = expected["detection_window"]
    drift_type = expected.get("type", "unknown")

    found = any(window[0] <= s <= window[1] for s in detected_starts)
    status = "✓ ЗНАЙДЕНО" if found else "✗ ПРОПУЩЕНО"

    if not found:
        all_ok = False

    true_inst = expected.get("true_instance", expected.get("true_start"))
    print(f"\n  [{i}] {drift_type.upper()} дрифт")
    print(f"      Очікуємо в діапазоні: [{window[0]}, {window[1]}]")
    print(f"      Результат: {status}")
    if found:
        in_window = [s for s in detected_starts if window[0] <= s <= window[1]]
        print(f"      Виявлено на інстанції(ях): {in_window}")

print("\n" + "="*60)
if all_ok:
    print("✓ РЕЗУЛЬТАТ: ВСІ ДРИФТИ ЗНАЙДЕНО")
    print("  Система коректно детектує дрифти в еталонному датасеті.")
else:
    print("✗ РЕЗУЛЬТАТ: ДЕЯКІ ДРИФТИ ПРОПУЩЕНО")
    print("  Потрібна коригування параметрів θ, α, window_size")
print("="*60 + "\n")

sys.exit(0 if all_ok else 1)
