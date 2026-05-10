#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_drift_types_test.py
────────────────────────────
Генерує 3 окремих датасети для тестування кожного типу дрейфу ОКРЕМО.

ЗАПУСК:
    python scripts/generate_drift_types_test.py

ВИХІД:
    test_drift_sudden.csv      ← для тестування РАПТОВОГО дрейфу
    test_drift_gradual.csv     ← для тестування ПОСТУПОВОГО дрейфу
    test_drift_incremental.csv ← для тестування ІНКРЕМЕНТАЛЬНОГО дрейфу
"""

import csv
import random
import json
from pathlib import Path

SEED = 42
random.seed(SEED)

N_FEATURES = 5
TOTAL_INSTANCES = 5000
SUDDEN_POINT = 2000
SUDDEN_DURATION = 20
GRADUAL_START = 1500
GRADUAL_DURATION = 2000
INCREMENTAL_START = 1500
INCREMENTAL_DURATION = 2000


# ═════════════════════════════════════ SUDDEN DRIFT ════════════════════════════════

def generate_sudden_drift():
    """
    РАПТОВИЙ ДРЕЙФ: різка ізольована зміна

    Інстанції 0–1999:   Стабільна концепція (mean=0.3)
    Інстанції 2000–2019: ⚡ РАПТОВИЙ ДРЕЙФ (mean=0.9) — 20 точок, потім назад
    Інстанції 2020–4999: Вертаємось до норми (mean=0.3)
    """
    rows = []

    # До дрейфу
    for i in range(SUDDEN_POINT):
        row = [round(random.gauss(0.3, 0.05), 3) for _ in range(N_FEATURES)]
        rows.append(row + [0])

    # ⚡ Раптова зміна (6 точок на піку)
    for i in range(SUDDEN_DURATION):
        row = [round(random.gauss(0.9, 0.05), 3) for _ in range(N_FEATURES)]
        rows.append(row + [1])

    # Повернення до норми
    for i in range(TOTAL_INSTANCES - SUDDEN_POINT - SUDDEN_DURATION):
        row = [round(random.gauss(0.3, 0.05), 3) for _ in range(N_FEATURES)]
        rows.append(row + [0])

    assert len(rows) == TOTAL_INSTANCES
    return rows


def generate_gradual_drift():
    """
    ПОСТУПОВИЙ ДРЕЙФ: повільна, але стійка зміна

    Інстанції 0–1499:    Стабільна концепція (mean=0.2)
    Інстанції 1500–3499:  🌊 ПОСТУПОВИЙ дрейф (лінійна зміна 0.2→0.8)
    Інстанції 3500–4999: Нова концепція (mean=0.8)
    """
    rows = []

    # До дрейфу
    for i in range(GRADUAL_START):
        row = [round(random.gauss(0.2, 0.05), 3) for _ in range(N_FEATURES)]
        rows.append(row + [0])

    # 🌊 Поступовий дрейф (2000 інстанцій переходу)
    for i in range(GRADUAL_DURATION):
        t = i / GRADUAL_DURATION  # 0.0 → 1.0
        mean = 0.2 * (1 - t) + 0.8 * t  # лінійна інтерполяція
        row = [round(random.gauss(mean, 0.05), 3) for _ in range(N_FEATURES)]
        rows.append(row + [1])

    # Після дрейфу
    for i in range(TOTAL_INSTANCES - GRADUAL_START - GRADUAL_DURATION):
        row = [round(random.gauss(0.8, 0.05), 3) for _ in range(N_FEATURES)]
        rows.append(row + [1])

    assert len(rows) == TOTAL_INSTANCES
    return rows


def generate_incremental_drift():
    """
    ІНКРЕМЕНТАЛЬНИЙ ДРЕЙФ: повільне МОНОТОННЕ нарощування

    Інстанції 0–1499:    Стабільна концепція (mean=0.2)
    Інстанції 1500–3499:  📈 ІНКРЕМЕНТАЛЬНИЙ дрейф (step +0.0003 за інстанцію)
    Інстанції 3500–4999: Нова концепція (mean=0.8)

    Особливість: ≥70% кроків мають бути позитивні (монотонні)
    """
    rows = []

    # До дрейфу
    for i in range(INCREMENTAL_START):
        row = [round(random.gauss(0.2, 0.05), 3) for _ in range(N_FEATURES)]
        rows.append(row + [0])

    # 📈 Інкрементальний дрейф (2000 інстанцій, step за step)
    current_mean = 0.2
    for i in range(INCREMENTAL_DURATION):
        # Позитивний крок: mean += 0.0003 (~+0.6 за 2000 кроків = 0.2 → 0.8)
        current_mean += 0.0003 + random.gauss(0, 0.01)  # трохи шуму
        current_mean = max(0.2, min(0.8, current_mean))  # clip
        row = [round(random.gauss(current_mean, 0.05), 3) for _ in range(N_FEATURES)]
        rows.append(row + [1])

    # Після дрейфу
    for i in range(TOTAL_INSTANCES - INCREMENTAL_START - INCREMENTAL_DURATION):
        row = [round(random.gauss(0.8, 0.05), 3) for _ in range(N_FEATURES)]
        rows.append(row + [1])

    assert len(rows) == TOTAL_INSTANCES
    return rows


# ═════════════════════════════════════ ГЕНЕРАЦІЯ ════════════════════════════════════

def write_csv(filename, rows, description):
    """Запис датасету у CSV"""
    path = Path(filename)
    header = [f"F{i+1}" for i in range(N_FEATURES)] + ["Label"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[OK] {description}: {path} ({len(rows)} instances)")


def write_ground_truth(filename, drift_type, position, duration):
    """Запис еталонної інформації"""
    gt = {
        "description": f"Test dataset for {drift_type} drift detection",
        "drift_type": drift_type,
        "total_instances": TOTAL_INSTANCES,
        "window_size_recommendation": 300,
        "expected_drift": {
            "type": drift_type,
            "start": position,
            "duration": duration,
            "end": position + duration - 1
        },
        "expected_detection_window": {
            "min": max(0, position - 500),
            "max": min(TOTAL_INSTANCES, position + duration + 500)
        }
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)


print("""
╔════════════════════════════════════════════════════════════════╗
║          ГЕНЕРАЦІЯ ТЕСТОВИХ ДАТАСЕТІВ ДРЕЙФІВ                ║
╚════════════════════════════════════════════════════════════════╝
""")

# ⚡ SUDDEN
rows_sudden = generate_sudden_drift()
write_csv("test_drift_sudden.csv", rows_sudden, "🚨 SUDDEN (раптовий) дрейф")
write_ground_truth("test_drift_sudden_gt.json", "sudden", SUDDEN_POINT, SUDDEN_DURATION)

# 🌊 GRADUAL
rows_gradual = generate_gradual_drift()
write_csv("test_drift_gradual.csv", rows_gradual, "🌊 GRADUAL (поступовий) дрейф")
write_ground_truth("test_drift_gradual_gt.json", "gradual", GRADUAL_START, GRADUAL_DURATION)

# 📈 INCREMENTAL
rows_incremental = generate_incremental_drift()
write_csv("test_drift_incremental.csv", rows_incremental, "📈 INCREMENTAL (інкрементальний) дрейф")
write_ground_truth("test_drift_incremental_gt.json", "incremental", INCREMENTAL_START, INCREMENTAL_DURATION)

print("""
╔════════════════════════════════════════════════════════════════╗
║                   ГОТОВО! ТЕСТУЖТЕ:                           ║
╚════════════════════════════════════════════════════════════════╝

🚨 РАПТОВИЙ дрейф:
     python main.py --custom-file test_drift_sudden.csv \\
         --window-size 300 --theta 0.4 --language en

🌊 ПОСТУПОВИЙ дрейф:
     python main.py --custom-file test_drift_gradual.csv \\
         --window-size 300 --theta 0.35 --language en

📈 ІНКРЕМЕНТАЛЬНИЙ дрейф:
     python main.py --custom-file test_drift_incremental.csv \\
         --window-size 300 --theta 0.3 --language en

""")
