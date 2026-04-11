# FCA Drift Detection System - Quick Start

**⏱️ Читайте: 5-10 хвилин**

Цей документ дає швидкий старт для розуміння системи.

---

## Що та Чому

### The Problem

Потокові дані змінювалися, але традиційні методи (DDM, EDDM) цього не помічали.

### The Solution

Наша система виявляє **концептуальні зміни** разом з класифікацієї їх в смислові **епізоди**.

```
Raw Stream Data
    ↓
┌──────────────────┐
│ FCA Detector     │  Knows: WHERE drift occurred
│ (55 raw points)  │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Drift Aggregator │  Knows: WHEN, WHAT TYPE, FOR HOW LONG
│ (8 episodes)     │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Reports & Graphs │  Shows: Clear visualization + statistics
└──────────────────┘
```

---

## 3 Основні Компоненти

### 1️⃣ FCA Drift Detector

**Що робить?** Знаходить точки, де структура данних змінилась.

**Як?**

- Використовує **формальні концептуальні решітки** (FCA)
- Порівнює послідовні решітки
- Обчислює "відстань" між ними (0 = однакові, 1 = різні)
- Флагує як drift якщо відстань > порогу

**Вхід:** Потік данних
**Вихід:** drift_indices = [3062, 3147, 6719, ...]
**Параметри:**

- `theta = 0.5` (базовий поріг)
- `alpha = 1.5` (адаптивний множник)

**Результат:** 55 сирих детекцій

---

### 2️⃣ Drift Aggregator V2

**Що робить?** Групує 55 сирих детекцій в 8 смислових епізодів.

**Як?** В 3 кроків:

```
Крок 1: Фільтрація warm-up (перше K вікон - шумові)
Крок 2: Класифікація типів (sudden/incremental/gradual)
Крок 3: Merging близько розташованих детекцій в епізоди
```

**Вхід:** drift_indices
**Вихід:** merged_episodes = [Episode1, Episode2, ...]
**Параметри:**

- `merge_gap = 25` (максимальна дистанція щоб об'єднати)
- `cooldown = 20` (період стабілізації)

**Результат:** 8 чистих епізодів з типами + статистикою

---

### 3️⃣ Graphics & Reports

**Що робить?** Створює 4 PNG графіки та 1 HTML звіт з усіма статистиками.

**Вихід:**

```
delta_L_analysis.png         ← ΔL trend + threshold
similarity_trend.png         ← Similarity тренд
drift_distribution.png       ← Pie chart + type breakdown
synchronized_view.png        ← Dual-axis (ΔL + Similarity)
detailed_report.html         ← Statistics in HTML
```

**ВАЖЛИВО:** Усі графіки та HTML показують ОДИН і той же номер дрифтів (SSOT)

---

## Типи Дрифтів

Система класифікує дрифти на 4 типи:

| Тип             | Сигнал                  | Приклад             |
| --------------- | ----------------------- | ------------------- |
| **SUDDEN**      | Раптовий стрибок        | ![sudden-icon]      |
| **INCREMENTAL** | Монотонне зростання     | ![incremental-icon] |
| **GRADUAL**     | Хвилеподібна зміна      | ![gradual-icon]     |
| **UNKNOWN**     | Не підпадає під правила | ![unknown-icon]     |

---

## Основні Параметри

| Параметр      | 值  | Опис                                  |
| ------------- | --- | ------------------------------------- |
| `theta`       | 0.5 | Базовий поріг ΔL                      |
| `alpha`       | 1.5 | Адаптивний множник (σ для порогу)     |
| `window_size` | 50  | Розмір слідуючого вікна для FCA       |
| `merge_gap`   | 25  | Макс дистанція щоб об'єднати детекції |
| `cooldown`    | 20  | Період перед новим епізодом           |

---

## Основні Проблеми (Які Вирішені)

### Problem 1: FALSE POSITIVES (87.5%減少)

**Було:** Детектор флагував точки навіть з ΔL=0
**Причина:** Поріг падав до 0 під час warm-up
**Виправлено:** Додано мінімальний floor: `max(theta, μ + α·σ)`

### Problem 2: NUMBER DISCREPANCY

**Було:** HTML показав "6 дрифтів", графіки "20 дрифтів"
**Причина:** Різні джерела (merged vs raw)
**Виправлено:** Усе тепер використовує SSOT (merged_episodes)

### Problem 3: 85.9% UNKNOWN EPISODES

**Було:** Більшість епізодів були UNKNOWN
**Причина:** False positives з problem 1
**Виправлено:** Разом з problem 1

---

## SSOT (Single Source of Truth)

### Принцип

```
DriftAggregatorV2 → merged_episodes = [8 episodes]
    ├→ HTML Report показує: "8 drifts detected"
    └→ Graphics показують: "Merged: 8"
```

### Результат

✅ HTML === Graphics (sempre same numbers)

---

## Типична Сесія

```python
# 1. Initialize Detector
detector = FCADriftDetector(theta=0.5, alpha=1.5)

# 2. Process Stream (detector finds 55 raw drifts)
for window in sliding_windows:
    lattice = build_lattice(window)
    event = detector.update(lattice)
    if event:
        drift_indices.append(event.instance_id)

# 3. Aggregate into Episodes (8 final episodes)
agg = DriftAggregatorV2(
    drift_indices=drift_indices,
    delta_L=detector.delta_L_history,
    similarity=detector.similarity_history,
    merge_gap=25,
    cooldown=20
)
episodes = agg.get_merged_episodes()

# 4. Generate Reports
export_graphics(episodes)  # PNG files
export_html_report(episodes)  # HTML file

# Result: Both say "8 episodes" ✓
```

---

## Архітектура (30,000 ft view)

```
Stream Input
    ↓
FCA Detector (50-60 raw detections)
    ↓
DriftAggregatorV2 (5-10 merged episodes)
    ↓
┌─── Graphics ───┐  ┌─── HTML Report ───┐
│ 4 PNG files   │  │ Statistics HTML    │
│ All show: 8   │  │ All show: 8        │
└───────────────┘  └───────────────────┘
```

---

## Де Знайти Що

| Потрібно мені...         | Де шукати                                                                          |
| ------------------------ | ---------------------------------------------------------------------------------- |
| **Запустити систему**    | `main.py` або notebook                                                             |
| **Редагувати параметри** | `fca_detector.py` (theta, alpha) чи `drift_aggregator_v2.py` (merge_gap, cooldown) |
| **Розумієти детальніше** | [COMPLETE_SYSTEM_DOCUMENTATION.md](COMPLETE_SYSTEM_DOCUMENTATION.md)               |
| **Дебажити проблеми**    | Перевірте SSOT синхронізацію                                                       |
| **Приклади коду**        | Дивіться в [EXAMPLES.md](EXAMPLES.md)                                              |

---

## Кроки для Розуміння

1. **Розумієте проблему?** ✓ (сирі дрифти → епізоди)
2. **Розумієте компоненти?** ✓ (Detector → Aggregator → Reports)
3. **Розумієте параметри?** ✓ (theta, alpha, merge_gap, cooldown)
4. **Розумієте виправлення?** ✓ (3 основних bug fixes)

---

## Наступні Кроки

- 🔍 Дивіться [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) для більшого контексту
- 📖 Дивіться [COMPLETE_SYSTEM_DOCUMENTATION.md](COMPLETE_SYSTEM_DOCUMENTATION.md) для всіх деталей
- 💻 Дивіться код у `src/fca_drift/` для реалізації
- 🧪 Експериментуйте з параметрами та спостерігайте результати

---

**Єї! Тепер ви готові гулять систему 🚀**
