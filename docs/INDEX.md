# FCA Drift Detection System - Navigation Index

**Вітаємо в документації системи виявлення дрифтів!**

Цей файл допоможе вам швидко знайти потрібну інформацію.

---

## 🚀 Почніть звідси

- **Новачок в системі?** → [QUICK_START.md](QUICK_START.md)
- **Потрібно швидко зрозуміти як це працює?** → [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)
- **Хочу весь контекст?** → [COMPLETE_SYSTEM_DOCUMENTATION.md](COMPLETE_SYSTEM_DOCUMENTATION.md)

---

## 📚 Навігація за темами

### Основні Компоненти

| Компонента              | Опис                                            | Де читати                                                     |
| ----------------------- | ----------------------------------------------- | ------------------------------------------------------------- |
| **FCA Detector**        | Виявляє raw drift точки через аналіз решіток    | [COMPLETE_SYSTEM_DOCUMENTATION.md#fca-drift-detector)         |
| **Drift Aggregator V2** | Групує детекції в смислові епізоди (SSOT)       | [COMPLETE_SYSTEM_DOCUMENTATION.md#drift-aggregator-v2)        |
| **Episodic Clustering** | Алгоритм merging на основі часових розривів     | [COMPLETE_SYSTEM_DOCUMENTATION.md#episodic-clustering)        |
| **Type Classification** | Визначає тип дрифту: sudden/incremental/gradual | [COMPLETE_SYSTEM_DOCUMENTATION.md#класифікація-типів-дрифтів) |
| **Graphics Export**     | 4 publication-quality графіки                   | [COMPLETE_SYSTEM_DOCUMENTATION.md#архітектура-системи)        |
| **HTML Reports**        | Детальні звіти в HTML форматі                   | [COMPLETE_SYSTEM_DOCUMENTATION.md#архітектура-системи)        |

### Архітектура і Дизайн

| Тема                    | Опис                                         | Де читати                                                      |
| ----------------------- | -------------------------------------------- | -------------------------------------------------------------- |
| **System Architecture** | Загальний огляд data flow                    | [COMPLETE_SYSTEM_DOCUMENTATION.md#архітектура-системи)         |
| **SSOT Principle**      | Single Source of Truth - синхронізація чисел | [COMPLETE_SYSTEM_DOCUMENTATION.md#ssot-single-source-of-truth) |
| **Parameters**          | Всі параметри системи і їх значення          | [COMPLETE_SYSTEM_DOCUMENTATION.md#параметри)                   |

### Алгоритми

| Алгоритм                             | Застосування     | Опис                                        | Де читати                                                            |
| ------------------------------------ | ---------------- | ------------------------------------------- | -------------------------------------------------------------------- |
| **Adaptive Thresholding**            | FCA Detector     | μ + α·σ з мінімальним floor                 | [COMPLETE_SYSTEM_DOCUMENTATION.md#параметри)                         |
| **Temporal-Gap Episodic Clustering** | Merging          | Групування за часовими розривами + cooldown | [COMPLETE_SYSTEM_DOCUMENTATION.md#алгоритм-merging)                  |
| **Z-Score Classification**           | Type Detection   | Адаптивна Z-score для визн. типу            | [COMPLETE_SYSTEM_DOCUMENTATION.md#z-score-computation)               |
| **Majority Voting**                  | Episode Building | Голосування для типу епізоду                | [COMPLETE_SYSTEM_DOCUMENTATION.md#етап-3-merging--episodic-building) |

### Проблеми і Виправлення

| Проблема                   | Статус        | Де читати                                                                        |
| -------------------------- | ------------- | -------------------------------------------------------------------------------- |
| **Threshold Collapse**     | ✅ Виправлено | [COMPLETE_SYSTEM_DOCUMENTATION.md#problem-1-threshold-collapse)                  |
| **Number Discrepancy**     | ✅ Виправлено | [COMPLETE_SYSTEM_DOCUMENTATION.md#problem-2-number-discrepancy-html-vs-graphics) |
| **85.9% Unknown Episodes** | ✅ Виправлено | [COMPLETE_SYSTEM_DOCUMENTATION.md#problem-3-859-episodes-classified-as-unknown)  |

---

## 🔍 Пошукайте по ключовому слові

### По алгоритмам

- **Як виявляються дрифти?** → Шукайте "FCA Detector"
- **Як дрифти групуються?** → Шукайте "Episodic Clustering"
- **Як визначається тип?** → Шукайте "Classification Rules"

### По проблемам

- **Чому так багато UNKNOWN?** → "Problem 3: 85.9%"
- **Чому HTML ≠ Graphics?** → "Problem 2: Number Discrepancy"
- **Чому false positives?** → "Problem 1: Threshold Collapse"

### По аспектам

- **Параметри настройки** → "parameters" або "alpha", "theta"
- **SSOT синхронізація** → "Single Source of Truth"
- **Наукові основи** → "scientific", "UIClust", "Gama et al"

---

## 📊 Файли системи

```
src/fca_drift/
├── detection/
│   └── fca_detector.py           ← Raw drift detection (FCADriftDetector)
├── utils/
│   └── drift_aggregator_v2.py    ← Episode aggregation (DriftAggregatorV2)
├── visualization/
│   ├── thesis_graphics.py        ← PNG graphs generation
│   └── detailed_report_v2.py     ← HTML report generation
└── fca/
    └── concept_lattice.py        ← FCA lattice operations

docs/ (THIS FOLDER)
├── INDEX.md                      ← You are here
├── QUICK_START.md                ← Getting started
├── SYSTEM_OVERVIEW.md            ← High-level summary
├── COMPLETE_SYSTEM_DOCUMENTATION.md  ← Full details
└── EXAMPLES.md                   ← Code examples
```

---

## 🎯 Типові задачі

### "Я хочу розуміти систему комплексно"

1. Прочитайте [QUICK_START.md](QUICK_START.md) (5 хв)
2. Прочитайте [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) (15 хв)
3. Заглибніться в [COMPLETE_SYSTEM_DOCUMENTATION.md](COMPLETE_SYSTEM_DOCUMENTATION.md) (60 хв)

### "Мені потрібно запустити систему"

1. Дивіться [QUICK_START.md](QUICK_START.md) - секцію "Installation"
2. Дивіться приклади в розділі "Examples"

### "Я отримую генерально як це працює, але хочу зрозуміти детально"

1. Прочитайте про конкретний компонент в [COMPLETE_SYSTEM_DOCUMENTATION.md](COMPLETE_SYSTEM_DOCUMENTATION.md)
2. Перегляньте код в `src/fca_drift/...`
3. Експериментуйте з параметрами

### "Щось не працює - як дебажити?"

1. Перевірте [COMPLETE_SYSTEM_DOCUMENTATION.md#виявлені-проблеми-і-виправлення)
2. Перегляньте SSOT синхронізацію: HTML === Graphics?
3. Перевірте параметри: theta, alpha, merge_gap, cooldown

---

## 🔗 Перехресні посилання

### FCA Detector ↔ DriftAggregatorV2

- Detector генерує `drift_indices` → Aggregator приймає їх
- Detector обчислює `delta_L_history` → Aggregator використовує для Z-score

### DriftAggregatorV2 ↔ Graphics

- Aggregator експортує `merged_episodes` → Graphics використовує для відображення
- SSOT: обидва показують один і той же `len(merged_episodes)`

### Graphics ↔ HTML Report

- Обидва запитують у Aggregator: `get_merged_episodes()`
- Результат: синхронізовані числа

---

## 📈 Версій та Історія

| Версія | Дата           | Статус        | Основні зміни                        |
| ------ | -------------- | ------------- | ------------------------------------ |
| 1.0    | March 15, 2026 | ✅ Production | FCA detector + aggregator + graphics |
| 0.9    | March 14, 2026 | 🐛 Bug fix    | SSOT synchronization added           |
| 0.8    | March 13, 2026 | 🐛 Bug fix    | Threshold collapse fixed             |

---

## 💡 Поради

- **Читати послідовно:** INDEX → QUICK_START → OVERVIEW → COMPLETE
- **Отримайте контекст:** Розуміння arquitecturi важливіше за детальні параметри
- **Експериментуйте:** Спробуйте різні alpha, theta значення
- **Дебажте SSOT:** Якщо числа не сходяться, перевірте synchronization

---

## ❓ Часті запитання

**Q: Якою мовою документація?**
A: Українська та Англійська (в коментарях коду)

**Q: Де знайти приклади?**
A: [EXAMPLES.md](EXAMPLES.md) або в тестах

**Q: Як настройити параметри?**
A: Дивіться "Parameters" секцію в COMPLETE_SYSTEM_DOCUMENTATION

**Q: Що означає SSOT?**
A: Single Source of Truth - принцип, коли всі звіти беруть дані з одного місця

---

**Приємического чтения! 🚀**
