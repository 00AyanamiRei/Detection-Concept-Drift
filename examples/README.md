# Приклади використання

## Example 1: Базовий експеримент

Простий приклад детекції дрейфу на потоці Agrawal.

```bash
python examples/example1_basic.py
```

**Що він робить:**

1. Завантажує потік даних (Agrawal)
2. Створює вікно розміром 50
3. Обробляє дані з EMA + гістерезис бінаризацією
4. Будує концепт-гратку для кожного вікна
5. Виявляє дрейфи за допомогою FCA
6. Виводить результати

**Очікуваний вихід:**

```
==================================================================
Example 1: Basic FCA Drift Detection
==================================================================

1. Loading data stream (Agrawal)...
   OK
2. Creating sliding window (size=50)...
   OK
3. Creating data preprocessor...
   OK
4. Creating FCA drift detector...
   OK
5. Processing stream...

   [DRIFT 1] Instance 123: ΔL=0.6234 (sudden)
   [DRIFT 2] Instance 456: ΔL=0.5123 (gradual)

6. Results:
   Total instances processed: 500
   Drifts detected: 2
   Drift indices: [123, 456]
   Avg similarity: 0.8234
```

## Example 2: Grid Search для оптимальних параметрів

Пошук найкращих параметрів FCA детектора.

```bash
python examples/example2_grid_search.py
```

**Що він робить:**

1. Створює сітку параметрів
2. Запускає експеримент для кожної комбінації
3. Зберігає результати в CSV
4. Знаходить найкращі параметри

**Параметри для пошуку:**

- Window sizes: [50, 75]
- Thetas (threshold): [0.3, 0.4]
- Alphas (adaptive): [1.5, 2.0]
- **Всього:** 2 × 2 × 2 = 8 експериментів

**Результати** в `experiments/results/example_grid_search/`:

- `grid_search_results.csv` - таблиця всіх результатів
- `grid_search_results.json` - JSON формат

**Очікуваний CSV формат:**

```csv
dataset,window_size,theta,alpha,max_instances,precision,recall,f1_score,...
agrawal,50,0.3,1.5,1000,-1.0,-1.0,-1.0,...
agrawal,50,0.3,2.0,1000,-1.0,-1.0,-1.0,...
agrawal,50,0.4,1.5,1000,-1.0,-1.0,-1.0,...
```

## Example 3: Порівняння детекторів

Порівняння FCA, DDM та EDDM на одних даних.

```bash
python examples/example3_comparison.py
```

**Що він робить:**

1. Запускає всі детектори на одному потоці
2. Обробляє дані одночасно
3. Вирівнює результати
4. Генерує звіт порівняння

**Результати** в `experiments/results/example_comparison/`:

- `detector_comparison.json` - метрики всіх детекторів
- `detector_comparison_report.txt` - читаємий звіт

**Очікуваний звіт:**

```
======================================================================
DETECTOR COMPARISON REPORT
======================================================================

Dataset: agrawal

----------------------------------------------------------------------

FCA:
  Drifts Detected: 5

DDM:
  Drifts Detected: 3

EDDM:
  Drifts Detected: 4

======================================================================
```

## Запуск вручну через command line

### Базовий експеримент

```bash
python -m fca_drift \
  --dataset agrawal \
  --max-instances 1000 \
  --window-size 50 \
  --theta 0.4 \
  --alpha 2.0 \
  --output-dir experiments/results/my_test
```

### Елек2 датасет

```bash
python -m fca_drift \
  --dataset elec2 \
  --max-instances 5000 \
  --window-size 100 \
  --theta 0.35 \
  --output-dir experiments/results/elec2_test
```

## Перегляд результатів

Результати зберігаються як CSV файли, які можна відкрити в Excel або аналізувати:

```bash
# Переглянути дрейфи
cat experiments/results/my_test/drifts.csv

# Переглянути всі кроки
head -20 experiments/results/my_test/steps.csv

# Переглянути метрики
cat experiments/results/my_test/metrics.json
```

## Написання своєї аналізу

```python
import pandas as pd
import matplotlib.pyplot as plt

# Завантажити дані
drifts_df = pd.read_csv('experiments/results/my_test/drifts.csv')
steps_df = pd.read_csv('experiments/results/my_test/steps.csv')

# Статистика
print(f"Дрейфи виявлені: {len(drifts_df)}")
print(f"Середня впевненість: {drifts_df['confidence'].mean():.4f}")

# Графік
plt.figure(figsize=(12, 6))
plt.plot(steps_df['instance'], steps_df['delta_L'], label='Delta L')
plt.axhline(y=0.4, color='r', linestyle='--', label='Threshold')
plt.scatter(drifts_df['instance'], drifts_df['delta_L'], color='red', s=100)
plt.xlabel('Instance')
plt.ylabel('Delta L')
plt.legend()
plt.show()
```

## Налаштування параметрів

### DataPreprocessor

```python
from fca_drift.core import DataPreprocessor

# Більш агресивне згладжування
preprocessor = DataPreprocessor(
    ema_alpha=0.5,      # Вищий = більш чутливо до змін
    hysteresis=0.01,    # Менший = менше коливань
    threshold=0.5       # Точка поділу
)
```

### FCADriftDetector

```python
from fca_drift.detection import FCADriftDetector

# Більш чутливий
detector = FCADriftDetector(
    theta=0.3,          # Нижче = більше дрейфів
    alpha=1.5,          # Нижче = більше дрейфів
    window_size=50
)

# Менш чутливий
detector = FCADriftDetector(
    theta=0.5,          # Вище = менше дрейфів
    alpha=2.5,          # Вище = менше дрейфів
    window_size=100
)
```

---

**Усі приклади готові до запуску!** 🚀
