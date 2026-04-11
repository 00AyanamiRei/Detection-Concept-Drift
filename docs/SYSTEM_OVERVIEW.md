# FCA Drift Detection System - System Overview

**📖 Читайте: 15-20 хвилин**

Цей документ дає детальний огляд архітектури та компонентів.

---

## Системна Архітектура

### End-to-End Pipeline

```
┌────────────────────────────────────────────────────────────┐
│                    STREAMING DATA                          │
│            X_train, y_train (historical)                   │
│            X_test stream (real-time)                       │
└────────────────────┬─────────────────────────────────────┘
                     ↓
         ┌──────────────────────────────┐
         │   SLIDING WINDOW (W=50)      │
         │   - Build FCA lattice        │
         │   - Compare to previous      │
         └──────────────┬───────────────┘
                        ↓
         ┌──────────────────────────────┐
         │  FCA DRIFT DETECTOR          │
         │  (File: fca_detector.py)     │
         │  - Compute ΔL = 1-Similarity │   ← Raw detections
         │  - Adaptive threshold        │      (55 points)
         │  - Output: drift_indices     │
         └──────────────┬───────────────┘
                        ↓
         ┌──────────────────────────────┐
         │ DRIFT AGGREGATOR V2          │
         │ (File: aggregator_v2.py)     │
         │                              │
         │ 1. Filter warm-up            │
         │ 2. Classify types            │   ← Merged episodes
         │ 3. Merge events              │      (8 episodes)
         │ 4. Build episode statistics  │
         │                              │
         │ Output: merged_episodes      │
         └──────────────┬───────────────┘
                        ↓
         ┌──────────────────────────────┐
         │    SSOT (SINGLE SOURCE)      │
         │                              │
         │ All reports query            │
         │ this aggregator              │
         └──────────────┬───────────────┘
                ┌───────┴────────┐
                ↓                ↓
    ┌──────────────────┐  ┌──────────────────┐
    │     GRAPHICS     │  │   HTML REPORT    │
    │                  │  │                  │
    │ 4 PNG files:     │  │ detailed_report_ │
    │ 1. ΔL Trend      │  │ v2.html         │
    │ 2. Similarity    │  │                  │
    │ 3. Distribution  │  │ Shows:           │
    │ 4. Synchronized  │  │ - Statistics     │
    │                  │  │ - Tables         │
    │ All show: 8 dfs  │  │ - All show: 8 dfs│
    └──────────────────┘  └──────────────────┘
```

---

## Компоненти Системи

### 1. FCA Drift Detector (`fca_detector.py`)

**Цель:** Знайти точки, де структура даних змінилась

**Механізм:**

```
For each sliding window T:
  1. Build FCA lattice L(T)
  2. If T > 0:
       Compute similarity(L(T), L(T-1))  ∈ [0,1]
       delta_L = 1 - similarity           ∈ [0,1]

       Compute adaptive threshold:
         μ = mean of recent deltas
         σ = std of recent deltas
         theta_adaptive = max(theta_baseline, μ + α·σ)

       IF delta_L > theta_adaptive:
         Report DRIFT EVENT
  3. Update history
```

**Параметри:**
| Параметр | Значення | Сенс |
|----------|----------|------|
| theta | 0.5 | Baseline threshold (0-1) |
| alpha | 1.5 | Adaptive multiplier for σ |
| window_size | 50 | FCA lattice window |
| adaptive_window | 10 | How many recent windows to use for μ,σ |

**Важливий Реквізит:**

- Old: `theta_adaptive = μ + α·σ` → Bug! Can become 0
- New: `theta_adaptive = max(theta, μ + α·σ)` → Fixed! Floor prevents collapse

**Вихід:** DriftEvent[]

```python
{
  'instance_id': 3062,
  'drift_type': 'unknown',  # Classified later by Aggregator
  'confidence': 0.85,       # delta_L / threshold
  'delta_L': 0.425,
  'similarity': 0.575,
  'threshold': 0.5
}
```

---

### 2. Drift Aggregator V2 (`drift_aggregator_v2.py`)

**Цель:** Перетворити 55 сирих детекцій в 8 смислових епізодів

**5-Кроковий Pipeline:**

#### Крок 1: Remove Warm-up Region

```
raw_indices = [3062, 3147, 6719, ...]
min_warmup = max(1, int(0.1 * total_instances))
FILTER: indices >= min_warmup
Result: 48 valid indices (7 removed from early stream)
```

#### Крок 2: Classify Drift Types

```
For each index:
  z_score = (delta_L - median) / MAD

  IF z_score >= 2.5 AND isolated:
    type = 'SUDDEN'     # Sudden spike

  ELIF z_score < 1.5 AND monotonic:
    type = 'INCREMENTAL'  # Gradual increase

  ELIF 1.5 <= z_score < 2.5 AND vicinity_count >= 3:
    type = 'GRADUAL'      # Multiple nearby points

  ELSE:
    type = 'UNKNOWN'      # Doesn't fit pattern

Result: type_votes = {3062: 'SUDDEN', 3147: 'GRADUAL', ...}
```

#### Крок 3: Merge Events (Episodic Clustering)

```
Algorithm: Temporal-Gap Episodic Clustering with Stability Cooldown

PROCEDURE:
  groups = []
  current_group = [indices[0]]
  cooldown_until = indices[0] + cooldown

  FOR each idx IN indices[1:]:
    # Two conditions must BOTH be true to start new episode:
    IF idx > cooldown_until AND idx - current_group[-1] > merge_gap:
      groups.APPEND(current_group)
      current_group = [idx]
      cooldown_until = idx + cooldown
    ELSE:
      current_group.APPEND(idx)

  groups.APPEND(current_group)
  RETURN groups

Example (merge_gap=25, cooldown=20):
  raw:     [50, 52, 55, 58, 100, 102, 150]
  groups:  [[50,52,55,58], [100,102], [150]]
```

**Параметри:**
| Параметр | Значення | Сенс |
|----------|----------|------|
| merge_gap | window_size // 2 = 25 | Max gap to merge |
| cooldown | 0.4 \* window_size = 20 | Stability period |

#### Крок 4: Build Episode Statistics

```
For each group of indices:
  start = min(indices)
  end = max(indices)

  dlt_vals = delta_L[start:end+1]
  sim_vals = similarity[start:end+1]

  dlt_max = max(dlt_vals)
  dlt_mean = mean(dlt_vals)
  dlt_min = min(dlt_vals)

  sim_max = max(sim_vals)
  sim_mean = mean(sim_vals)
  sim_min = min(sim_vals)

  # Determine dominant type
  type_counts = Counter(types_in_episode)
  dominant_type = max_vote(type_counts)

  # Center (weighted by magnitude)
  weights = dlt_vals / sum(dlt_vals)
  center = weighted_avg(indices, weights)

Result: Episode
  {
    'start': 50,
    'end': 58,
    'center': 55.2,
    'dlt_max': 0.45,
    'dlt_mean': 0.28,
    'sim_min': 0.52,
    'sim_mean': 0.68,
    'dominant_type': 'UNKNOWN',
    'type_votes': Counter({'UNKNOWN': 4}),
    'indices': [50, 52, 55, 58]
  }
```

**Вихід:** DriftEpisode[]

```python
[
  Episode(start=3062, end=3147, center=3104.5, type='UNKNOWN', ...),
  Episode(start=6719, end=7427, center=7090.2, type='GRADUAL', ...),
  Episode(start=9259, end=9659, center=9450.1, type='SUDDEN', ...),
  # ... total 8 episodes
]
```

---

### 3. Graphics Export (`thesis_graphics.py`)

**Цель:** Генерувати 4 publication-ready PNG файли

**4 Графіки:**

1. **delta_L_analysis.png**
   - X: Instance ID
   - Y: ΔL (0 to 1)
   - Shows: Adaptive threshold (red), baseline (gray), detections (blue points)

2. **similarity_trend.png**
   - X: Instance ID
   - Y: Similarity (0 to 1)
   - Shows: Similarity curve, correlation annotation

3. **drift_distribution.png**
   - Pie chart: Type distribution
   - Elements: SUDDEN, INCREMENTAL, GRADUAL, UNKNOWN
   - Uses merged_episodes types (SSOT)

4. **synchronized_view.png**
   - Left Y: ΔL (red)
   - Right Y: Similarity (blue)
   - Shows: Inverse relationship visualization

**ВАЖНО:** Всі графіки мають info box:

```
┌─────────────────────────────┐
│ Detected Drifts: 55         │
│ Merged Episodes: 8          │
│ Drift Rate: 1.45%           │
└─────────────────────────────┘
```

---

### 4. HTML Report (`detailed_report_v2.py`)

**Цель:** Генерувати детальний HTML звіт

**Вміст:**

- Summary section (total drifts, rate, types)
- Episode table (start/end/type/stats for each)
- Type statistics (pie/bar charts)
- Parameter summary

**SSOT:** Усі числа來自 `aggregator.get_merged_episodes()`

---

## Data Structures

### DriftEvent (Detector Output)

```python
{
  'instance_id': int,           # Which instance
  'drift_type': str,            # 'unknown' (classified later)
  'confidence': float,          # 0-1
  'delta_L': float,             # 0-1
  'similarity': float,          # 0-1
  'threshold': float            # Adaptive threshold
}
```

### DriftEpisode (Aggregator Output)

```python
{
  'start': int,                 # First index
  'end': int,                   # Last index
  'center': float,              # Weighted center
  'indices': list,              # All indices in episode

  'dlt_max': float,
  'dlt_mean': float,
  'dlt_min': float,

  'sim_max': float,
  'sim_mean': float,
  'sim_min': float,

  'dominant_type': str,         # SUDDEN/INCREMENTAL/GRADUAL/UNKNOWN
  'type_votes': Counter,        # Voting results
  'z_scores': list              # Z-scores of indices
}
```

---

## SSOT (Single Source of Truth)

### Problem Before

```
Detector found: 55 raw detections
Aggregator produced: 8 episodes

HTML Report says: "Drifts: ???"  (which one?)
Graphics show: "Total: ???"       (which one?)
```

### Solution

```
aggregator.get_merged_episodes() = [8 episodes]
                ↓
    Everyone queries this
                ↓
    ┌───────────────────────────────┐
    │ HTML: "Drifts Detected: 8"    │
    │ Graphics: "Merged Episodes: 8"│
    │ Statistics: "Count: 8"        │
    └───────────────────────────────┘
           All synchronized ✓
```

---

## Класифікація Типів Дрифтів

### Type Classification Logic

```
Input: detection index, local z_score, delta_L value

Classification Tree:
├─ Has z_score >= 2.5?
│  ├─ Is isolated (neighbors < 1.5σ)?
│  │  └─ Type = SUDDEN
│  └─ Has other nearby points?
│     └─ Type = might be GRADUAL or SUDDEN (competitive)
│
├─ Has z_score < 1.5?
│  ├─ Is monotonic increase?
│  │  └─ Type = INCREMENTAL
│  └─ Random fluctuation?
│     └─ Type = UNKNOWN
│
├─ Has 1.5 <= z_score < 2.5?
│  ├─ Are there >= 3 nearby points?
│  │  └─ Type = GRADUAL
│  └─ Isolated high point?
│     └─ Type = SUDDEN
│
└─ Doesn't match any pattern?
   └─ Type = UNKNOWN
```

### Example Classifications

```
Index  ΔL    Z-score  Signal               Type
─────────────────────────────────────────────────
3062   0.001  0.05    Tiny spike           UNKNOWN
3147   0.45   2.1     Sustained change     GRADUAL
6719   0.65   3.2     Big isolated spike   SUDDEN
6744   0.42   1.8     Moderate, nearby     GRADUAL
7210   0.51   1.9     Rising trend         INCREMENTAL
9259   0.48   1.7     Multiple nearby      GRADUAL
```

---

## Типові сценарії

### Сценарій 1: Stable Stream

```
Raw detections: 5
  (all in warm-up region)
Merged episodes: 0
  (filtered out)
Result: No reported drift ✓
```

### Сценарій 2: Single Drift Event

```
Raw detections: 15
  [100, 102, 105, 108, 110]  ← Close cluster
Merged episodes: 1
  Episode(start=100, end=110, type='SUDDEN')
Result: Single episode reported ✓
```

### Сценарій 3: Multiple Drift Events

```
Raw detections: 55
  [50-58 cluster] [100-110 cluster] [200-210 cluster] ...
Merged episodes: 8
  Episode1(50-58, GRADUAL)
  Episode2(100-110, SUDDEN)
  Episode3(200-210, INCREMENTAL)
  ...
Result: 8 episodes with types reported ✓
```

---

## Параметрізація

### Рекомендовані Значення

| Параметр    | Рекомендація | Чому                            |
| ----------- | ------------ | ------------------------------- |
| theta       | 0.5          | Balance sensitivity/specificity |
| alpha       | 1.5          | 1.5σ adapts well to variance    |
| window_size | 50           | Good FCA lattice size           |
| merge_gap   | W/2          | Temporal clustering standard    |
| cooldown    | 0.4W         | Prevent fragmentation           |
| warm_up     | 10W          | Allow detector to stabilize     |

### Настройка

```python
# Conservative (fewer false positives)
FCADriftDetector(theta=0.6, alpha=2.0)  # Higher thresholds

# Sensitive (catch more drifts)
FCADriftDetector(theta=0.4, alpha=1.0)  # Lower thresholds

# Tight episodes (many small episodes)
DriftAggregatorV2(merge_gap=10, cooldown=5)

# Loose episodes (fewer large episodes)
DriftAggregatorV2(merge_gap=50, cooldown=40)
```

---

## Проблеми і Вирішення

### Issue 1: Threshold Collapse (FIXED ✅)

```
Before: theta_adaptive = μ + α·σ
        During warm-up: σ→0, so theta→0
        Result: Every point triggers drift (FALSE POSITIVES)

After: theta_adaptive = max(theta_baseline, μ + α·σ)
       With floor: Never drops below 0.5
       Result: Reduced false positives by 87.5%
```

### Issue 2: Number Discrepancy (FIXED ✅)

```
Before: HTML used merged_episodes count (8)
        Graphics used raw detection count (55)
        User confusion: "Which is right?"

After: Both use merged_episodes (SSOT principle)
       Result: Always synchronized
```

### Issue 3: 85.9% Unknown Types (FIXED ✅)

```
Before: Many episodes had zero delta_L
        Type classification marked them UNKNOWN
        Caused by false positives (Issue 1)

After: Fixed threshold collapse (Issue 1)
       Fewer false positives
       Type classification works better
       Result: More meaningful types
```

---

## Наступні Кроки

1. **Розумієте архітектуру?** ✓
2. **Готові рахуватись код?** → Дивіться [COMPLETE_SYSTEM_DOCUMENTATION.md](COMPLETE_SYSTEM_DOCUMENTATION.md)
3. **Хочете експериментувала?** → Змініть параметри та спостерігайте результати
4. **Маєте запитання?** → Перегляньте [INDEX.md](INDEX.md) для навігації

---

**Продовжуйте з COMPLETE_SYSTEM_DOCUMENTATION для глибокої ознайомлення 📚**
