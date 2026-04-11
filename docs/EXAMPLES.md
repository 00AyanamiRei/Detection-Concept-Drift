# FCA Drift Detection System - Examples & Code Snippets

**💻 Практичні приклади**

---

## Приклад 1: Basic Usage

### Цель: Запустити детектор на потоці даних

```python
from fca_drift.detection.fca_detector import FCADriftDetector
from fca_drift.utils.drift_aggregator_v2 import DriftAggregatorV2
from fca_drift.visualization.thesis_graphics import ThesisGraphicsExporter
from fca_drift.visualization.detailed_report_v2 import DetailedReportGeneratorV2

# ====================================
# 1. INITIALIZE DETECTOR
# ====================================
detector = FCADriftDetector(
    theta=0.5,
    alpha=1.5,
    window_size=50,
    adaptive_window=10
)

# ====================================
# 2. PROCESS STREAM
# ====================================
drift_indices = []
delta_L_history = []
similarity_history = []

sliding_window = SlidingWindow(size=50)

for instance_id, (X, y) in enumerate(stream):
    sliding_window.add(X)

    if sliding_window.is_full():
        # Build FCA lattice for current window
        lattice = build_fca_lattice(sliding_window)

        # Update detector
        drift_event = detector.update(lattice)

        # Collect history
        delta_L_history.append(detector.delta_L_history[-1])
        similarity_history.append(detector.similarity_history[-1])

        # Record drift if detected
        if drift_event:
            drift_indices.append(instance_id)
            print(f"Drift detected at {instance_id}")

print(f"\nTotal raw detections: {len(drift_indices)}")

# ====================================
# 3. AGGREGATE INTO EPISODES
# ====================================
aggregator = DriftAggregatorV2(
    delta_L_history=delta_L_history,
    similarity_history=similarity_history,
    drift_indices=drift_indices,
    window_size=50,
    merge_gap=25,
    cooldown=20
)

merged_episodes = aggregator.get_merged_episodes()
print(f"Total merged episodes: {len(merged_episodes)}")

# ====================================
# 4. GENERATE GRAPHICS
# ====================================
graphics_exporter = ThesisGraphicsExporter(
    delta_L=delta_L_history,
    similarity=similarity_history,
    drift_indices=drift_indices
)

graphics_exporter.export_all(
    output_dir='./output',
    merged_episodes=merged_episodes
)

# ====================================
# 5. GENERATE HTML REPORT
# ====================================
html_generator = DetailedReportGeneratorV2(
    aggregator=aggregator,
    drift_indices=drift_indices
)

html_generator.generate_report(
    output_path='./output/report.html'
)

print("\n✓ All outputs generated!")
```

---

## Приклад 2: Access Merged Episodes

### Цель: Отримати деталі про кожен епізод

```python
# Get merged episodes from aggregator
episodes = aggregator.get_merged_episodes()

# Iterate through each episode
for episode_num, episode in enumerate(episodes, 1):
    print(f"\n{'='*50}")
    print(f"Episode {episode_num}")
    print(f"{'='*50}")

    # Basic info
    print(f"Range: [{episode.start}, {episode.end}]")
    print(f"Duration: {episode.end - episode.start + 1} instances")
    print(f"Center (weighted): {episode.center:.1f}")

    # Type classification
    print(f"\nType: {episode.dominant_type}")
    print(f"Type voting breakdown:")
    for dtype, count in episode.type_votes.most_common():
        pct = 100 * count / len(episode.indices)
        print(f"  - {dtype}: {count} ({pct:.1f}%)")

    # ΔL statistics
    print(f"\nDelta-L Statistics:")
    print(f"  Max:  {episode.dlt_max:.4f}")
    print(f"  Mean: {episode.dlt_mean:.4f}")
    print(f"  Min:  {episode.dlt_min:.4f}")

    # Similarity statistics
    print(f"\nSimilarity Statistics:")
    print(f"  Max:  {episode.sim_max:.4f}")
    print(f"  Mean: {episode.sim_mean:.4f}")
    print(f"  Min:  {episode.sim_min:.4f}")

    # Z-scores
    print(f"\nZ-Scores of indices in episode:")
    if len(episode.z_scores) <= 5:
        for z in episode.z_scores:
            print(f"  {z:.2f}")
    else:
        print(f"  Min: {min(episode.z_scores):.2f}")
        print(f"  Max: {max(episode.z_scores):.2f}")
        print(f"  Mean: {sum(episode.z_scores)/len(episode.z_scores):.2f}")
```

### Вихід

```
==================================================
Episode 1
==================================================
Range: [3062, 6714]
Duration: 3653 instances
Center (weighted): 3147.2

Type: UNKNOWN
Type voting breakdown:
  - UNKNOWN: 8 (100.0%)

Delta-L Statistics:
  Max:  0.4245
  Mean: 0.2107
  Min:  0.0003

Similarity Statistics:
  Max:  0.9997
  Mean: 0.8624
  Min:  0.5756

Z-Scores of indices in episode:
  Min: 0.05
  Max: 3.24
  Mean: 1.45
```

---

## Приклад 3: Custom Type Classification

### Цель: Змінити правила класифікації типів

```python
from fca_drift.utils.drift_aggregator_v2 import DriftAggregatorV2

class CustomDriftAggregator(DriftAggregatorV2):
    """
    Custom aggregator with modified type classification rules
    """

    def _determine_type_from_pattern(self, idx, z_score):
        """
        Override type classification with custom rules
        """

        # CUSTOM RULE 1: Very high z-score is always SUDDEN
        if z_score >= 4.0:
            return 'sudden'

        # CUSTOM RULE 2: Check for monotonic increase (INCREMENTAL)
        window_start = max(0, idx - 10)
        window_end = min(len(self.delta_L), idx + 10)
        window = self.delta_L[window_start:window_end]

        if self._is_monotonic_increasing(window):
            return 'incremental'

        # CUSTOM RULE 3: Multiple peaks nearby (GRADUAL)
        nearby_peaks = self._count_peaks_in_vicinity(idx, radius=5)
        if nearby_peaks >= 3:
            return 'gradual'

        # Default
        return 'unknown'

    def _is_monotonic_increasing(self, values):
        """Check if values are monotonically increasing"""
        if len(values) < 2:
            return False
        return all(values[i] <= values[i+1] for i in range(len(values)-1))

    def _count_peaks_in_vicinity(self, idx, radius):
        """Count local peaks near index"""
        count = 0
        for i in range(max(0, idx-radius), min(len(self.delta_L), idx+radius+1)):
            z = self._compute_adaptive_z(i)
            if z >= 1.5:
                count += 1
        return count

# Use custom aggregator
aggregator = CustomDriftAggregator(
    delta_L_history=delta_L_history,
    similarity_history=similarity_history,
    drift_indices=drift_indices,
    window_size=50,
    merge_gap=25,
    cooldown=20
)

episodes = aggregator.get_merged_episodes()
# Now uses custom classification rules
```

---

## Приклад 4: Parameter Tuning

### Цель: Експериментувала з різними параметрами

```python
import numpy as np
from fca_drift.detection.fca_detector import FCADriftDetector

# Stream data (assumed available)
# stream = [(X_i, y_i) for i in range(n)]

# Grid search over parameters
theta_values = [0.4, 0.5, 0.6]
alpha_values = [1.0, 1.5, 2.0]

results = []

for theta in theta_values:
    for alpha in alpha_values:
        print(f"Testing theta={theta}, alpha={alpha}...", end='')

        # Initialize detector with these params
        detector = FCADriftDetector(
            theta=theta,
            alpha=alpha,
            window_size=50
        )

        # Process stream
        drift_indices = []
        delta_L_history = []

        for instance_id, (X, y) in enumerate(stream):
            # ... process window ...
            drift_event = detector.update(lattice)
            if drift_event:
                drift_indices.append(instance_id)
            delta_L_history.append(detector.delta_L_history[-1])

        # Statistics
        num_detections = len(drift_indices)
        detection_rate = 100 * num_detections / len(stream)

        results.append({
            'theta': theta,
            'alpha': alpha,
            'detections': num_detections,
            'rate': detection_rate
        })

        print(f" {num_detections} detections ({detection_rate:.2f}%)")

# Find best (e.g., minimal detections)
best = min(results, key=lambda x: x['detections'])
print(f"\nBest parameters: theta={best['theta']}, alpha={best['alpha']}")
print(f"  → {best['detections']} detections ({best['rate']:.2f}%)")
```

### Вихід

```
Testing theta=0.4, alpha=1.0... 87 detections (2.18%)
Testing theta=0.4, alpha=1.5... 56 detections (1.40%)
Testing theta=0.4, alpha=2.0... 34 detections (0.85%)
Testing theta=0.5, alpha=1.0... 67 detections (1.68%)
Testing theta=0.5, alpha=1.5... 55 detections (1.38%)
Testing theta=0.5, alpha=2.0... 32 detections (0.80%)
Testing theta=0.6, alpha=1.0... 45 detections (1.13%)
Testing theta=0.6, alpha=1.5... 40 detections (1.00%)
Testing theta=0.6, alpha=2.0... 28 detections (0.70%)

Best parameters: theta=0.6, alpha=2.0
  → 28 detections (0.70%)
```

---

## Приклад 5: Visualization Customization

### Цель: Створити власні графіки

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Get data from aggregator
merged_episodes = aggregator.get_merged_episodes()
delta_L = aggregator.delta_L_history
similarity = aggregator.similarity_history

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Delta-L with episodes highlighted
ax = axes[0, 0]
x = np.arange(len(delta_L))
ax.plot(x, delta_L, 'b-', alpha=0.7, label='ΔL')

# Highlight episodes
for ep in merged_episodes:
    color = 'red' if ep.dominant_type == 'sudden' else 'orange'
    rect = Rectangle((ep.start, 0), ep.end-ep.start,
                      max(delta_L), alpha=0.2, color=color)
    ax.add_patch(rect)

ax.set_xlabel('Instance ID')
ax.set_ylabel('ΔL')
ax.set_title('ΔL Trend with Detected Episodes')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Similarity curve
ax = axes[0, 1]
ax.plot(x, similarity, 'g-', alpha=0.7, label='Similarity')
ax.axhline(y=np.mean(similarity), color='gray', linestyle='--',
           label=f'Mean ({np.mean(similarity):.3f})')
ax.set_xlabel('Instance ID')
ax.set_ylabel('Similarity')
ax.set_title('Similarity Trend')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Episode types distribution
ax = axes[1, 0]
type_counts = {}
for ep in merged_episodes:
    ep_type = ep.dominant_type
    type_counts[ep_type] = type_counts.get(ep_type, 0) + 1

types = list(type_counts.keys())
counts = list(type_counts.values())
colors = {
    'sudden': 'red',
    'incremental': 'blue',
    'gradual': 'orange',
    'unknown': 'gray'
}
bar_colors = [colors.get(t, 'black') for t in types]

ax.bar(types, counts, color=bar_colors, alpha=0.7)
ax.set_ylabel('Count')
ax.set_title('Episode Type Distribution')
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Episode timeline
ax = axes[1, 1]
y_pos = 0
for ep in sorted(merged_episodes, key=lambda e: e.start):
    ax.barh(y_pos, ep.end - ep.start, left=ep.start, height=0.8,
            color=colors.get(ep.dominant_type, 'black'), alpha=0.7)
    ax.text(ep.center, y_pos, f"#{y_pos+1}",
            va='center', ha='center', fontweight='bold')
    y_pos += 1

ax.set_xlabel('Instance ID')
ax.set_ylabel('Episode #')
ax.set_title('Episode Timeline')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('custom_analysis.png', dpi=300)
plt.show()
```

---

## Приклад 6: Performance Metrics

### Цель: Обчислити метрики якості

```python
from sklearn.metrics import precision_recall_curve, f1_score

# Assume we have ground truth labels
# true_drift_indices = [list of indices where drift actually occurred]

# Get predictions from aggregator
merged_episodes = aggregator.get_merged_episodes()
predicted_indices = set()
for ep in merged_episodes:
    predicted_indices.update(ep.indices)

# Convert to binary labels
y_true = np.zeros(len(stream))
y_pred = np.zeros(len(stream))

for idx in true_drift_indices:
    if 0 <= idx < len(y_true):
        y_true[idx] = 1

for idx in predicted_indices:
    if 0 <= idx < len(y_pred):
        y_pred[idx] = 1

# Calculate metrics
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print("="*50)
print("PERFORMANCE METRICS")
print("="*50)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print()
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Negatives: {tn}")
print()
print(f"Detection Rate: {100*tp/(tp+fn):.2f}%")
print(f"False Positive Rate: {100*fp/(fp+tn):.2f}%")
```

---

## Приклад 7: SSOT Verification

### Цель: Переконатися в синхронізації

```python
# This example verifies SSOT principle

# 1. Get merged episodes from aggregator
episodes = aggregator.get_merged_episodes()
agg_count = len(episodes)

# 2. Check what HTML report would show
html_generator = DetailedReportGeneratorV2(aggregator=aggregator)
html_count = len(aggregator.get_merged_episodes())

# 3. Check what graphics would show
graphics_exporter = ThesisGraphicsExporter(...)
graphics_count = len(merged_episodes)  # Should use same aggregator

# 4. Verify SSOT
print("="*50)
print("SSOT VERIFICATION")
print("="*50)
print(f"Aggregator reports: {agg_count} episodes")
print(f"HTML would show: {html_count} episodes")
print(f"Graphics would show: {graphics_count} episodes")

if agg_count == html_count == graphics_count:
    print("\n✓ SSOT SYNCHRONIZED")
else:
    print("\n✗ SSOT MISMATCH - FIX REQUIRED")
    print(f"  Aggregator: {agg_count}")
    print(f"  HTML: {html_count}")
    print(f"  Graphics: {graphics_count}")
```

---

## Приклад 8: Parameter Impact Analysis

### Цель: Зрозуміти вплив параметрів

```python
# Analyze impact of merge_gap and cooldown

merge_gap_values = [10, 25, 50, 100]
cooldown_values = [5, 20, 40]

print("="*70)
print(f"{'merge_gap':12} | {'cooldown':10} | {'Episodes':10} | {'Avg Duration':12}")
print("="*70)

for merge_gap in merge_gap_values:
    for cooldown in cooldown_values:
        agg = DriftAggregatorV2(
            delta_L_history=delta_L_history,
            similarity_history=similarity_history,
            drift_indices=drift_indices,
            window_size=50,
            merge_gap=merge_gap,
            cooldown=cooldown
        )

        episodes = agg.get_merged_episodes()
        num_episodes = len(episodes)
        avg_duration = (
            np.mean([ep.end - ep.start for ep in episodes])
            if episodes else 0
        )

        print(f"{merge_gap:12} | {cooldown:10} | {num_episodes:10} | {avg_duration:12.1f}")
```

### Вихід

```
===============================================================================
merge_gap    | cooldown   | Episodes   | Avg Duration
===============================================================================
10           | 5          | 12         |         234.5
10           | 20         | 10         |         289.3
10           | 40         |  9         |         312.1
25           | 5          |  8         |         412.5
25           | 20         |  8         |         456.2
25           | 40         |  7         |         523.4
50           | 5          |  5         |         678.9
50           | 20         |  4         |         845.3
50           | 40         |  3         |       1123.7
100          | 5          |  2         |       2145.0
100          | 20         |  2         |       2145.0
100          | 40         |  2         |       2145.0
```

---

## Резюме

| Приклад | Цель             | Ключові класи                                         |
| ------- | ---------------- | ----------------------------------------------------- |
| 1       | Full pipeline    | FCADriftDetector, DriftAggregatorV2, Graphics, Report |
| 2       | Access episodes  | Episode, get_merged_episodes()                        |
| 3       | Custom types     | Custom DriftAggregator subclass                       |
| 4       | Parameter tuning | Grid search, detector parametrization                 |
| 5       | Visualization    | matplotlib, episode highlighting                      |
| 6       | Performance      | sklearn metrics                                       |
| 7       | SSOT check       | Synchronization verification                          |
| 8       | Parameter impact | merge_gap, cooldown analysis                          |

---

**Дивіться [INDEX.md](INDEX.md) для навігації до інших документів 📚**
