# Drift Detection System - File Dependencies Map

**Documentation Date**: March 27, 2026

---

## 🎯 Quick Overview

### **Core Files for Drift Detection** (ESSENTIAL):

1. **main.py** (22.7 KB) - Entry point, orchestration
2. **src/fca_drift/detection/fca_detector.py** (27.6 KB) - Main detector logic ⭐
3. **src/fca_drift/fca/** - Formal Context Analysis
4. **src/fca_drift/core/** - Data handling

### **Total Project Files**: 40+ Python files

---

## 📊 Data Flow Architecture

```
main.py (Entry Point)
    ↓
[1. DATA LOADING] core/stream_reader.py → Dataset (agrawal, hyperplane, etc.)
    ↓
[2. WINDOWING] core/window_manager.py → SlidingWindow(50)
    ↓
[3. PREPROCESSING] core/preprocessing.py → Binary features
    ↓
[4. FCA]
    ├─ fca/context_builder.py → FormalContext
    └─ fca/lattice_builder.py → ConceptLattice
    ↓
[5. DETECTION] detection/fca_detector.py → Drift detected? ⭐
    ↓
[6. AGGREGATION] utils/drift_aggregator_v2.py → Merge episodes (SSOT)
    ↓
[7. REPORTING & VISUALIZATION]
    ├─ utils/drift_classifier.py
    ├─ visualization/detailed_report_v2.py
    ├─ visualization/thesis_graphics.py
    └─ visualization/lattice_visualizer.py
    ↓
[OUTPUT] experiments/results/[dataset]_w[size]_t[theta]_a[alpha]/
    ├─ report_en.html
    ├─ results.json
    ├─ *_thesis.png (4 graphs)
    └─ report_debug.json
```

---

## 📁 File Organization by Function

### **[1] DATA LOADING & PREPARATION**

| File                     | Purpose                         | Key Classes/Functions           |
| ------------------------ | ------------------------------- | ------------------------------- |
| `main.py`                | Main orchestration              | `DriftDetectionRunner`          |
| `core/stream_reader.py`  | Load datasets (river/synthetic) | `StreamReader`, `StreamWrapper` |
| `core/window_manager.py` | Sliding window management       | `SlidingWindow`                 |
| `core/preprocessing.py`  | Feature preprocessing           | `DataPreprocessor`              |

**Execution Order**: stream_reader → window_manager → preprocessing

---

### **[2] CORE DRIFT DETECTION** ⭐⭐⭐

| File                       | Purpose                | Status      |
| -------------------------- | ---------------------- | ----------- |
| **fca_detector.py**        | Main drift detector    | ✅ CRITICAL |
| **fca/context_builder.py** | Create formal contexts | Required    |
| **fca/lattice_builder.py** | Build concept lattices | Required    |
| **fca/similarity.py**      | Compare lattices       | Required    |

**Key Function**: `FCADriftDetector._classify_drift_type()`

- Detects: sudden, gradual, incremental, recurring

---

### **[3] POST-DETECTION PROCESSING**

| File                                 | Purpose                               |
| ------------------------------------ | ------------------------------------- |
| `utils/drift_aggregator_v2.py`       | Merge raw drifts into episodes (SSOT) |
| `utils/drift_classifier.py`          | Classify episodes by type             |
| `history/lattice_history_manager.py` | Track lattice evolution               |

---

### **[4] REPORTING & VISUALIZATION** (Optional)

| File                                         | Purpose                  |
| -------------------------------------------- | ------------------------ |
| `visualization/detailed_report_v2.py`        | HTML report generation   |
| `visualization/thesis_graphics.py`           | Publication-ready graphs |
| `visualization/lattice_visualizer.py`        | Lattice PNG rendering    |
| `visualization/lattice_integration_patch.py` | Lattice UI integration   |
| `visualization/plots.py`                     | Matplotlib plots         |

**Note**: These run AFTER detection completes

---

## 🔍 Detailed Module Breakdown

### **CORE MODULES (src/fca_drift/core/)**

```
core/
├── __init__.py (335 B)
│   Exports: StreamReader, StreamWrapper, SlidingWindow, DataPreprocessor
│
├── stream_reader.py (7.4 KB) ⭐
│   Purpose: Load data from river or fallback generators
│   Key Classes: StreamReader
│   Usage: Load agrawal, hyperplane, elec2, sea datasets
│   Dependencies: river, numpy
│
├── window_manager.py (1.3 KB)
│   Purpose: Implement sliding window
│   Key Classes: SlidingWindow
│   Usage: Buffer last N instances
│
└── preprocessing.py (3.3 KB)
    Purpose: Transform raw features to binary
    Key Classes: DataPreprocessor
    Usage: EMA + thresholding
```

### **DETECTION MODULES (src/fca_drift/detection/)**

```
detection/
├── __init__.py
│   Exports: FCADriftDetector, ADWINDetector, DDMDetector, EDDMDetector
│
├── fca_detector.py (27.6 KB) ⭐⭐⭐ MAIN DETECTOR
│   Purpose: Detect drift using FCA lattice comparison
│   Key Methods:
│   ├── update(lattice, instance_id) → DriftEvent or None
│   ├── _classify_drift_type() → "sudden"|"gradual"|"incremental"|"recurring"
│   ├── _check_recurring() → bool
│   └── _add_to_stable_buffer() → manage buffer
│
│   Detection Logic:
│   1. Compare current lattice vs previous (similarity)
│   2. Calculate ΔL = 1 - similarity
│   3. Check if ΔL > adaptive_theta
│   4. If yes → classify type
│   Dependencies: numpy, ConceptLattice, LatticeSimilarityCalculator
│
├── adwin_detector.py (1.5 KB)
│   Purpose: Alternative detector (ADWIN algorithm)
│   Status: Implemented but not used in main
│
├── ddm_detector.py (2.4 KB)
│   Purpose: Alternative detector (DDM algorithm)
│   Status: Implemented but not used in main
│
└── eddm_detector.py (2.4 KB)
    Purpose: Alternative detector (EDDM algorithm)
    Status: Implemented but not used in main
```

### **FCA MODULES (src/fca_drift/fca/)**

```
fca/
├── __init__.py
│   Exports: build_formal_context, ConceptLattice, LatticeSimilarityCalculator
│
├── context_builder.py (1.9 KB)
│   Purpose: Create formal context from binary data
│   Key Functions: build_formal_context()
│   Input: {x0: 0/1, x1: 0/1, ...} (binary features + label)
│   Output: FormalContext object
│
├── lattice_builder.py (7.4 KB)
│   Purpose: Build concept lattice from formal context
│   Key Classes: ConceptLattice
│   Methods:
│   ├── build_from_context(context)
│   ├── get_intents() → set of intent tuples
│   └── concepts → list of (extent, intent) pairs
│
│   Used by: FCADriftDetector for lattice comparison
│
└── similarity.py (4.1 KB)
    Purpose: Calculate similarity between lattices
    Key Classes: LatticeSimilarityCalculator
    Methods: compute_similarity(lattice1, lattice2) → float [0, 1]
    Algorithm: Based on intent/extent overlap
```

### **UTILS MODULES (src/fca_drift/utils/)**

```
utils/
├── drift_aggregator_v2.py (31.7 KB) ⭐
│   Purpose: Merge raw drifts into episodes (SINGLE SOURCE OF TRUTH)
│   Key Classes: DriftAggregatorV2, DriftEpisode
│   Methods:
│   ├── get_merged_episodes() → List[DriftEpisode]
│   ├── get_drift_rate() → float %
│   ├── get_dominant_type() → str
│   └── save_diagnostics() → JSON
│
│   Used by: main.py to aggregate detector results
│
├── drift_classifier.py (17.7 KB)
│   Purpose: Classify drift episodes by characteristics
│   Key Classes: DriftTypeAggregator, DriftTypeNormalizer
│   Used by: Advanced reporting (optional)
│
├── advanced_conclusion.py (12.8 KB)
│   Purpose: Generate advanced conclusions
│   Key Classes: ConclusionBuilder
│   Used by: Report generation
│
├── drift_results_ssot.py (7.0 KB)
│   Purpose: SSOT (Single Source of Truth) results storage
│   Used by: Data consistency
│
├── logger.py (2.1 KB)
│   Purpose: Experiment logging
│   Key Classes: ExperimentLogger
│   Used by: Diagnostics
│
└── __init__.py (100 B)
```

### **HISTORY MODULES (src/fca_drift/history/)**

```
history/
├── lattice_history_manager.py (10.8 KB)
│   Purpose: Track lattice evolution over time
│   Key Classes: LatticeHistoryManager, LatticeFrame
│   Methods:
│   ├── add_frame(frame)
│   ├── get_drift_frames()
│   └── frames → list of LatticeFrame
│
│   Used by: FCADriftDetector to maintain history
│   Input: Calculated metrics per window
│   Output: Historical data for visualization
│
└── __init__.py (343 B)
```

### **VISUALIZATION MODULES (src/fca_drift/visualization/)**

```
visualization/
├── detailed_report_v2.py (38.3 KB) ⭐
│   Purpose: Generate HTML reports
│   Key Classes: DetailedReportGeneratorV2
│   Methods: generate_html_report()
│   Output: report_en.html (with embedded images)
│
├── thesis_graphics.py (34.5 KB) ⭐
│   Purpose: Create publication-ready graphs
│   Key Classes: ThesisGraphicsExporter
│   Methods: export_all()
│   Output: 4 PNG files (DeltaL, Similarity, Distribution, Synchronized)
│
├── lattice_visualizer.py (27.3 KB)
│   Purpose: Render lattice diagrams
│   Key Classes: LatticeVisualizer
│   Methods: render_comparison_png()
│   Output: PNG images of lattices
│
├── lattice_integration_patch.py (16.7 KB)
│   Purpose: Integrate lattice visuals into HTML
│   Key Functions: generate_episode_row()
│   Used by: HTML report generation
│
├── plots.py (21.2 KB)
│   Purpose: Various analytical plots
│   Key Functions: create_drift_timeline(), plot_concepts_distribution()
│
├── analysis.py (12.5 KB)
│   Purpose: Analysis and summarization
│   Key Classes: DriftAnalyzer
│
├── lattice_snapshot_collector.py (6.1 KB)
│   Purpose: Collect lattice snapshots for visualization
│   Key Classes: LatticeSnapshotCollector
│
└── __init__.py (712 B)
```

---

## 🔗 Dependencies Between Key Files

### **For Basic Detection** (Minimum Required):

```
main.py
  ↓
fca_detector.py
  ├─ fca/lattice_builder.py
  ├─ fca/context_builder.py
  ├─ fca/similarity.py
  └─ core/
      ├─ stream_reader.py
      ├─ window_manager.py
      └─ preprocessing.py
```

### **For Complete System** (With Reports):

```
main.py
  ├─ fca_detector.py (+ all above)
  ├─ drift_aggregator_v2.py ← SSOT
  ├─ detailed_report_v2.py ← HTML
  ├─ thesis_graphics.py ← PNG graphs
  └─ lattice_visualizer.py
```

---

## 📊 File Statistics

| Category          | Files | Total Size |
| ----------------- | ----- | ---------- |
| **CORE**          | 3     | 12 KB      |
| **DETECTION**     | 5     | 35 KB      |
| **FCA**           | 3     | 13 KB      |
| **UTILS**         | 6     | 80 KB      |
| **HISTORY**       | 1     | 11 KB      |
| **VISUALIZATION** | 8     | 155 KB     |
| **MAIN**          | 1     | 23 KB      |
| **TOTAL**         | 27    | ~330 KB    |

**Code Concentration**:

- Detection logic: 35 KB (fca_detector.py)
- Reporting: 155 KB (visualization/)
- Utilities: 80 KB (aggregator, classifier)

---

## 🎯 Which Files to Focus On?

### **For Understanding Detection**:

1. **main.py** - Entry point and workflow
2. **fca_detector.py** - Core detection logic ⭐⭐⭐
3. **fca/similarity.py** - Lattice comparison
4. **drift_aggregator_v2.py** - Episode merging

### **For Debugging**:

1. Check **fca_detector.\_classify_drift_type()** - Type classification
2. Check **fca/similarity.py** - Similarity calculation
3. Check **drift_aggregator_v2.py** - Episode aggregation

### **For Optimization**:

1. **fca_detector.py** - Slow? Check lattice comparison
2. **visualization/thesis_graphics.py** - Slow reports? Disable PNG rendering
3. **drift_aggregator_v2.py** - Slow aggregation?

---

## 🚀 Typical Execution Flow

```
1. main.py runs
   ↓
2. Load dataset (stream_reader.py)
   ↓
3. Create sliding window (window_manager.py)
   ↓
4. For each window:
   ├─ Preprocess (preprocessing.py)
   ├─ Build lattice (fca/lattice_builder.py)
   ├─ Detect drift (fca_detector.py) ← MAIN LOGIC
   └─ Store in history (history_manager)
   ↓
5. Aggregate drifts (drift_aggregator_v2.py)
   ↓
6. Generate reports (visualization/)
   ↓
7. Save results to experiments/results/
```

---

## 📝 Configuration Files Used

| File                   | Purpose                                  |
| ---------------------- | ---------------------------------------- |
| `configs/default.yaml` | Default configuration                    |
| `requirements.txt`     | Dependencies (river, numpy, scipy, etc.) |
| `pyproject.toml`       | Project metadata                         |
| `setup.py`             | Installation configuration               |

---

## ✅ Summary

**Essential Files for Drift Detection**:

- ✅ main.py (orchestration)
- ✅ detection/fca_detector.py (detection engine)
- ✅ fca/\* (lattice operations)
- ✅ core/\* (data loading)
- ✅ utils/drift_aggregator_v2.py (result merging)

**Optional (for reports only)**:

- visualization/\* (HTML/PNG generation)
- history/\* (tracking)

**Total Working Code**: ~130 KB in 12-15 essential files
