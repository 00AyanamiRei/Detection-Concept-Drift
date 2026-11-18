# API Reference

This page summarizes the main public-facing classes and functions in the `fca_drift` package. For function signatures and implementation details, see [`src/fca_drift/`](../src/fca_drift/).

## `fca_drift.core`

| Interface | Description |
| --- | --- |
| `StreamReader(dataset_name, seed=42, **kwargs)` | Unified stream source interface. Supports datasets from River and synthetic streams, including `agrawal`, `sea`, `waveform`, `hyperplane`, and `mixed`. Synthetic drift positions can be specified through `drift_position` or `drift_positions`, depending on the generator. |
| `StreamWrapper(stream, max_instances=None)` | Limits the number of observations read from a stream. |
| `SlidingWindow(size)` | Maintains a fixed-size FIFO window of observations. |
| `DataPreprocessor(...)` | Converts window data into a binary representation for FCA using quantile-based thresholds. Key methods: `preprocess_window` and `get_frozen_thresholds`. |
| `get_available_datasets()` | Returns supported dataset names. |

## `fca_drift.fca`

| Interface | Description |
| --- | --- |
| `FormalContext` | Represents an FCA context $(G, M, I)$. |
| `build_formal_context(binary_matrix)` | Creates a formal context from a binary matrix. |
| `FormalConcept` | Represents a concept through its extent and intent. |
| `ConceptLattice` | Builds concepts, hierarchy, and lattice levels. Key method: `build_from_context(formal_context, max_objects=50)`. |
| `LatticeSimilarityCalculator` | Computes similarity between concept lattices. |

## `fca_drift.detection`

| Interface | Description |
| --- | --- |
| `DriftEvent` | Data structure describing a detected drift event. |
| `BaseDriftDetector` | Abstract base class for drift detectors. |
| `FCADriftDetector` | Compares successive FCA lattices to detect drift. Key method: `update(lattice, instance_id)`. |
| `DDM`, `EDDM`, `ADWINDetector` | Reference drift detectors based on prediction-error or numeric signals. |

## `fca_drift.evaluation`

| Interface | Description |
| --- | --- |
| `DriftEvaluator(true_drift_points, tolerance=50)` | Evaluates detections against reference drift positions using precision, recall, F1-score, and detection delay. |
| `print_evaluation_report(metrics, method_name="FCA")` | Prints a formatted evaluation summary. |

## `fca_drift.experiments`

| Interface | Description |
| --- | --- |
| `GridSearchRunner` | Explores combinations of `window_size`, `theta`, and `alpha`. |
| `DetectorComparison` | Compares FCA detection with DDM and EDDM on the same stream. |

## `fca_drift.history`

| Interface | Description |
| --- | --- |
| `LatticeFrame` | Captures a lattice snapshot and associated signal. |
| `LatticeHistoryManager` | Stores lattice history and statistics. |
| `LatticeComparator` | Compares windows before and after a drift event. |
| `LatticeHistorySerializer` | Exports history to JSON or CSV. |
| `DriftStatusEnum` | Represents the `normal`, `soft`, `hard`, and `virtual` statuses. |

## `fca_drift.utils`

| Interface | Description |
| --- | --- |
| `ExperimentLogger` | Records experiment data in CSV and JSON formats. |
| `DriftAggregatorV2` | Groups detections into drift episodes and provides the consolidated source of drift results. |
| `DriftResults` | Consolidates results for reporting. |
| `DriftTypeClassifier` | Classifies drift patterns from the change signal. |
| `AdvancedConclusionBuilder`, `ConclusionBuilder` | Generate analytical conclusions for reports. |

## `fca_drift.visualization`

| Interface | Description |
| --- | --- |
| `DetailedReportGeneratorV2` | Generates localized HTML reports with plots. |
| `DriftAnalyzer` | Produces a text-based analysis of detected drift. |
| `LatticeSnapshotCollector` | Collects lattice pairs for reporting. |
| `plots` | Provides plots of $\Delta L$, distributions, and synchronized signals. |

## Related Documentation

See the [Quick Start](quickstart.md) for usage examples and [Configuration](configuration.md) for parameter descriptions.
