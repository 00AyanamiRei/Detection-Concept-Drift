# API referencie (manuálny prehľad)

Táto sekcia mapuje hlavné triedy a funkcie podľa balíkov. Detaily implementácie
nájdeš priamo v `src/fca_drift/`.

## `fca_drift.core`
- `StreamReader(dataset_name, seed=42, **kwargs)`: jednotný vstup pre streamy.
  Podporuje real datasets z `river` a syntetické (`agrawal`, `sea`, `waveform`,
  `hyperplane`, `mixed`). Parametre `drift_position`/`drift_positions` riadia
  syntetické drifty.
- `StreamWrapper(stream, max_instances=None)`: obmedzí počet inštancií zo streamu.
- `SlidingWindow(size)`: FIFO okno nad streamom.
- `DataPreprocessor(...)`: kvantilová binarizácia pre FCA kontext.
  Kľúčové metódy: `preprocess_window`, `get_frozen_thresholds`.
- `get_available_datasets()`: zoznam podporovaných datasetov.

## `fca_drift.fca`
- `FormalContext`: reprezentácia (G, M, I) pre FCA.
- `build_formal_context(binary_matrix)`: vytvorenie kontextu z binárnej matice.
- `FormalConcept`: dvojica extent/intent.
- `ConceptLattice`: výpočet mriežky, hierarchie a úrovní.
  Kľúčová metóda: `build_from_context(formal_context, max_objects=50)`.
- `LatticeSimilarityCalculator`: porovnanie mriežok a výpočet podobnosti.

## `fca_drift.detection`
- `DriftEvent`: dátová trieda udalosti driftu.
- `BaseDriftDetector`: abstraktný základ detektorov.
- `FCADriftDetector`: hlavný detektor nad FCA mriežkami.
  Kľúčová metóda: `update(lattice, instance_id)`.
- `DDM`, `EDDM`, `ADWINDetector`: baseline detektory nad chybou.

## `fca_drift.evaluation`
- `DriftEvaluator(true_drift_points, tolerance=50)`: precision/recall/F1/delay.
- `print_evaluation_report(metrics, method_name="FCA")`.

## `fca_drift.experiments`
- `GridSearchRunner`: grid search pre parametre `window_size`, `theta`, `alpha`.
- `DetectorComparison`: porovná FCA s DDM/EDDM (na rovnakom streame).

## `fca_drift.history`
- `LatticeFrame`: snapshot mriežky a signálu.
- `LatticeHistoryManager`: uchovávanie histórie a štatistík.
- `LatticeComparator`: porovnanie okien pred/po drifte.
- `LatticeHistorySerializer`: export do JSON/CSV.
- `DriftStatusEnum`: normal/soft/hard/virtual.

## `fca_drift.utils`
- `ExperimentLogger`: CSV/JSON logging.
- `DriftAggregatorV2`: agregácia drift epizód (SSOT pre výsledky).
- `DriftResults`: konsolidovaný výsledok pre reporty.
- `DriftTypeClassifier`: klasifikácia typu driftu podľa signálu.
- `AdvancedConclusionBuilder`, `ConclusionBuilder`: odporúčania pre závery.

## `fca_drift.visualization`
- `DetailedReportGeneratorV2`: HTML report s grafmi a lokalizáciou.
- `DriftAnalyzer`: textový rozbor driftov.
- `LatticeSnapshotCollector`: zber párov mriežok pre reporty.
- `plots`: kreslenie ΔL, distribúcie a synchronizovaných grafov.
