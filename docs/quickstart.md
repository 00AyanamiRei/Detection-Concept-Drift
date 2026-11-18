# Rýchly štart

## Požiadavky

- Python 3.10+.
- Balíky z `requirements.txt`.
- `graphviz` v systéme (používa ho knižnica `concepts`).

## Inštalácia

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Najkratší programový príklad

```python
from fca_drift.core import StreamReader, StreamWrapper, SlidingWindow, DataPreprocessor
from fca_drift.fca import build_formal_context, ConceptLattice
from fca_drift.detection import FCADriftDetector

reader = StreamReader("agrawal", seed=42, drift_positions=[5000])
stream = StreamWrapper(reader.stream, max_instances=10000)
window = SlidingWindow(size=300)
pre = DataPreprocessor()

detector = FCADriftDetector(theta=0.4, alpha=2.0, window_size=300)

for idx, (x, y) in enumerate(stream):
    window.append(x)
    if window.is_full():
        binary = pre.preprocess_window(window.get_data())
        context = build_formal_context(binary)
        lattice = ConceptLattice()
        lattice.build_from_context(context, max_objects=50)
        detector.update(lattice, idx)

print("Drifty:", detector.drift_indices)
```

## CLI (fca-drift)

Po instalacii projektu je dostupny CLI entrypoint.

```bash
python -m fca_drift --dataset agrawal --max-instances 10000 --window-size 300 --theta 0.4 --alpha 2.0 --drift-position 5000
```

Ak mas projekt nainstalovany ako balicek, funguje aj:

```bash
fca-drift --dataset agrawal --max-instances 10000 --window-size 300 --theta 0.4 --alpha 2.0 --drift-position 5000
```

## Spustenie skriptov

- Baseline porovnanie (ADWIN, DDM, EDDM):
  ```bash
  python scripts\run_baseline_comparison.py
  ```
- Virtual drift probe (duálny signál):
  ```bash
  python scripts\virtual_drift_probe.py --dataset elec2 --max-instances 2500 --window-size 50
  ```

## Overenie výsledkov

```bash
python verify_results.py experiments\results\<run>\report_debug.json ground_truth_5000_labels.json
```

## EXE build

Navod na vytvorenie .exe je v [docs/binaries.md](binaries.md).

## Poznámky a obmedzenia

- `run_methodology_suite.py` pouziva legacy `main.py` (zakomentovany). Odporucane je
  spustat CLI cez `python -m fca_drift` alebo vytvoreny EXE.
- Konfigurácie v `configs/` sú šablóny, momentálne ich priamo nenačítava kód.
