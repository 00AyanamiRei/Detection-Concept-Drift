# Quick Start

This guide covers installation, a command-line experiment, and programmatic access to the FCA-based drift detector.

## Requirements

- Python 3.10 or later.
- Dependencies listed in `requirements.txt`.
- Graphviz installed on the system where required by the FCA toolchain.

## Installation

From the project root, create a virtual environment:

```bash
python -m venv .venv
```

Activate it using the command for your terminal:

```bash
# Windows / Git Bash
source .venv/Scripts/activate

# Linux / macOS
# source .venv/bin/activate
```

For Windows PowerShell, use `.venv\Scripts\Activate.ps1` instead.

Install the dependencies and the project in editable mode:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run an Experiment

The package exposes a command-line entry point:

```bash
python -m fca_drift --dataset agrawal --max-instances 10000 --window-size 300 --theta 0.4 --alpha 2.0 --drift-position 5000
```

If the console script is available after package installation, the equivalent command is:

```bash
fca-drift --dataset agrawal --max-instances 10000 --window-size 300 --theta 0.4 --alpha 2.0 --drift-position 5000
```

List the CLI arguments with:

```bash
python -m fca_drift --help
```

## Python Example

The example below constructs concept lattices from sliding windows and sends them to the drift detector:

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

print("Detected drift positions:", detector.drift_indices)
```

This is a minimal API example. The full CLI workflow additionally handles experiment outputs and reporting.

## Additional Scripts

Run a baseline-detector comparison (ADWIN, DDM, EDDM):

```bash
python scripts/run_baseline_comparison.py
```

Run the virtual-drift probe:

```bash
python scripts/virtual_drift_probe.py --dataset elec2 --max-instances 2500 --window-size 50
```

## Result Verification

To compare a generated debug report against known drift positions, run:

```bash
python verify_results.py "experiments/results/<run>/report_debug.json" ground_truth_5000_labels.json
```

Replace `<run>` with the output directory for the relevant experiment, and ensure both input files exist.

## Windows Executable

For optional Windows packaging, see [Windows Executable](binaries.md).

## Current Limitations

- Configuration files under `configs/` are templates and are not automatically loaded by the current application.
- `run_methodology_suite.py` refers to a legacy `main.py` workflow that is commented out. Prefer the package CLI (`python -m fca_drift`) for regular runs.
- Detection results depend on the dataset and preprocessing and detector parameters; validate reported events against the available ground truth or reference methods.
