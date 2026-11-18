# FCA-Based Concept Drift Detection

A Python application for detecting **concept drift in streaming data** using **Formal Concept Analysis (FCA)**. The system represents data from sliding windows as formal concept lattices, measures changes between consecutive lattices, and uses an adaptive threshold to identify possible changes in the underlying data distribution.

The project combines data stream processing, numerical preprocessing, lattice-based similarity analysis, drift detection, and experiment reporting in a modular codebase.

## Overview

Concept drift occurs when the statistical properties or relationships within incoming data change over time. Detecting these changes is important for systems that continuously process data, including online machine learning and real-time analytics.

Instead of relying solely on prediction errors, this project analyzes changes in **formal concept lattices**. Each lattice captures relationships between objects and attributes within a window of observations. Comparing consecutive lattices provides a structural signal for potential drift.

## Features

- **Stream processing:** supports sliding-window analysis of data streams.
- **Dataset integration:** includes loaders for Elec2 and synthetic Agrawal, SEA, and Hyperplane streams through the River library.
- **Data preprocessing:** exponential moving average (EMA) smoothing, hysteresis, and threshold-based binarization.
- **Formal Concept Analysis:** constructs formal contexts and concept lattices, including concepts and parent–child relationships.
- **Lattice comparison:** combines concept, hierarchy, and structural similarity measures.
- **Drift detection:** monitors lattice dissimilarity using an adaptive statistical threshold and heuristic drift-type classification.
- **Evaluation utilities:** calculates precision, recall, F1-score, and detection delay against known drift points.
- **Reference methods:** includes DDM, EDDM, and an ADWIN wrapper as separate detector modules.
- **Experiment outputs:** records processing steps and drift events in CSV and produces static plots for analysis.

## Processing Pipeline

```text
Data stream
    |
    v
Sliding window
    |
    v
Preprocessing (EMA, hysteresis, binarization)
    |
    v
Formal context construction
    |
    v
Concept lattice construction
    |
    v
Comparison with previous lattice
    |
    v
Lattice dissimilarity (Delta L)
    |
    v
Adaptive drift detection
    |
    v
CSV logs and visualizations
```

The lattice change score is calculated as:

\[
\Delta L_t = 1 - \operatorname{Sim}(L_t, L_{t-1})
\]

where the similarity function combines concept-level, hierarchical, and structural comparisons.

## Technology Stack

- **Language:** Python 3.10+
- **Data processing:** NumPy, pandas, River
- **Formal Concept Analysis:** `concepts`
- **Visualization:** Matplotlib, Seaborn, Graphviz
- **Configuration:** PyYAML
- **Testing and development:** pytest, Black, Flake8, isort, mypy

## Installation

Clone the repository and create a virtual environment:

```bash
git clone <repository-url>
cd fca-drift-detector
python -m venv .venv
```

Activate the environment:

```bash
# Windows (Git Bash)
source .venv/Scripts/activate

# Linux / macOS
source .venv/bin/activate
```

Install dependencies and the local package:

```bash
pip install -r requirements.txt
pip install -e .
```

Graphviz may also need to be installed separately at the operating-system level for graph-related functionality.

## Usage

Run a drift detection experiment:

```bash
python -m fca_drift \
  --dataset agrawal \
  --max-instances 2000 \
  --window-size 50 \
  --theta 0.4 \
  --output-dir experiments/results/first_test
```

The documented CLI supports the following options:

| Argument | Description |
| --- | --- |
| `--dataset` | Dataset name (`elec2`, `agrawal`, `sea`, or `hyperplane`) |
| `--max-instances` | Maximum number of stream observations to process |
| `--window-size` | Number of observations in each sliding window |
| `--theta` | Initial drift detection threshold |
| `--alpha` | Sensitivity multiplier for the adaptive threshold |
| `--output-dir` | Directory for experiment outputs |

Generated output includes per-step CSV logs, recorded drift events, and a plot of lattice dissimilarity over time.

## Project Structure

```text
src/fca_drift/
├── core/           # Stream loading, sliding windows, preprocessing
├── fca/            # Formal contexts, concept lattices, similarity
├── detection/      # FCA-based detection and reference detectors
├── evaluation/     # Detection performance metrics
├── experiments/    # Experiment-related modules
├── visualization/  # Plot generation
├── data/           # Data-related modules
├── utils/          # Logging utilities
└── __main__.py     # Command-line entry point

configs/            # Configuration files
docs/               # Project documentation
experiments/        # Experiment outputs and logs
tests/              # Tests
```

## Documentation

Additional information is available in the [`docs/`](docs/) directory, including setup, configuration, API, and binary distribution documentation.

## Notes

This project is intended for experimenting with structural approaches to concept drift detection. Detection behavior depends on the dataset, preprocessing settings, window size, and thresholds. The presence of evaluation and baseline-detector modules does not imply any particular performance advantage; such claims require reproducible comparative experiments.
