#!/usr/bin/env python3
"""Run baseline detector comparison (ADWIN, DDM, EDDM) and merge with FCA results.

This script is intended to generate inputs for thesis section 4.4.
It expects FCA runs already present in experiments/results/*_w*_t0.5_a1.5/results.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import datetime as dt
import hashlib
from pathlib import Path
from typing import Dict, List

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from river.naive_bayes import GaussianNB

from fca_drift.core import StreamReader, StreamWrapper
from fca_drift.detection import DDM, EDDM
from fca_drift.detection.adwin_detector import ADWINDetector


def _stable_hash_to_float(value: str) -> float:
    """Convert a string into a deterministic numeric value."""
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:12]
    return float(int(digest, 16) % 1_000_000)


def _sanitize_features(x: Dict) -> Dict[str, float]:
    """Map mixed-type feature dict to numeric values for baseline classifiers."""
    out: Dict[str, float] = {}
    for k, v in x.items():
        key = str(k)
        if isinstance(v, bool):
            out[key] = 1.0 if v else 0.0
            continue
        if isinstance(v, (int, float)):
            out[key] = float(v)
            continue
        if isinstance(v, dt.datetime):
            out[key] = float(v.timestamp())
            continue
        if isinstance(v, dt.date):
            out[key] = float(dt.datetime(v.year, v.month, v.day).timestamp())
            continue

        # Fallback for categorical / unknown objects
        out[key] = _stable_hash_to_float(str(v))

    return out


def _run_error_based_detector(dataset: str, max_instances: int, method: str) -> Dict[str, float]:
    """Run one supervised error-based drift detector on a stream."""
    reader = StreamReader(dataset, seed=42)
    stream = StreamWrapper(reader.stream, max_instances=max_instances)

    model = GaussianNB()
    if method == "adwin":
        detector = ADWINDetector(delta=0.002)
    elif method == "ddm":
        detector = DDM(min_instances=30, warning_level=2.0, drift_level=3.0)
    elif method == "eddm":
        detector = EDDM(min_instances=30, warning_level=0.95, drift_level=0.90)
    else:
        raise ValueError(f"Unknown method: {method}")

    start = time.perf_counter()
    instance_count = 0

    for idx, (x, y) in enumerate(stream):
        instance_count += 1
        x_num = _sanitize_features(x)

        # Predict first, then learn; detector consumes classification error signal.
        y_pred = model.predict_one(x_num)
        if y_pred is not None:
            error = float(y_pred != y)
            if method == "adwin":
                detector.update(error, idx)
            else:
                detector.update(bool(error), idx)

        model.learn_one(x_num, y)

    elapsed = time.perf_counter() - start

    return {
        "instances": instance_count,
        "raw_alarms": len(detector.drift_indices),
        "runtime_sec": elapsed,
    }


def _load_fca_result(dataset: str, window_size: int, max_instances: int) -> Dict[str, float]:
    """Load FCA result JSON if present.

    Supports both legacy run directory naming and the current collision-safe naming
    that includes max instances.
    """
    candidates = [
        (
            PROJECT_ROOT
            / "experiments"
            / "results"
            / f"{dataset}_n{max_instances}_w{window_size}_t0.5_a1.5"
            / "results.json"
        ),
        (
            PROJECT_ROOT
            / "experiments"
            / "results"
            / f"{dataset}_w{window_size}_t0.5_a1.5"
            / "results.json"
        ),
    ]

    result_path = next((p for p in candidates if p.exists()), None)

    if result_path is None:
        return {
            "fca_instances": 0,
            "fca_raw": 0,
            "fca_episodes": 0,
            "fca_episode_rate_pct": 0.0,
            "fca_max_delta_L": 0.0,
        }

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    det = data.get("detection_results", {})
    return {
        "fca_instances": int(det.get("total_instances", 0)),
        "fca_raw": int(det.get("drifts_detected", 0)),
        "fca_episodes": int(data.get("aggregated_episodes", 0)),
        "fca_episode_rate_pct": float(data.get("drift_rate", 0.0)),
        "fca_max_delta_L": float(det.get("max_delta_L", 0.0)),
    }


def run_comparison(output_csv: Path) -> List[Dict[str, float]]:
    """Run all baseline comparisons and merge with FCA records."""
    # Datasets aligned with currently available FCA results in experiments/results.
    dataset_specs = [
        {"dataset": "agrawal", "max_instances": 20000, "window_size": 50},
        {"dataset": "elec2", "max_instances": 20000, "window_size": 50},
        {"dataset": "elec2", "max_instances": 20000, "window_size": 100},
        {"dataset": "phishing", "max_instances": 20000, "window_size": 50},
        {"dataset": "keystroke", "max_instances": 20000, "window_size": 50},
        {"dataset": "bikes", "max_instances": 20000, "window_size": 50},
        {"dataset": "http", "max_instances": 20000, "window_size": 50},
        {"dataset": "water_flow", "max_instances": 20000, "window_size": 50},
        {"dataset": "hyperplane", "max_instances": 20000, "window_size": 50},
    ]

    rows: List[Dict[str, float]] = []

    for spec in dataset_specs:
        dataset = spec["dataset"]
        max_instances = spec["max_instances"]
        window_size = spec["window_size"]

        print(f"\n[RUN] dataset={dataset}, W={window_size}, max_instances={max_instances}")

        try:
            # Run baseline detectors.
            adwin = _run_error_based_detector(dataset, max_instances, "adwin")
            ddm = _run_error_based_detector(dataset, max_instances, "ddm")
            eddm = _run_error_based_detector(dataset, max_instances, "eddm")
        except Exception as e:
            print(f"[ERROR] Baseline run failed for dataset={dataset}: {e}")
            adwin = {"instances": 0, "raw_alarms": 0, "runtime_sec": 0.0}
            ddm = {"instances": 0, "raw_alarms": 0, "runtime_sec": 0.0}
            eddm = {"instances": 0, "raw_alarms": 0, "runtime_sec": 0.0}

        # Load existing FCA outputs.
        fca = _load_fca_result(dataset, window_size, max_instances)

        row = {
            "dataset": dataset,
            "window_size": window_size,
            "instances": int(fca["fca_instances"] or adwin["instances"]),
            "fca_raw": int(fca["fca_raw"]),
            "fca_episodes": int(fca["fca_episodes"]),
            "fca_episode_rate_pct": float(fca["fca_episode_rate_pct"]),
            "fca_max_delta_L": float(fca["fca_max_delta_L"]),
            "adwin_raw": int(adwin["raw_alarms"]),
            "adwin_runtime_sec": round(float(adwin["runtime_sec"]), 3),
            "ddm_raw": int(ddm["raw_alarms"]),
            "ddm_runtime_sec": round(float(ddm["runtime_sec"]), 3),
            "eddm_raw": int(eddm["raw_alarms"]),
            "eddm_runtime_sec": round(float(eddm["runtime_sec"]), 3),
        }
        rows.append(row)

        print(
            "[OK] "
            f"FCA(ep)={row['fca_episodes']} | ADWIN={row['adwin_raw']} | "
            f"DDM={row['ddm_raw']} | EDDM={row['eddm_raw']}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "window_size",
        "instances",
        "fca_raw",
        "fca_episodes",
        "fca_episode_rate_pct",
        "fca_max_delta_L",
        "adwin_raw",
        "adwin_runtime_sec",
        "ddm_raw",
        "ddm_runtime_sec",
        "eddm_raw",
        "eddm_runtime_sec",
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Comparison CSV saved to: {output_csv}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FCA vs ADWIN/DDM/EDDM comparison")
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "results" / "baseline_comparison.csv"),
        help="Path to output CSV",
    )
    args = parser.parse_args()

    run_comparison(Path(args.output_csv))


if __name__ == "__main__":
    main()
