#!/usr/bin/env python3
"""Run baseline detector comparison (ADWIN, DDM, EDDM) and merge with FCA results.

This script is intended to generate inputs for thesis section 4.4.
It expects FCA runs already present in experiments/results with matching
dataset/window-size and the lookup parameters defined below.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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
from river.drift import ADWIN
from river.drift import binary as river_binary

from fca_drift.core import StreamReader, StreamWrapper


FCA_LOOKUP_DEFAULTS = {
    "theta": 0.5,
    "alpha": 1.5,
    "merge_gap": None,
}

FCA_LOOKUP_BY_DATASET = {
    "elec2": {"theta": 0.3, "alpha": 2.0, "merge_gap": 20},
    "http": {"theta": 0.3, "alpha": 2.0, "merge_gap": 60},
}


def _fmt_param(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        return f"{value:.1f}"
    return f"{value}"


def _get_fca_lookup_params(dataset: str) -> Dict[str, float]:
    params = dict(FCA_LOOKUP_DEFAULTS)
    params.update(FCA_LOOKUP_BY_DATASET.get(dataset.lower(), {}))
    return params


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
    reader_kwargs = {}
    if dataset.lower() in {"agrawal", "sea"}:
        reader_kwargs["drift_positions"] = [max_instances // 2]
    reader = StreamReader(dataset, seed=42, **reader_kwargs)
    stream = StreamWrapper(reader.stream, max_instances=max_instances)

    model = GaussianNB()
    if method == "adwin":
        detector = ADWIN(delta=0.002)
    elif method == "ddm":
        detector = river_binary.DDM(warm_start=30, warning_threshold=2.0, drift_threshold=3.0)
    elif method == "eddm":
        detector = river_binary.EDDM(warm_start=30, alpha=0.95, beta=0.90)
    else:
        raise ValueError(f"Unknown method: {method}")

    drift_indices = []
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
                detector.update(error)
                if detector.drift_detected:
                    drift_indices.append(idx)
            else:
                detector.update(1 if error else 0)
                if detector.drift_detected:
                    drift_indices.append(idx)

        model.learn_one(x_num, y)

    elapsed = time.perf_counter() - start

    return {
        "instances": instance_count,
        "raw_alarms": len(drift_indices),
        "runtime_sec": elapsed,
    }


def _load_fca_result(dataset: str, window_size: int, max_instances: int) -> Dict[str, float]:
    """Load FCA result JSON using dataset-specific lookup parameters."""
    lookup = _get_fca_lookup_params(dataset)
    theta = lookup["theta"]
    alpha = lookup["alpha"]
    merge_gap = lookup.get("merge_gap")

    results_root = PROJECT_ROOT / "experiments" / "results"
    t_str = _fmt_param(theta)
    a_str = _fmt_param(alpha)

    patterns = [
        f"{dataset}_n*_w{window_size}_t{t_str}_a{a_str}*",
        f"{dataset}_w{window_size}_t{t_str}_a{a_str}*",
    ]

    candidate_dirs = []
    for pat in patterns:
        candidate_dirs.extend([p for p in results_root.rglob(pat) if p.is_dir()])

    def pick_best(strict_merge_gap: bool) -> Dict[str, float] | None:
        scored = []
        for run_dir in candidate_dirs:
            result_path = run_dir / "results.json"
            if not result_path.exists():
                continue
            try:
                data = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            params = data.get("parameters", {}) or {}
            if strict_merge_gap and merge_gap is not None:
                if params.get("merge_gap") is None:
                    continue
                if int(params.get("merge_gap")) != int(merge_gap):
                    continue

            run_max_instances = params.get("max_instances")
            max_match = int(run_max_instances) == int(max_instances) if run_max_instances is not None else False
            scored.append((max_match, run_dir.stat().st_mtime, data))

        if not scored:
            return None

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2]

    data = pick_best(strict_merge_gap=True)
    if data is None and merge_gap is not None:
        data = pick_best(strict_merge_gap=False)

    if data is None:
        return {
            "fca_instances": 0,
            "fca_raw": 0,
            "fca_episodes": 0,
            "fca_episode_rate_pct": 0.0,
            "fca_max_delta_L": 0.0,
        }

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
        {"dataset": "agrawal", "max_instances": 20000, "window_size": 300},
        {"dataset": "elec2", "max_instances": 10000, "window_size": 300},
        {"dataset": "elec2", "max_instances": 10000, "window_size": 500},
        {"dataset": "phishing", "max_instances": 20000, "window_size": 300},
        {"dataset": "keystroke", "max_instances": 20000, "window_size": 300},
        {"dataset": "bikes", "max_instances": 20000, "window_size": 300},
        {"dataset": "http", "max_instances": 10000, "window_size": 300},
        {"dataset": "water_flow", "max_instances": 20000, "window_size": 300},
        {"dataset": "hyperplane", "max_instances": 20000, "window_size": 300},
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
