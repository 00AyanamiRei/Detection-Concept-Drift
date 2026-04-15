#!/usr/bin/env python3
"""Lightweight dual-channel virtual/real drift probe.

Combines:
- S_X: structural signal from FCA episode dlt_max
- S_E: rolling classification error from GaussianNB

This is an analysis helper (not wired into production detector).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np
from river.naive_bayes import GaussianNB

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fca_drift.core import StreamReader, StreamWrapper


def _sanitize_features(x: Dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in x.items():
        key = str(k)
        if isinstance(v, bool):
            out[key] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            out[key] = float(v)
        else:
            out[key] = float(abs(hash(str(v))) % 1_000_000)
    return out


def _find_run_dir(results_root: Path, dataset: str, max_instances: int, window_size: int) -> Path:
    pattern = f"{dataset}_n{max_instances}_w{window_size}_t0.5_a1.5*"
    candidates = sorted(results_root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No run dir matching {pattern}")
    return candidates[0]


def _rolling_error_series(dataset: str, max_instances: int, seed: int, error_window: int) -> List[float]:
    reader = StreamReader(dataset, seed=seed)
    stream = StreamWrapper(reader.stream, max_instances=max_instances)

    model = GaussianNB()
    errors: List[float] = []
    buf = deque(maxlen=error_window)

    for x, y in stream:
        x_num = _sanitize_features(x)
        y_pred = model.predict_one(x_num)
        if y_pred is None:
            err = 0.0
        else:
            err = float(y_pred != y)
        buf.append(err)
        errors.append(float(np.mean(buf)) if buf else 0.0)
        model.learn_one(x_num, y)

    return errors


def run_probe(dataset: str, max_instances: int, window_size: int, seed: int, error_window: int, output_csv: Path) -> Path:
    results_root = PROJECT_ROOT / "experiments" / "results"
    run_dir = _find_run_dir(results_root, dataset, max_instances, window_size)

    with open(run_dir / "report_debug.json", "r", encoding="utf-8") as f:
        debug = json.load(f)

    episodes = debug.get("episodes", [])
    if not episodes:
        raise RuntimeError("No episodes in report_debug.json")

    err = _rolling_error_series(dataset, max_instances, seed, error_window)

    sx_values = [float(ep.get("dlt_max", 0.0)) for ep in episodes]
    se_values = []
    for ep in episodes:
        center = int(ep.get("center_instance_id", 0))
        idx = max(0, min(center, len(err) - 1))
        se_values.append(float(err[idx]))

    sx_high = float(np.quantile(sx_values, 0.75))
    se_low = float(np.quantile(se_values, 0.25))
    se_high = float(np.quantile(se_values, 0.75))

    rows = []
    for ep, se in zip(episodes, se_values):
        sx = float(ep.get("dlt_max", 0.0))
        if sx >= sx_high and se <= se_low:
            label = "virtual_candidate"
        elif sx >= sx_high and se >= se_high:
            label = "real_candidate"
        else:
            label = "mixed"

        rows.append(
            {
                "start_instance_id": int(ep.get("start_instance_id", 0)),
                "end_instance_id": int(ep.get("end_instance_id", 0)),
                "center_instance_id": int(ep.get("center_instance_id", 0)),
                "dlt_max": sx,
                "rolling_error": se,
                "episode_type": str(ep.get("dominant_type", "unknown")),
                "dual_channel_label": label,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "start_instance_id",
                "end_instance_id",
                "center_instance_id",
                "dlt_max",
                "rolling_error",
                "episode_type",
                "dual_channel_label",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": dataset,
        "run_dir": str(run_dir),
        "episodes": len(rows),
        "sx_high": round(sx_high, 6),
        "se_low": round(se_low, 6),
        "se_high": round(se_high, 6),
        "virtual_candidates": sum(r["dual_channel_label"] == "virtual_candidate" for r in rows),
        "real_candidates": sum(r["dual_channel_label"] == "real_candidate" for r in rows),
        "mixed": sum(r["dual_channel_label"] == "mixed" for r in rows),
        "output_csv": str(output_csv),
    }

    summary_path = output_csv.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-channel virtual/real drift probe")
    parser.add_argument("--dataset", type=str, default="elec2")
    parser.add_argument("--max-instances", type=int, default=2500)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--error-window", type=int, default=100)
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "results_patch_validation" / "virtual_drift_probe.csv"),
    )
    args = parser.parse_args()

    summary_path = run_probe(
        dataset=args.dataset,
        max_instances=args.max_instances,
        window_size=args.window_size,
        seed=args.seed,
        error_window=args.error_window,
        output_csv=Path(args.output_csv),
    )

    print(f"[DONE] Virtual-drift probe summary: {summary_path}")


if __name__ == "__main__":
    main()
