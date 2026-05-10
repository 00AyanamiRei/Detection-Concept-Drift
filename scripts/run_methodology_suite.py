#!/usr/bin/env python3
"""Run focused methodology probes for thesis hardening.

Outputs CSV summaries for:
1) gradual-dominance ablation (merge-gap / smoothing)
2) multi-seed stability (episodes, type shares, localization)
3) scalability probe (runtime and peak Python memory vs dimensionality)
"""

from __future__ import annotations

import csv
import json
import re
import argparse
import statistics as stats
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = PROJECT_ROOT / "main.py"
RESULTS_ROOT = PROJECT_ROOT / "experiments" / "results"
OUT_ROOT = PROJECT_ROOT / "experiments" / "results_patch_validation"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from fca_drift.fca import build_formal_context, ConceptLattice


def _sanitize_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip("-_.")
    return cleaned[:48] if cleaned else "run"


def _run_dir_name(dataset: str, n: int, w: int, theta: float, alpha: float, run_id: str) -> str:
    return f"{dataset}_n{n}_w{w}_t{theta:.1f}_a{alpha:.1f}_id{_sanitize_run_id(run_id)}"


def _find_run_dir(expected_name: str) -> Path:
    exact = RESULTS_ROOT / expected_name
    if exact.exists():
        return exact

    candidates = sorted(RESULTS_ROOT.glob(f"{expected_name}*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"Run directory not found for prefix: {expected_name}")
    return candidates[0]


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _run_main_case(
    *,
    dataset: str,
    max_instances: int,
    window_size: int,
    theta: float = 0.5,
    alpha: float = 1.5,
    run_id: str,
    seed: int = 42,
    merge_gap: int = 150,
    smoothing_window: int = 10,
    dominant_mode: str = "weighted",
    n_features: int = 10,
    max_fca_attrs: int = 15,
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(MAIN_PY),
        "--dataset",
        dataset,
        "--max-instances",
        str(max_instances),
        "--window-size",
        str(window_size),
        "--theta",
        str(theta),
        "--alpha",
        str(alpha),
        "--language",
        "en",
        "--output",
        "experiments/results",
        "--run-id",
        run_id,
        "--seed",
        str(seed),
        "--merge-gap",
        str(merge_gap),
        "--signal-smoothing-window",
        str(smoothing_window),
        "--dominant-type-mode",
        dominant_mode,
        "--n-features",
        str(n_features),
        "--max-fca-attrs",
        str(max_fca_attrs),
    ]

    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Run failed for dataset={dataset}, run_id={run_id}")

    expected = _run_dir_name(dataset, max_instances, window_size, theta, alpha, run_id)
    run_dir = _find_run_dir(expected)

    debug = _read_json(run_dir / "report_debug.json")
    results = _read_json(run_dir / "results.json")

    return {
        "run_dir": str(run_dir),
        "debug": debug,
        "results": results,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_ablation_gradual() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    configs = [
        {"label": "baseline_mg150_sw10", "merge_gap": 150, "sw": 10},
        {"label": "tight_merge_mg30_sw10", "merge_gap": 30, "sw": 10},
        {"label": "no_smooth_mg150_sw1", "merge_gap": 150, "sw": 1},
        {"label": "tight_no_smooth_mg30_sw1", "merge_gap": 30, "sw": 1},
    ]

    for cfg in configs:
        run = _run_main_case(
            dataset="elec2",
            max_instances=10000,
            window_size=300,
            run_id=f"abl_{cfg['label']}",
            seed=42,
            merge_gap=cfg["merge_gap"],
            smoothing_window=cfg["sw"],
            dominant_mode="weighted",
            n_features=10,
            max_fca_attrs=15,
        )

        debug = run["debug"]
        counts = debug.get("counts_by_type", {})
        total = sum(counts.values())
        gradual = counts.get("gradual", 0)
        sudden = counts.get("sudden", 0)
        unknown = counts.get("unknown", 0)

        rows.append(
            {
                "config": cfg["label"],
                "dataset": "elec2",
                "instances": debug.get("total_instances", 0),
                "episodes": debug.get("merged_episodes_count", 0),
                "gradual_count": gradual,
                "sudden_count": sudden,
                "unknown_count": unknown,
                "gradual_share_pct": round(100.0 * _safe_div(gradual, total), 2),
                "sudden_share_pct": round(100.0 * _safe_div(sudden, total), 2),
                "unknown_share_pct": round(100.0 * _safe_div(unknown, total), 2),
                "dominant_majority": debug.get("dominant_type_majority", "none"),
                "dominant_weighted": debug.get("dominant_type_weighted", "none"),
                "runtime_sec": run["results"].get("detection_results", {}).get("runtime_sec_detection", 0),
                "peak_python_mem_mb": run["results"].get("detection_results", {}).get("peak_python_mem_mb", 0),
                "run_dir": run["run_dir"],
            }
        )

    return rows


def _agrawal_localization(debug: Dict[str, Any], drift_point: int = 5000) -> Dict[str, Any]:
    episodes = debug.get("episodes", [])
    if not episodes:
        return {
            "delta_center_abs": None,
            "delta_center_signed": None,
            "delay_start_after_drift": None,
            "hits_within_200": 0,
            "hits_within_500": 0,
        }

    centers = [int(ep.get("center_instance_id", 0)) for ep in episodes]
    starts = [int(ep.get("start_instance_id", 0)) for ep in episodes]

    nearest = min(centers, key=lambda c: abs(c - drift_point))
    first_after = min((s for s in starts if s >= drift_point), default=None)

    return {
        "delta_center_abs": abs(nearest - drift_point),
        "delta_center_signed": nearest - drift_point,
        "delay_start_after_drift": (first_after - drift_point) if first_after is not None else None,
        "hits_within_200": sum(1 for c in centers if abs(c - drift_point) <= 200),
        "hits_within_500": sum(1 for c in centers if abs(c - drift_point) <= 500),
    }


def run_multiseed() -> List[Dict[str, Any]]:
    seeds = [21, 42, 84]
    specs = [
        {"dataset": "elec2", "window_size": 300, "max_instances": 5000},
    ]

    rows: List[Dict[str, Any]] = []

    for spec in specs:
        for seed in seeds:
            run = _run_main_case(
                dataset=spec["dataset"],
                max_instances=spec["max_instances"],
                window_size=spec["window_size"],
                run_id=f"ms_{spec['dataset']}_s{seed}",
                seed=seed,
                merge_gap=150,
                smoothing_window=10,
                dominant_mode="weighted",
                n_features=10,
                max_fca_attrs=15,
            )

            debug = run["debug"]
            counts = debug.get("counts_by_type", {})
            total = sum(counts.values())
            row = {
                "dataset": spec["dataset"],
                "seed": seed,
                "instances": debug.get("total_instances", 0),
                "episodes": debug.get("merged_episodes_count", 0),
                "drift_rate_pct": round(float(debug.get("drift_rate_percent", 0.0)), 4),
                "gradual_count": counts.get("gradual", 0),
                "sudden_count": counts.get("sudden", 0),
                "unknown_count": counts.get("unknown", 0),
                "gradual_share_pct": round(100.0 * _safe_div(counts.get("gradual", 0), total), 2),
                "dominant_majority": debug.get("dominant_type_majority", "none"),
                "dominant_weighted": debug.get("dominant_type_weighted", "none"),
                "runtime_sec": run["results"].get("detection_results", {}).get("runtime_sec_detection", 0),
                "peak_python_mem_mb": run["results"].get("detection_results", {}).get("peak_python_mem_mb", 0),
                "run_dir": run["run_dir"],
            }

            if spec["dataset"] == "agrawal":
                row.update(_agrawal_localization(debug, drift_point=5000))
            else:
                row.update(
                    {
                        "delta_center_abs": None,
                        "delta_center_signed": None,
                        "delay_start_after_drift": None,
                        "hits_within_200": None,
                        "hits_within_500": None,
                    }
                )

            rows.append(row)

    return rows


def summarize_multiseed(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for dataset in sorted({r["dataset"] for r in rows}):
        subset = [r for r in rows if r["dataset"] == dataset]

        def mean_sd(values: List[float]) -> str:
            if not values:
                return "n/a"
            if len(values) == 1:
                return f"{values[0]:.3f}"
            return f"{stats.mean(values):.3f} ± {stats.stdev(values):.3f}"

        episodes = [float(r["episodes"]) for r in subset]
        gradual_share = [float(r["gradual_share_pct"]) for r in subset]
        runtime = [float(r["runtime_sec"]) for r in subset]
        mem = [float(r["peak_python_mem_mb"]) for r in subset]

        row = {
            "dataset": dataset,
            "seeds": len(subset),
            "episodes_mean_sd": mean_sd(episodes),
            "gradual_share_pct_mean_sd": mean_sd(gradual_share),
            "runtime_sec_mean_sd": mean_sd(runtime),
            "peak_python_mem_mb_mean_sd": mean_sd(mem),
        }

        if dataset == "agrawal":
            dabs = [float(r["delta_center_abs"]) for r in subset if r["delta_center_abs"] is not None]
            dstart = [float(r["delay_start_after_drift"]) for r in subset if r["delay_start_after_drift"] is not None]
            row["delta_center_abs_mean_sd"] = mean_sd(dabs)
            row["delay_start_mean_sd"] = mean_sd(dstart)
        else:
            row["delta_center_abs_mean_sd"] = "n/a"
            row["delay_start_mean_sd"] = "n/a"

        out.append(row)

    return out


def run_scalability_probe() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    feature_levels = [10, 20, 50]
    rows_per_window = 8
    repeats = 2
    rng = np.random.default_rng(42)

    for fcount in feature_levels:
        runtimes: List[float] = []
        peaks: List[float] = []
        concept_counts: List[float] = []

        for _ in range(repeats):
            binary_data = rng.integers(0, 2, size=(rows_per_window, fcount), dtype=np.int8)

            tracemalloc.start()
            start = time.perf_counter()

            context = build_formal_context(binary_data)
            lattice = ConceptLattice()
            lattice.build_from_context(context)

            elapsed = time.perf_counter() - start
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            runtimes.append(float(elapsed))
            peaks.append(float(peak_bytes) / (1024 * 1024))
            concept_counts.append(float(lattice.get_concept_count()))

        rows.append(
            {
                "dataset": "synthetic_context_probe",
                "n_features": fcount,
                "rows_per_window": rows_per_window,
                "repeats": repeats,
                "concept_count_mean": round(stats.mean(concept_counts), 3),
                "concept_count_sd": round(stats.stdev(concept_counts), 3) if len(concept_counts) > 1 else 0.0,
                "runtime_sec_mean": round(stats.mean(runtimes), 6),
                "runtime_sec_sd": round(stats.stdev(runtimes), 6) if len(runtimes) > 1 else 0.0,
                "peak_python_mem_mb_mean": round(stats.mean(peaks), 3),
                "peak_python_mem_mb_sd": round(stats.stdev(peaks), 3) if len(peaks) > 1 else 0.0,
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run targeted methodology probes")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip gradual-dominance ablation step")
    parser.add_argument("--skip-multiseed", action="store_true", help="Skip multi-seed stability step")
    parser.add_argument("--skip-scalability", action="store_true", help="Skip scalability probe step")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not args.skip_ablation:
        print("[1/3] Running gradual-dominance ablation...")
        ablation_rows = run_ablation_gradual()
        _write_csv(
            OUT_ROOT / "ablation_gradual.csv",
            ablation_rows,
            [
                "config",
                "dataset",
                "instances",
                "episodes",
                "gradual_count",
                "sudden_count",
                "unknown_count",
                "gradual_share_pct",
                "sudden_share_pct",
                "unknown_share_pct",
                "dominant_majority",
                "dominant_weighted",
                "runtime_sec",
                "peak_python_mem_mb",
                "run_dir",
            ],
        )

    if not args.skip_multiseed:
        print("[2/3] Running multi-seed stability suite...")
        ms_rows = run_multiseed()
        _write_csv(
            OUT_ROOT / "multiseed_runs.csv",
            ms_rows,
            [
                "dataset",
                "seed",
                "instances",
                "episodes",
                "drift_rate_pct",
                "gradual_count",
                "sudden_count",
                "unknown_count",
                "gradual_share_pct",
                "dominant_majority",
                "dominant_weighted",
                "runtime_sec",
                "peak_python_mem_mb",
                "delta_center_abs",
                "delta_center_signed",
                "delay_start_after_drift",
                "hits_within_200",
                "hits_within_500",
                "run_dir",
            ],
        )

        ms_summary = summarize_multiseed(ms_rows)
        _write_csv(
            OUT_ROOT / "multiseed_summary.csv",
            ms_summary,
            [
                "dataset",
                "seeds",
                "episodes_mean_sd",
                "gradual_share_pct_mean_sd",
                "runtime_sec_mean_sd",
                "peak_python_mem_mb_mean_sd",
                "delta_center_abs_mean_sd",
                "delay_start_mean_sd",
            ],
        )

    if not args.skip_scalability:
        print("[3/3] Running scalability probe...")
        scale_rows = run_scalability_probe()
        _write_csv(
            OUT_ROOT / "scalability_probe.csv",
            scale_rows,
            [
                "dataset",
                "n_features",
                "rows_per_window",
                "repeats",
                "concept_count_mean",
                "concept_count_sd",
                "runtime_sec_mean",
                "runtime_sec_sd",
                "peak_python_mem_mb_mean",
                "peak_python_mem_mb_sd",
            ],
        )

    print(f"[DONE] Methodology suite outputs saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
