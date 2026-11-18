"""
Command-line interface for FCA drift detection.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from fca_drift.core import (
    StreamReader,
    StreamWrapper,
    SlidingWindow,
    DataPreprocessor,
    get_available_datasets,
)
from fca_drift.fca import build_formal_context, ConceptLattice
from fca_drift.detection import FCADriftDetector


def _parse_drift_positions(raw: Optional[str], fallback: Optional[int]) -> List[int]:
    if raw is None or str(raw).strip() == "":
        return [int(fallback)] if fallback is not None else []
    parts = re.split(r"[,;\s]+", str(raw).strip())
    values = []
    for part in parts:
        if not part:
            continue
        values.append(int(part))
    return sorted(set(values))


def _encode_target(y, label_map: Dict[str, int]) -> float:
    if y is None:
        return 0.0
    if isinstance(y, (bool, int, float, np.integer, np.floating)):
        return float(y)
    key = str(y)
    if key not in label_map:
        label_map[key] = len(label_map)
    return float(label_map[key])


def run_detection(args: argparse.Namespace) -> Dict:
    random.seed(args.seed)
    np.random.seed(args.seed)

    drift_positions = _parse_drift_positions(args.drift_positions, args.drift_position)

    if args.list_datasets:
        print("Available datasets:")
        for name in get_available_datasets():
            print(f"- {name}")
        return {}

    reader_kwargs = {}
    if drift_positions:
        reader_kwargs["drift_positions"] = drift_positions
    if args.n_features is not None:
        reader_kwargs["n_features"] = args.n_features

    reader = StreamReader(args.dataset, seed=args.seed, **reader_kwargs)
    stream = StreamWrapper(reader.stream, max_instances=args.max_instances)

    window = SlidingWindow(size=args.window_size)
    preprocessor = DataPreprocessor(
        threshold=args.threshold,
        adaptive_threshold=args.adaptive_threshold,
        freeze_after=args.preprocess_freeze_after,
        ema_alpha=args.preprocess_ema_alpha,
    )
    detector = FCADriftDetector(theta=args.theta, alpha=args.alpha, window_size=args.window_size)

    instance_count = 0
    full_windows_seen = 0
    label_map: Dict[str, int] = {}

    for idx, (x, y) in enumerate(stream):
        x_for_window = dict(x)
        if args.supervised:
            x_for_window["__target__"] = _encode_target(y, label_map)

        window.append(x_for_window)
        instance_count += 1

        if window.is_full():
            full_windows_seen += 1
            if args.window_step > 1 and (full_windows_seen - 1) % args.window_step != 0:
                continue

            binary = preprocessor.preprocess_window(window.get_data())
            if binary.ndim == 2 and args.max_fca_attrs > 0 and binary.shape[1] > args.max_fca_attrs:
                col_var = binary.var(axis=0)
                top_cols = col_var.argsort()[-args.max_fca_attrs:]
                binary = binary[:, top_cols]

            context = build_formal_context(binary)
            lattice = ConceptLattice()
            lattice.build_from_context(context, max_objects=args.max_objects)
            detector.update(lattice, idx)

    drift_count = len(detector.drift_indices)
    drift_rate = (drift_count / instance_count * 100.0) if instance_count else 0.0

    results = {
        "parameters": {
            "dataset": args.dataset,
            "max_instances": args.max_instances,
            "window_size": args.window_size,
            "window_step": args.window_step,
            "theta": args.theta,
            "alpha": args.alpha,
            "seed": args.seed,
            "drift_position": args.drift_position,
            "drift_positions": drift_positions,
            "adaptive_threshold": args.adaptive_threshold,
            "preprocess_freeze_after": args.preprocess_freeze_after,
            "preprocess_ema_alpha": args.preprocess_ema_alpha,
            "threshold": args.threshold,
            "supervised": args.supervised,
            "n_features": args.n_features,
            "max_fca_attrs": args.max_fca_attrs,
            "max_objects": args.max_objects,
        },
        "detection_results": {
            "total_instances": instance_count,
            "drifts_detected": drift_count,
            "drift_rate_pct": round(drift_rate, 3),
            "drift_indices": detector.drift_indices,
            "avg_delta_L": round(float(np.mean(detector.delta_L_history)), 6)
            if detector.delta_L_history
            else 0.0,
            "max_delta_L": round(float(np.max(detector.delta_L_history)), 6)
            if detector.delta_L_history
            else 0.0,
        },
    }

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[OK] Results saved to: {output_path}")

    print(
        f"[OK] Done. Instances={instance_count}, drifts={drift_count}, "
        f"drift_rate={drift_rate:.2f}%"
    )

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FCA drift detection CLI")
    parser.add_argument("--dataset", type=str, default="agrawal", help="Dataset name")
    parser.add_argument("--list-datasets", action="store_true", help="Print available datasets")
    parser.add_argument("--max-instances", type=int, default=10000, help="Max instances from stream")
    parser.add_argument("--window-size", type=int, default=300, help="Sliding window size")
    parser.add_argument("--window-step", type=int, default=1, help="Process every N windows")
    parser.add_argument("--theta", type=float, default=0.4, help="Base threshold theta")
    parser.add_argument("--alpha", type=float, default=2.0, help="Adaptive threshold scale alpha")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--drift-position", type=int, default=5000, help="Single drift position")
    parser.add_argument(
        "--drift-positions",
        type=str,
        default=None,
        help="Comma/space-separated drift positions",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Fixed threshold for binarization")
    parser.add_argument(
        "--adaptive-threshold",
        action="store_true",
        default=True,
        help="Enable adaptive thresholds (default)",
    )
    parser.add_argument(
        "--no-adaptive-threshold",
        action="store_false",
        dest="adaptive_threshold",
        help="Disable adaptive thresholds",
    )
    parser.add_argument(
        "--preprocess-freeze-after",
        type=int,
        default=2500,
        help="Windows count before freezing thresholds",
    )
    parser.add_argument(
        "--preprocess-ema-alpha",
        type=float,
        default=0.1,
        help="EMA alpha for adaptive thresholds",
    )
    parser.add_argument(
        "--supervised",
        action="store_true",
        help="Append target label as __target__ attribute",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=None,
        help="Number of features for synthetic datasets",
    )
    parser.add_argument(
        "--max-fca-attrs",
        type=int,
        default=0,
        help="Limit FCA attributes by variance (0 = no limit)",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=50,
        help="Max objects used to build a lattice",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results.json",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    results = run_detection(args)
    return 0 if results is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
