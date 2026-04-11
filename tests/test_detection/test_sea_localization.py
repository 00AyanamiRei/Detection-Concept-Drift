"""SEA integration checks for drift localization.

These tests are intentionally marked as slow because FCA lattice construction is
computationally heavy. Enable with:
    RUN_SLOW_TESTS=1 pytest tests/test_detection/test_sea_localization.py
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fca_drift.core import DataPreprocessor, SlidingWindow, StreamReader, StreamWrapper
from fca_drift.detection import FCADriftDetector
from fca_drift.fca import ConceptLattice, build_formal_context
from fca_drift.utils.drift_aggregator_v2 import DriftAggregatorV2


def _run_sea_pipeline(context_mode: str, max_instances: int = 6500):
    reader = StreamReader("sea", seed=42, drift_position=5000)
    stream = StreamWrapper(reader.stream, max_instances=max_instances)

    window_size = 50
    window = SlidingWindow(size=window_size)
    preprocessor = DataPreprocessor(threshold=0.5, adaptive_threshold=True)

    detector = FCADriftDetector(
        theta=0.5,
        alpha=1.5,
        window_size=window_size,
        adaptive_window=10,
        min_persistence_windows=2,
        cooldown_windows=max(window_size // 2, 5),
        spike_multiplier=1.5,
    )

    for idx, (x, y) in enumerate(stream):
        sample = dict(x)
        if context_mode == "supervised_mode":
            sample["__target__"] = float(y)

        window.append(sample)
        if not window.is_full():
            continue

        binary_data = preprocessor.preprocess_window(window.get_data())
        context = build_formal_context(binary_data)
        lattice = ConceptLattice()
        lattice.build_from_context(context)
        detector.update(lattice, idx)

    aggregator = DriftAggregatorV2(
        delta_L_history=detector.delta_L_history,
        similarity_history=detector.similarity_history,
        drift_signal_indices=detector.drift_signal_indices,
        signal_to_instance_id=detector.signal_to_instance_id,
        drift_types_raw={
            int(event.signal_idx): str(event.drift_type).strip().lower()
            for event in detector.drift_events
            if event.signal_idx is not None
        },
        window_size=window_size,
        warm_up_windows=5,
        alpha=1.5,
        n_adapt=10,
        merge_gap=window_size,
        cooldown=max(int(0.3 * window_size), 1),
        intent_diffs=detector.intent_diffs,
    )

    return detector, aggregator, aggregator.get_merged_episodes()


@pytest.mark.skipif(os.getenv("RUN_SLOW_TESTS") != "1", reason="Slow SEA integration test")
def test_sea_unsupervised_has_few_detections():
    _, aggregator, episodes = _run_sea_pipeline("unsupervised_mode")
    assert len(episodes) <= 10
    assert aggregator.get_type_counts().get("sudden", 0) <= len(episodes)


@pytest.mark.skipif(os.getenv("RUN_SLOW_TESTS") != "1", reason="Slow SEA integration test")
def test_sea_supervised_localizes_near_drift_and_not_incremental_dominated():
    _, aggregator, episodes = _run_sea_pipeline("supervised_mode")

    assert len(episodes) > 0

    centers = [ep.center_instance_id for ep in episodes]
    assert any(abs(center - 5000) <= 600 for center in centers)

    counts = aggregator.get_type_counts()
    assert counts.get("sudden", 0) >= counts.get("incremental", 0)
